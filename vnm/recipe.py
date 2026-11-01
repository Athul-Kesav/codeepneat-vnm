"""Shared helpers for retraining evolved CoDeepNEAT architectures with a modern recipe and for V:N:M fine-tuning.

Data stays on the host as uint8 arrays; crops/flips/normalisation happen on the GPU.
Recipe: AdamW + warmup/cosine, label smoothing, random crop + flip, bf16 autocast, no ReLU on the logits.
"""

from __future__ import annotations

import math
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "CoDeepNEAT"))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "runs"), os.path.join(ROOT, "configuration")]

import numpy as np
import torch
import torch.nn.functional as F

from configuration import config
from configuration.config_loader import load_simple_config

CACHE = os.path.join(os.environ.get("VENOM_BUILD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "venom_build")), "cache", "in100")
DATASETS = os.path.join(ROOT, "data", "Datasets")

# dataset -> (input hw, train crop, pad, classes)
SPECS = {
    "cifar10": dict(hw=32, crop=32, pad=4, classes=10),
    "cifar100": dict(hw=32, crop=32, pad=4, classes=100),
    "imagenet100": dict(hw=144, crop=128, pad=0, classes=100),
}


def setup_config(config_name: str, dataset: str):
    load_simple_config(config_name, lambda: None, lambda: None, ngpus=1, use_wandb_override=False)
    config.fully_train = True
    config.use_wandb = False
    config.evolve_da = False
    config.final_relu = False
    config.max_model_params = 1e9
    spec = SPECS[dataset]
    config.image_size = spec["crop"] if dataset == "imagenet100" else 0
    config.num_output_classes = spec["classes"]
    return spec


class Data:
    """uint8 NCHW arrays + labels with GPU-side augmentation."""

    def __init__(self, dataset: str, device):
        self.device, self.spec = device, SPECS[dataset]
        if dataset in ("cifar10", "cifar100"):
            from torchvision.datasets import CIFAR10, CIFAR100
            ds = CIFAR10 if dataset == "cifar10" else CIFAR100
            tr = ds(DATASETS, train=True, download=True)
            te = ds(DATASETS, train=False, download=True)
            self.xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous()
            self.ytr = torch.tensor(tr.targets)
            self.xte = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous()
            self.yte = torch.tensor(te.targets)
        else:
            self.xtr = torch.from_numpy(np.load(f"{CACHE}/train_x.npy", mmap_mode="r")[:]).permute(0, 3, 1, 2)
            self.ytr = torch.from_numpy(np.load(f"{CACHE}/train_y.npy"))
            self.xte = torch.from_numpy(np.load(f"{CACHE}/test_x.npy")).permute(0, 3, 1, 2)
            self.yte = torch.from_numpy(np.load(f"{CACHE}/test_y.npy"))
        if dataset in ("cifar10", "cifar100"):  # small enough to live on the GPU
            self.xtr, self.ytr = self.xtr.to(device), self.ytr.to(device)
        self.on_gpu = self.xtr.is_cuda

    @property
    def input_shape(self):
        c = self.spec["crop"]
        return [2, 3, c, c]

    def _prep(self, xb, train):
        xb = xb.to(self.device, non_blocking=True).float().div_(255)
        if train:
            b, crop, pad = xb.shape[0], self.spec["crop"], self.spec["pad"]
            if pad:
                xb = F.pad(xb, (pad,) * 4, mode="reflect")
            hw = xb.shape[-1]
            if hw > crop:
                oy = torch.randint(0, hw - crop + 1, (b,), device=xb.device)
                ox = torch.randint(0, hw - crop + 1, (b,), device=xb.device)
                ar = torch.arange(crop, device=xb.device)
                rows, cols = (oy[:, None] + ar)[:, None, :, None], (ox[:, None] + ar)[:, None, None, :]
                xb = xb[torch.arange(b, device=xb.device)[:, None, None, None],
                        torch.arange(3, device=xb.device)[None, :, None, None], rows, cols]
            flip = torch.rand(b, device=xb.device) < 0.5
            xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
        return ((xb - 0.5) / 0.5).contiguous()

    def train_batches(self, bs):
        n = self.xtr.shape[0]
        perm = torch.randperm(n, device=self.xtr.device if self.on_gpu else "cpu")
        for i in range(0, n - bs + 1, bs):  # drop the last partial batch
            idx = perm[i:i + bs]
            yield self._prep(self.xtr[idx], True), self.ytr[idx].to(self.device, non_blocking=True)

    def n_train_batches(self, bs):
        return self.xtr.shape[0] // bs

    @torch.no_grad()
    def test_acc(self, model, bs=500):
        model.eval()
        correct = 0
        for i in range(0, self.xte.shape[0], bs):
            xb = self._prep(self.xte[i:i + bs], False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(xb)
            correct += (out.float().argmax(1).cpu() == self.yte[i:i + bs]).sum().item()
        return correct / self.xte.shape[0]


def smooth_nll(logp, y, eps):
    nll = -logp.gather(1, y[:, None]).squeeze(1)
    return ((1 - eps) * nll + eps * (-logp.mean(1))).mean()


def train(model, data: Data, epochs, lr, wd=0.05, bs=128, smoothing=0.1, warmup=3, teacher=None, kd_alpha=0.5,
          kd_temp=4.0, masks=None, log=print, eval_every=10, cutmix=0.0, ema=0.0):
    """Train `model` in place. `masks` (name->mask for conv weights) are re-applied after every optimizer step."""
    device = next(model.parameters()).device
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=wd), dict(params=no_decay, weight_decay=0.0)], lr=lr)
    steps_per = data.n_train_batches(bs)
    total, warm = epochs * steps_per, warmup * steps_per

    def lr_at(step):
        if step < warm:
            return lr * (step + 1) / warm
        return 0.5 * lr * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    conv_params = None
    if masks is not None:
        by_name = dict(model.named_modules())
        conv_params = [(by_name[n].weight, m.to(device)) for n, m in masks.items()]
    step, t0, best = 0, time.time(), 0.0
    ema_p = [p.detach().clone() for p in model.state_dict().values()] if ema > 0 else None
    for epoch in range(epochs):
        model.train()
        run_loss = 0.0
        for xb, yb in data.train_batches(bs):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            lam, yb2 = 1.0, None
            if cutmix > 0 and np.random.rand() < cutmix:  # CutMix, alpha=1
                lam = np.random.beta(1.0, 1.0)
                perm_i = torch.randperm(xb.shape[0], device=xb.device)
                h, w_ = xb.shape[-2:]
                ch, cw = int(h * math.sqrt(1 - lam)), int(w_ * math.sqrt(1 - lam))
                cy, cx = np.random.randint(h), np.random.randint(w_)
                y0, y1, x0, x1 = max(cy - ch // 2, 0), min(cy + ch // 2, h), max(cx - cw // 2, 0), min(cx + cw // 2, w_)
                xb = xb.clone()
                xb[:, :, y0:y1, x0:x1] = xb[perm_i][:, :, y0:y1, x0:x1]
                lam = 1 - (y1 - y0) * (x1 - x0) / (h * w_)
                yb2 = yb[perm_i]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logp = model(xb)
                if teacher is not None:
                    with torch.no_grad():
                        tlogp = teacher(xb)
            logp = logp.float()
            loss = smooth_nll(logp, yb, smoothing)
            if yb2 is not None:
                loss = lam * loss + (1 - lam) * smooth_nll(logp, yb2, smoothing)
            if teacher is not None:
                t = kd_temp
                kd = F.kl_div(F.log_softmax(logp / t, 1), F.softmax(tlogp.float() / t, 1), reduction="batchmean") * t * t
                loss = (1 - kd_alpha) * loss + kd_alpha * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if conv_params is not None:
                with torch.no_grad():
                    for w, m in conv_params:
                        w.mul_(m)
            if ema_p is not None:
                with torch.no_grad():
                    for e, p_ in zip(ema_p, model.state_dict().values()):
                        if p_.dtype.is_floating_point:
                            e.mul_(ema).add_(p_.detach(), alpha=1 - ema)
                        else:
                            e.copy_(p_)
            run_loss += loss.item() if step % 50 == 0 else 0
            step += 1
        msg = f"epoch {epoch + 1}/{epochs} loss~{run_loss / max(1, steps_per // 50):.3f} lr={lr_at(step - 1):.2e} t={time.time() - t0:.0f}s"
        if (epoch + 1) % eval_every == 0 or epoch + 1 == epochs:
            acc = data.test_acc(model)
            best = max(best, acc)
            msg += f" test_acc={acc * 100:.2f}"
        log(msg)
    if ema_p is not None:  # evaluate with EMA weights; keep them if better
        live = {k: v.clone() for k, v in model.state_dict().items()}
        acc_live = data.test_acc(model)
        model.load_state_dict(dict(zip(live.keys(), ema_p)))
        acc_ema = data.test_acc(model)
        log(f"final live={acc_live * 100:.2f} ema={acc_ema * 100:.2f}")
        if acc_live > acc_ema:
            model.load_state_dict(live)
        return max(acc_live, acc_ema), max(best, acc_live, acc_ema)
    return data.test_acc(model), best
