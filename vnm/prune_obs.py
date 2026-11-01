"""Second-order (OBS / SparseGPT-style) V:N:M pruning of conv layers, with layer-wise weight compensation.

For each conv, the Hessian of the layer's output reconstruction error is H = 2/N * sum_x x x^T over im2col patches x of
calibration images. Columns of the (C_out x kh*kw*C_in) weight matrix are processed left to right in groups of M:
  1. at the start of a group the V:N:M mask is chosen with the OBS saliency w^2 / [H^-1]_cc^2 (summed over the V rows of
     a block to pick the 4 columns, then per row to pick the 2 weights inside those columns),
  2. each column is then pruned and the error is pushed onto the not-yet-processed columns via H^-1 (the OBS update).
Layers are pruned in module order and each layer's Hessian is collected through the already-pruned layers before it.

    python vnm/prune_obs.py --run venom_cifar100_res --bp 482 --vs 8,32 --ms 8,16     # zero-shot comparison vs magnitude
"""
import argparse
import json
import os

import recipe  # noqa: F401  (sets cwd / sys.path)
import torch
import torch.nn.functional as F
from recipe import Data, setup_config
from configuration import config
from prune_eval import named_convs
from vnm_pack import conv_from_2d, conv_to_2d, pack_vnm_2d, packed_nbytes_tight


def _unfold_khwc(x, conv):
    """im2col patches with the column order (kh, kw, C_in) that conv_to_2d uses -> (B*L, K)."""
    kh, kw = conv.kernel_size
    p = F.unfold(x, (kh, kw), dilation=conv.dilation, padding=conv.padding, stride=conv.stride)  # B, C*kh*kw, L
    b, _, l = p.shape
    p = p.view(b, x.shape[1], kh * kw, l).permute(0, 3, 2, 1)  # B, L, kh*kw, C
    return p.reshape(b * l, kh * kw * x.shape[1])


@torch.no_grad()
def collect_hessian(model, conv, calib, k, bs=128, chunk=16):
    """H = 2/N sum x x^T for the inputs of `conv` (model runs in eval mode, fp32)."""
    h = torch.zeros(k, k, device=calib.device)
    n = [0]

    def hook(_, inp):
        x = inp[0].float()
        for i in range(0, x.shape[0], chunk):
            p = _unfold_khwc(x[i:i + chunk], conv)
            h.addmm_(p.T, p)
            n[0] += p.shape[0]

    handle = conv.register_forward_pre_hook(hook)
    for i in range(0, calib.shape[0], bs):
        model(calib[i:i + bs])
    handle.remove()
    return h * (2.0 / max(n[0], 1))


@torch.no_grad()
def obs_prune_2d(w, hess, v, m, n=2, damp=0.01, block=128, kcols=4):
    """Returns (pruned weights, bool mask), both R x K. Requires m <= block and block % m == 0.
    kcols columns are kept per V x M block, then n weights per row inside them (n == kcols keeps every weight of
    the kept columns: the column-only intermediate steps of the gradual schedule)."""
    r, k = w.shape
    kp = -(-k // m) * m
    w = F.pad(w.float(), (0, kp - k))
    h = torch.zeros(kp, kp, device=w.device)
    h[:k, :k] = hess
    if kp > k:  # padded columns: unit-scale diagonal so they stay inert
        idx = torch.arange(k, kp, device=w.device)
        h[idx, idx] = hess.diagonal().mean().clamp_min(1e-8)
    dead = h.diagonal() == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h.diagonal().add_(damp * h.diagonal().mean())
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)

    rg = -(-r // v)
    q = torch.zeros_like(w)
    mask = torch.zeros_like(w, dtype=torch.bool)
    block = max(block // m, 1) * m
    for i1 in range(0, kp, block):
        i2 = min(i1 + block, kp)
        w1, hinv1 = w[:, i1:i2].clone(), hinv[i1:i2, i1:i2]
        err1 = torch.zeros_like(w1)
        keep1 = torch.zeros_like(w1, dtype=torch.bool)
        for i in range(i2 - i1):
            if i % m == 0:  # choose the V:N:M mask for this group of m columns from the current weights
                hd = hinv1.diagonal()[i:i + m]
                score = w1[:, i:i + m] ** 2 / hd ** 2
                grp = F.pad(score, (0, 0, 0, rg * v - r)).view(rg, v, m).sum(1)
                top4 = grp.topk(min(kcols, m), dim=1).indices
                colmask = torch.zeros(rg, m, dtype=torch.bool, device=w.device).scatter_(1, top4, True)
                colmask = colmask[:, None, :].expand(rg, v, m).reshape(rg * v, m)[:r]
                sel = score.masked_fill(~colmask, -1.0)
                keep1[:, i:i + m] = torch.zeros_like(colmask).scatter_(1, sel.topk(n, dim=1).indices, True) & colmask
            col, d = w1[:, i], hinv1[i, i]
            qc = col * keep1[:, i]
            q[:, i1 + i] = qc
            e = (col - qc) / d
            w1[:, i:] -= e[:, None] * hinv1[i, i:][None, :]
            err1[:, i] = e
        mask[:, i1:i2] = keep1
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return q[:, :k], mask[:, :k]


@torch.no_grad()
def obs_prune_model(model, calib, v, m, damp=0.01, kcols=4, n=2):
    """In-place sequential OBS V:N:M pruning of every conv. Returns (masks, dense_bytes, sparse_bytes).
    sparse_bytes is only meaningful for the final (kcols=4, n=2) step."""
    model.eval()
    masks, dense_b, sparse_b = {}, 0, 0
    for name, conv in named_convs(model):
        c_out, c_in, kh, kw = conv.weight.shape
        w2d = conv_to_2d(conv.weight.data)
        hess = collect_hessian(model, conv, calib, w2d.shape[1])
        pruned, mask = obs_prune_2d(w2d, hess, v, m, n=n, damp=damp, kcols=kcols)
        pruned = pruned * mask
        dense_b += conv.weight.numel() * 4
        if kcols == 4 and n == 2:
            sparse_b += packed_nbytes_tight(pack_vnm_2d(pruned, v=v, m=m, n=2))
        conv.weight.data.copy_(conv_from_2d(pruned, c_out, c_in, kh, kw))
        masks[name] = conv_from_2d(mask.float(), c_out, c_in, kh, kw)
    return masks, dense_b, sparse_b


def calib_batch(data, n=1024, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(data.xtr.shape[0], generator=g)[:n]
    return data._prep(data.xtr[idx.to(data.xtr.device)], False)


def main():
    from src.analysis.run import get_run
    from src.phenotype.neural_network.evaluator.fully_train import _create_model
    from prune_eval import apply_vnm_to_model
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="venom_cifar100_res")
    p.add_argument("--bp", type=int, required=True)
    p.add_argument("--src", default="retrained_v2")
    p.add_argument("--vs", default="8,32,128")
    p.add_argument("--ms", default="4,8,16,32")
    p.add_argument("--nsamples", type=int, default=1024)
    p.add_argument("--damp", type=float, default=0.01)
    args = p.parse_args()
    setup_config(args.run, "cifar100")
    device = config.get_device()
    data = Data("cifar100", device)
    run = get_run(args.run)
    bpgen = {}
    for bp, gen in run.get_most_accurate_blueprints(200):
        bpgen.setdefault(bp.id, (bp, gen))
    sd = torch.load(os.path.join("runs", args.run, args.src, f"bp-{args.bp}.pt"), map_location=device)
    calib = calib_batch(data, args.nsamples)

    def fresh():
        model = _create_model(run, *bpgen[args.bp], data.input_shape, 1)
        model.load_state_dict(sd)
        return model.to(device).eval()

    print("dense", data.test_acc(fresh()), flush=True)
    for v in map(int, args.vs.split(",")):
        for m in map(int, args.ms.split(",")):
            a = fresh()
            apply_vnm_to_model(a, v=v, m=m)
            acc_mag = data.test_acc(a)
            b = fresh()
            obs_prune_model(b, calib, v, m, damp=args.damp)
            acc_obs = data.test_acc(b)
            print(json.dumps(dict(bp=args.bp, v=v, m=m, zero_shot_magnitude=acc_mag, zero_shot_obs=acc_obs)), flush=True)


if __name__ == "__main__":
    main()
