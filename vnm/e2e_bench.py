"""End-to-end inference timing of V:N:M-pruned networks with Spatha convolutions (VENOM env only).

Every Conv2d is replaced by SpathaConv2d: im2col (F.unfold, reordered to the (kh, kw, C_in) column order of the V:N:M
layout) -> spatha.spmm -> NHWC view of the output. Everything else (BN, ReLU, pooling, aggregation, head) runs as usual
in fp16. A whole forward pass is timed against:
  dense_cudnn   the dense fp16 model, channels_last, cudnn.benchmark (what one would deploy)
  dense_im2col  the same im2col path with a dense fp16 GEMM (isolates the kernel from the im2col cost)
  hybrid        per conv, Spatha or cuDNN, whichever was faster for that layer in isolation
The three variants are timed in interleaved rounds (median reported) because the GPUs are shared.
Weights are magnitude V:N:M-pruned on the fly: kernel time depends on the sparsity pattern, not the weight values.
Correctness: logits of the Spatha model vs the same masked weights run through cuDNN, on real test images.

    source scripts/venom_env.sh
    SPATHA_LIB=$VENOM_BUILD/spatha_v32_lib python vnm/e2e_bench.py --v 32 --out results/end_to_end/e2e.jsonl
"""
import argparse
import copy
import json
import math
import os
import pickle
import statistics
import sys

import venom_compat

venom_compat.install(["wandb", "PIL", "graphviz", "tarjan", "imgaug", "cv2", "matplotlib", "torchvision", "sklearn"])

import torch  # noqa: E402  (must precede spatha: it links against libc10)

VENOM = os.path.join(os.environ.get("VENOM_BUILD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "venom_build")), "venom", "end2end")
sys.path.insert(0, VENOM)
if os.environ.get("SPATHA_LIB"):
    sys.path.insert(0, os.environ["SPATHA_LIB"])
import spatha  # noqa: E402
from grouped_nmv_tensor import SrNMTensor, nm_vector_mask_sparsify  # noqa: E402

import numpy as np  # noqa: E402
import recipe  # noqa: E402,F401  (sets cwd / sys.path)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from recipe import setup_config  # noqa: E402
from configuration import config  # noqa: E402
from vnm_pack import conv_from_2d, conv_to_2d, prune_vnm_2d  # noqa: E402

KPAD, CPAD = 128, 64
IM2COL = "taps"
P = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = {  # name -> (run, dataset, source, kind)
    "bp493": ("venom_cifar100_res", "cifar100", "retrained_v2", "evo"),
    "bp482": ("venom_cifar100_res", "cifar100", "retrained_v2", "evo"),
    "bp741": ("venom_cifar100_res", "cifar100", "retrained_v2", "evo"),
    "bp430": ("venom_cifar100", "cifar100", "retrained_v2", "evo"),
    "bp427": ("venom_cifar100", "cifar100", "retrained_v2", "evo"),
    "bp499": ("venom_cifar100", "cifar100", "retrained_v2", "evo"),
    "r18": ("venom_cifar100", "cifar100", "resnet_baselines", "resnet"),
    "r20w58": ("venom_cifar100", "cifar100", "resnet_baselines", "resnet"),
    "r56w32": ("venom_cifar100", "cifar100", "resnet_baselines", "resnet"),
    "bp207": ("imagenet100", "imagenet100", "retrained", "evo"),
    "bp182": ("imagenet100", "imagenet100", "retrained", "evo"),
    "bp230": ("imagenet100", "imagenet100", "retrained", "evo"),
    # CIFAR-10 strong search: CoDeepNEAT full-train weights at feature multiplier 3, legacy ReLU on the logits
    "bp1485": ("ada_cifar10_strong", "cifar10", "dense_fm3", "evo", 3, True),
}


def ru(x, y):
    return int(math.ceil(x / y) * y)


def cuda_time(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


class SpathaConv2d(nn.Module):
    """Conv2d as im2col + Spatha V:N:M SpMM. `mode="dense"` runs the same path with a dense GEMM instead."""

    def __init__(self, conv: nn.Conv2d, v, m, n=2):
        super().__init__()
        c_out, c_in, kh, kw = conv.weight.shape
        self.geo = (c_out, c_in, kh, kw, conv.stride, conv.padding, conv.dilation)
        w2d = conv_to_2d(conv.weight.data.float().cpu())
        self.r, self.k = w2d.shape
        self.r_pad, self.k_pad = ru(self.r, v), ru(self.k, max(KPAD, 8 * m))  # M=32 needs K % 256 == 0 (else illegal access)
        w = torch.zeros(self.r_pad, self.k_pad)
        w[: self.r, : self.k] = w2d
        g = torch.Generator().manual_seed(0)  # break ties between the already-pruned zeros
        mask, cols = nm_vector_mask_sparsify(w + 1e-7 * torch.randn(w.shape, generator=g), n, m, v)
        self.sp = SrNMTensor(n, m, v, w, mask, cols, torch.device("cuda"))
        self.v, self.m, self.n = v, m, n
        self.w_eff = (w * mask)[: self.r, : self.k].clone()  # weights the kernel really uses
        self.w_dense = self.w_eff.cuda().half()
        self.bias = None if conv.bias is None else conv.bias.data.cuda().half().view(1, -1, 1, 1)
        self.zero_bias = torch.zeros(self.r_pad, device="cuda").half()
        self.buf = {}
        self.mode = "spatha"

    def _cols(self, x):
        c_out, c_in, kh, kw, st, pd, dl = self.geo
        b, _, h, w_ = x.shape
        ho = (h + 2 * pd[0] - dl[0] * (kh - 1) - 1) // st[0] + 1
        wo = (w_ + 2 * pd[1] - dl[1] * (kw - 1) - 1) // st[1] + 1
        ncol = b * ho * wo
        key = (b, h, w_)
        if key not in self.buf:  # padded rows/cols stay zero
            self.buf[key] = torch.zeros(self.k_pad, ru(ncol, CPAD), device="cuda", dtype=torch.half)
        xb = self.buf[key]
        if IM2COL == "unfold":
            p = F.unfold(x, (kh, kw), dilation=dl, padding=pd, stride=st)  # B, C*kh*kw, L  (column order C, kh, kw)
            xb[: self.k, :ncol].view(kh * kw, c_in, b, ho * wo).copy_(p.view(b, c_in, kh * kw, ho * wo).permute(2, 1, 0, 3))
        else:  # one strided copy per kernel tap straight into the (kh, kw, C) x (B, H_out, W_out) layout
            xp = F.pad(x, (pd[1], pd[1], pd[0], pd[0])) if pd[0] or pd[1] else x
            dst = xb[: self.k, :ncol].view(kh * kw, c_in, b, ho, wo)
            for i in range(kh):
                for j in range(kw):
                    src = xp[:, :, i * dl[0]: i * dl[0] + st[0] * (ho - 1) + 1: st[0],
                             j * dl[1]: j * dl[1] + st[1] * (wo - 1) + 1: st[1]]
                    dst[i * kw + j].copy_(src.permute(1, 0, 2, 3))
        return xb, b, ho, wo, ncol

    def forward(self, x):
        xb, b, ho, wo, ncol = self._cols(x)
        if self.mode == "spatha":
            out = spatha.spmm(self.sp.metadata, self.sp.columns, self.sp.values, xb, self.zero_bias, self.r_pad,
                              self.k_pad, xb.shape[1], self.v, self.n, self.m, self.sp.nnz, 0, 32, 4)  # C_pad x R_pad
            y = out[:ncol, : self.r].view(b, ho, wo, self.r).permute(0, 3, 1, 2)
        else:
            y = (xb[: self.k, :ncol].t() @ self.w_dense.t()).view(b, ho, wo, self.r).permute(0, 3, 1, 2)
        return y if self.bias is None else y + self.bias


class Fixed(nn.Module):
    """Runs a shared SpathaConv2d in a fixed mode (lets one layer object serve several model variants)."""

    def __init__(self, layer, mode):
        super().__init__()
        self.layer, self.md = layer, mode

    def forward(self, x):
        self.layer.mode = self.md
        return self.layer(x)


def swap_convs(model, fn):
    for name, mod in list(model.named_modules()):
        for cn, child in list(mod.named_children()):
            if isinstance(child, nn.Conv2d):
                setattr(mod, cn, fn(f"{name}.{cn}" if name else cn, child))


def build(name):
    run, ds, src, kind, *extra = MODELS[name]
    fm, final_relu = extra if extra else (1, False)
    setup_config(run, ds)
    config.final_relu = final_relu
    if kind == "resnet":
        from train_resnet import ResNet, SPECS
        model = ResNet(*SPECS[name])
        sd = torch.load(f"runs/{run}/{src}/{name}.pt", map_location="cpu")
    else:
        from src.analysis.run import get_run
        from src.phenotype.neural_network.evaluator.fully_train import _create_model
        r = get_run(run)
        bps = {}
        for bp, gen in r.get_most_accurate_blueprints(300):
            bps.setdefault(bp.id, (bp, gen))
        hw = 128 if ds == "imagenet100" else 32
        model = _create_model(r, *bps[int(name[2:])], [2, 3, hw, hw], fm)
        sd = torch.load(f"runs/{run}/{src}/bp-{name[2:]}.pt", map_location="cpu")
    model.load_state_dict(sd)
    return model.eval(), ds


def test_images(ds, n=256):
    if ds in ("cifar10", "cifar100"):
        f = "cifar-10-batches-py/test_batch" if ds == "cifar10" else "cifar-100-python/test"
        d = pickle.load(open(f"{P}/CoDeepNEAT/data/Datasets/{f}", "rb"), encoding="latin1")
        x = torch.from_numpy(d["data"][:n].reshape(n, 3, 32, 32))
    else:
        x = torch.from_numpy(np.load(os.path.join(os.environ.get("VENOM_BUILD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "venom_build")), "cache", "in100", "test_x.npy"))[:n]).permute(0, 3, 1, 2)
    return ((x.float() / 255 - 0.5) / 0.5).cuda().half()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=int, required=True)
    ap.add_argument("--ms", default="4,8,16,32")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--batches", default="128,1")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--im2col", default="taps", choices=["taps", "unfold"])
    ap.add_argument("--out", default=f"{P}/results/end_to_end/e2e.jsonl")
    args = ap.parse_args()
    global IM2COL
    IM2COL = args.im2col
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.ones(64, 64, device="cuda").half() @ torch.ones(64, 64, device="cuda").half()  # create the cuBLAS handle before Spatha runs
    done = set()
    if os.path.exists(args.out):
        done = {(r["model"], r["v"], r["m"], r["batch"]) for r in map(json.loads, open(args.out))}
    gpu = torch.cuda.get_device_name(0)

    for name in args.models.split(","):
        base, ds = build(name)
        hw = 128 if ds == "imagenet100" else 32
        imgs = test_images(ds, 32 if ds == "imagenet100" else 256)  # Spatha: at most 65535 column tiles per call
        for m in map(int, args.ms.split(",")):
            todo = [b for b in map(int, args.batches.split(",")) if (name, args.v, m, b) not in done]
            if not todo:
                continue
            pruned = copy.deepcopy(base)
            for _, c in [(n_, c_) for n_, c_ in pruned.named_modules() if isinstance(c_, nn.Conv2d)]:
                co, ci, kh, kw = c.weight.shape
                w2d, _ = prune_vnm_2d(conv_to_2d(c.weight.data.float()), v=args.v, m=m)
                c.weight.data.copy_(conv_from_2d(w2d, co, ci, kh, kw))
            sp_layers = {}

            def mk(nm, conv):
                sp_layers[nm] = SpathaConv2d(conv, args.v, m)
                return sp_layers[nm]

            spm = copy.deepcopy(pruned)
            swap_convs(spm, mk)
            spm = spm.cuda().half().eval()
            ref = copy.deepcopy(pruned)  # cuDNN model with exactly the kernel's weights
            refconvs = dict((n_, c_) for n_, c_ in ref.named_modules() if isinstance(c_, nn.Conv2d))
            for nm, layer in sp_layers.items():
                co, ci, kh, kw = layer.geo[:4]
                refconvs[nm].weight.data.copy_(conv_from_2d(layer.w_eff, co, ci, kh, kw))
            ref = ref.cuda().half().eval().to(memory_format=torch.channels_last)
            dense = copy.deepcopy(base).cuda().half().eval().to(memory_format=torch.channels_last)

            with torch.no_grad():
                free, tot = torch.cuda.mem_get_info()
                print(f"[mem] free {free / 2**30:.1f} GiB of {tot / 2**30:.1f}, allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
                a, b_ = spm(imgs).float(), ref(imgs.contiguous(memory_format=torch.channels_last)).float()
                relerr = ((a - b_).abs().max() / b_.abs().max()).item()
                agree = (a.argmax(1) == b_.argmax(1)).float().mean().item()

                for batch in todo:
                    x = torch.randn(batch, 3, hw, hw, device="cuda").half()
                    xcl = x.contiguous(memory_format=torch.channels_last)
                    iters = 20 if batch > 1 else 100
                    # per-layer choice for the hybrid model: time each conv on its real input
                    inputs = {}
                    hooks = [l.register_forward_pre_hook(lambda mod, inp, nm=nm: inputs.__setitem__(nm, inp[0].contiguous()))
                             for nm, l in sp_layers.items()]
                    spm(x)
                    for h in hooks:
                        h.remove()
                    lt = {}  # per conv: ms for spatha, dense GEMM on the same im2col path, cuDNN
                    for nm, layer in sp_layers.items():
                        xi, rc = inputs[nm], refconvs[nm]
                        xic = xi.contiguous(memory_format=torch.channels_last)
                        fs = {"spatha": Fixed(layer, "spatha"), "im2col": Fixed(layer, "dense")}
                        lt[nm] = {k: statistics.median(cuda_time(lambda: f(xi), iters) for _ in range(3)) for k, f in fs.items()}
                        lt[nm]["cudnn"] = statistics.median(cuda_time(lambda: rc(xic), iters) for _ in range(3))
                    layer.mode = "spatha"
                    del inputs  # the cached per-layer inputs are large at 128 px

                    def variant(pick):  # pick(times) -> "spatha" | "im2col" | "cudnn"
                        mdl = copy.deepcopy(ref)
                        sel = {nm: pick(t) for nm, t in lt.items()}
                        swap_convs(mdl, lambda nm, c: c if sel[nm] == "cudnn" else Fixed(sp_layers[nm], "dense" if sel[nm] == "im2col" else "spatha"))
                        return mdl, sel

                    dense_best, sel_db = variant(lambda t: "im2col" if t["im2col"] < t["cudnn"] else "cudnn")
                    hyb, sel_h = variant(lambda t: min(t, key=t.get))
                    spm_s, _ = variant(lambda t: "spatha")
                    spm_d, _ = variant(lambda t: "im2col")
                    runs = {"dense_cudnn": dense, "dense_best": dense_best, "dense_im2col": spm_d, "spatha": spm_s, "hybrid": hyb}
                    T = {k: [] for k in runs}
                    for _ in range(args.rounds):  # interleaved rounds
                        for k, mdl in runs.items():
                            T[k].append(cuda_time(lambda: mdl(xcl), iters))
                    med = {k: statistics.median(v) for k, v in T.items()}
                    rec = dict(model=name, dataset=ds, v=args.v, m=m, batch=batch, gpu=gpu, n_convs=len(sp_layers),
                               im2col=IM2COL, n_spatha_in_hybrid=sum(v_ == "spatha" for v_ in sel_h.values()),
                               conv_ms={k: sum(t[k] for t in lt.values()) for k in ("spatha", "im2col", "cudnn")},
                               relerr=relerr, argmax_agree=agree, rounds=args.rounds,
                               **{f"ms_{k}": v for k, v in med.items()},
                               **{f"spread_{k}": (max(v) - min(v)) / statistics.median(v) for k, v in T.items()},
                               speedup_spatha=med["dense_best"] / med["spatha"],
                               speedup_hybrid=med["dense_best"] / med["hybrid"],
                               speedup_hybrid_vs_cudnn=med["dense_cudnn"] / med["hybrid"],
                               speedup_spatha_vs_im2col=med["dense_im2col"] / med["spatha"])
                    with open(args.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"{name} V={args.v} 2:{m} bs={batch}: cudnn {med['dense_cudnn']:.2f} dense-best {med['dense_best']:.2f} "
                          f"im2col-dense {med['dense_im2col']:.2f} spatha {med['spatha']:.2f} ({rec['speedup_spatha']:.2f}x) "
                          f"hybrid {med['hybrid']:.2f}ms ({rec['speedup_hybrid']:.2f}x vs dense-best, "
                          f"{rec['n_spatha_in_hybrid']}/{len(sp_layers)} sparse) relerr {relerr:.1e} agree {agree:.3f}", flush=True)
            del spm, ref, dense, pruned, sp_layers, runs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
