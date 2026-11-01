"""Time V:N:M-pruned conv layers with the Spatha SpMM kernel against dense fp16 baselines.

Run inside the VENOM env (torch 1.13 + spatha), NOT the CoDeepNEAT env:

    source scripts/venom_env.sh
    CUDA_VISIBLE_DEVICES=0 python vnm/spatha_bench.py --run venom_cifar10 [--batch 128]

Each conv is treated as an im2col GEMM: W2d (C_out x kh*kw*C_in) @ X (kh*kw*C_in x B*H_out*W_out).
Spatha times cover the sparse GEMM only (im2col is not included); the dense baselines are
  gemm  - fp16 cuBLAS GEMM of the same shape (apples to apples with Spatha)
  cudnn - the real fp16 channels_last nn.functional.conv2d (what a dense deployment would run)
Inputs are the `*_layers.pt` files written by vnm/prune_eval.py.
"""

import argparse
import glob
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

VENOM = os.path.join(os.environ.get("VENOM_BUILD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "venom_build")), "venom", "end2end")
sys.path.insert(0, VENOM)
if os.environ.get("SPATHA_LIB"):  # e.g. the V=64 build
    sys.path.insert(0, os.environ["SPATHA_LIB"])
import spatha  # noqa: E402
from grouped_nmv_tensor import SrNMTensor, nm_vector_mask_sparsify  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "CoDeepNEAT"))


KPAD = int(os.environ.get("KPAD", 128))


def round_up(x, y):
    return int(math.ceil(x / y) * y)


def cuda_time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_layer(layer, v, m, batch, n=2):
    w2d = layer["w2d"].float()
    r, k = w2d.shape
    c = batch * layer["h_out"] * layer["w_out"]
    r_pad, k_pad, c_pad = round_up(r, v), round_up(k, KPAD), round_up(c, 64)
    w = torch.zeros(r_pad, k_pad)
    w[:r, :k] = w2d
    # break the all-zero ties of the already-pruned weights so the kernel's mask picks our kept entries
    g = torch.Generator().manual_seed(0)
    w_sel = w + 1e-7 * torch.randn(w.shape, generator=g)
    mask, cols = nm_vector_mask_sparsify(w_sel, n, m, v)
    mask_keep = mask.clone()
    sp = SrNMTensor(n, m, v, w, mask, cols, torch.device("cuda"))
    x = torch.randn(k_pad, c_pad, device="cuda").half()
    bias = torch.zeros(r_pad, device="cuda").half()

    def sparse():
        return spatha.spmm(sp.metadata, sp.columns, sp.values, x, bias, r_pad, k_pad, c_pad, v, n, m, sp.nnz, 0, 32, 4)

    out = sparse()
    ref = (w * mask_keep).cuda().half() @ x
    relerr = ((out.float().T - ref.float()).abs().max() / ref.abs().max().clamp_min(1e-6)).item()
    covered = ((w != 0) & (mask_keep == 0)).sum().item()  # nonzeros the kernel's own mask would drop

    t_sparse = cuda_time(sparse)
    wd = w[:r, :k].cuda().half()
    xd = torch.randn(k, c, device="cuda").half()
    t_gemm = cuda_time(lambda: wd @ xd)

    xin = torch.randn(batch, layer["c_in"], layer["h_in"], layer["w_in"], device="cuda").half()
    xin = xin.contiguous(memory_format=torch.channels_last)
    wc = layer["w2d"].cuda().half().reshape(r, layer["kh"], layer["kw"], layer["c_in"]).permute(0, 3, 1, 2)
    wc = wc.contiguous(memory_format=torch.channels_last)
    t_cudnn = cuda_time(lambda: F.conv2d(xin, wc, stride=layer["stride"], padding=layer["padding"]))
    return dict(name=layer["name"], R=r, K=k, C=c, relerr=relerr, mask_mismatch=covered,
                ms_sparse=t_sparse, ms_gemm=t_gemm, ms_cudnn=t_cudnn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--pattern", default="*_layers.pt")
    p.add_argument("--dir", default="vnm")
    args = p.parse_args()

    vdir = os.path.join(ROOT, "runs", args.run, args.dir)
    out_path = os.path.join(vdir, "spatha_bench.jsonl")
    gpu = torch.cuda.get_device_name(0)
    for f in sorted(glob.glob(os.path.join(vdir, args.pattern))):
        tag = os.path.basename(f)[: -len("_layers.pt")]
        # tag: bp-<id>_v<V>_m<M>_ft<E>
        parts = dict((s[0], s[1:]) for s in tag.split("_")[1:] if s[0] in "vm")
        v, m = int(parts["v"]), int(parts["m"])
        layers = torch.load(f)
        rows, skipped = [], []
        try:
            for layer in layers:
                if layer["groups"] != 1:
                    skipped.append(layer["name"])
                    continue
                rows.append(bench_layer(layer, v, m, args.batch))
        except Exception as exc:  # unsupported V/M combination or shape
            print(f"{tag}: FAILED {str(exc)[:120]}", flush=True)
            with open(out_path, "a") as fo:
                fo.write(json.dumps(dict(run=args.run, tag=tag, v=v, m=m, error=str(exc)[:200])) + "\n")
            continue
        tot = {k: sum(r[k] for r in rows) for k in ("ms_sparse", "ms_gemm", "ms_cudnn")}
        best = sum(min(r["ms_sparse"], r["ms_gemm"]) for r in rows)
        rec = dict(run=args.run, tag=tag, v=v, m=m, batch=args.batch, gpu=gpu, n_layers=len(rows), skipped=skipped,
                   total_ms=tot, speedup_vs_gemm=tot["ms_gemm"] / tot["ms_sparse"],
                   speedup_vs_cudnn=tot["ms_cudnn"] / tot["ms_sparse"],
                   speedup_vs_cudnn_best_of=tot["ms_cudnn"] / best, max_relerr=max(r["relerr"] for r in rows),
                   layers=rows)
        print(f"{tag}: conv-only speedup vs gemm {rec['speedup_vs_gemm']:.2f}x, vs cudnn {rec['speedup_vs_cudnn']:.2f}x, "
              f"best-of {rec['speedup_vs_cudnn_best_of']:.2f}x, max relerr {rec['max_relerr']:.1e}", flush=True)
        with open(out_path, "a") as fo:
            fo.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
