"""Correctness + speed of one Spatha build (one V) on conv-as-GEMM shapes, for M in {4,8,16,32}.

    SPATHA_LIB=$VENOM_BUILD/spatha_v32_lib python vnm/spatha_vscan.py --v 32 [--out results/spatha_gemm/vscan.jsonl]
(no SPATHA_LIB -> the default V=128 build). Run inside scripts/venom_env.sh.
"""
import argparse, json, math, os, sys
import torch

sys.path.insert(0, os.path.join(os.environ.get("VENOM_BUILD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "venom_build")), "venom", "end2end"))
if os.environ.get("SPATHA_LIB"):
    sys.path.insert(0, os.environ["SPATHA_LIB"])
import spatha
from grouped_nmv_tensor import SrNMTensor, nm_vector_mask_sparsify

# (name, R=C_out, K=kh*kw*C_in, C=batch*H*W) with batch 128
SHAPES = [
    ("3x3 64->256 @32x32", 256, 576, 128 * 32 * 32),
    ("3x3 128->256 @16x16", 256, 1152, 128 * 16 * 16),
    ("3x3 256->256 @16x16", 256, 2304, 128 * 16 * 16),
    ("3x3 256->512 @8x8", 512, 2304, 128 * 8 * 8),
    ("1x1 256->256 @32x32", 256, 256, 128 * 32 * 32),
    ("3x3 512->512 @4x4", 512, 4608, 128 * 4 * 4),
]


def cuda_time(fn, iters=30, warmup=10, trials=7):
    """min over several trials: the GPUs are shared, so the fastest trial is the least disturbed one"""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(trials):
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v", type=int, required=True)
    p.add_argument("--ms", default="4,8,16,32")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "spatha_gemm", "vscan.jsonl"))
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    gpu = torch.cuda.get_device_name(0)
    torch.manual_seed(0)
    for name, R, K, C in SHAPES:
        for m in map(int, a.ms.split(",")):
            rec = dict(v=a.v, m=m, shape=name, R=R, K=K, C=C, gpu=gpu)
            try:
                kp = max(128, m * 8)
                kpad = int(math.ceil(K / kp) * kp)
                rpad = int(math.ceil(R / a.v) * a.v)
                cpad = int(math.ceil(C / 64) * 64)
                w = torch.randn(rpad, kpad)
                mask, cols = nm_vector_mask_sparsify(w, 2, m, a.v)
                keep = mask.clone()
                sp = SrNMTensor(2, m, a.v, w, mask, cols, torch.device("cuda"))
                x = torch.randn(kpad, cpad, device="cuda").half()
                bias = torch.zeros(rpad, device="cuda").half()
                f = lambda: spatha.spmm(sp.metadata, sp.columns, sp.values, x, bias, rpad, kpad, cpad, a.v, 2, m, sp.nnz, 0, 32, 4)
                out = f()
                wd = (w * keep).cuda().half()
                ref = wd @ x
                rec["relerr"] = ((out.float().T - ref.float()).abs().max() / ref.abs().max()).item()
                rec["ms_sparse"] = cuda_time(f)
                rec["ms_dense"] = cuda_time(lambda: wd @ x)
                rec["speedup"] = rec["ms_dense"] / rec["ms_sparse"]
            except Exception as exc:
                rec["error"] = str(exc)[:150]
            print(json.dumps(rec), flush=True)
            open(a.out, "a").write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
