"""Whole-model Spatha vs cuDNN agreement at a given number of test images (mirrors e2e_bench's check)."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import e2e_bench as E
import copy, torch, torch.nn as nn
name, v, m = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
base, ds = E.build(name)
for c in [c for c in base.modules() if isinstance(c, nn.Conv2d)]:
    co, ci, kh, kw = c.weight.shape
    w2d, _ = E.prune_vnm_2d(E.conv_to_2d(c.weight.data.float()), v=v, m=m)
    c.weight.data.copy_(E.conv_from_2d(w2d, co, ci, kh, kw))
layers = {}
spm = copy.deepcopy(base)
E.swap_convs(spm, lambda nm, c: layers.setdefault(nm, E.SpathaConv2d(c, v, m)))
spm = spm.cuda().half().eval()
ref = copy.deepcopy(base)
rc = {n: c for n, c in ref.named_modules() if isinstance(c, nn.Conv2d)}
for nm, l in layers.items():
    rc[nm].weight.data.copy_(E.conv_from_2d(l.w_eff, *l.geo[:4]))
ref = ref.cuda().half().eval()
imgs = E.test_images(ds, 256)
with torch.no_grad():
    for n in (32, 128, 256):
        a, b = spm(imgs[:n]).float(), ref(imgs[:n]).float()
        print(f"{name} V={v} 2:{m} n={n}: agree {(a.argmax(1) == b.argmax(1)).float().mean().item():.3f} "
              f"relerr {((a - b).abs().max() / b.abs().max()).item():.1e} nonfinite {(~torch.isfinite(a)).sum().item()}", flush=True)
