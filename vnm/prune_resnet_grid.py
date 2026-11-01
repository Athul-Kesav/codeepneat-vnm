"""V:N:M grid on the from-scratch ResNet baselines (vnm/train_resnet.py), same recipe as prune_grid2.py.

    python vnm/prune_resnet_grid.py --ft-epochs 30 --shard 0/3

Original docstring: V:N:M grid on the retrained models: prune, then fine-tune with the mask applied every step and knowledge
distillation from the dense model. Results -> runs/<run>/<out>/results.jsonl (same schema as prune_eval.py).

    python vnm/prune_grid2.py --run imagenet100 --dataset imagenet100 --top 3 --ft-epochs 30 --shard 0/3
"""
import argparse
import copy
import json
import os

import recipe  # noqa: F401  (sets cwd / sys.path)
import torch
from recipe import Data, setup_config, train
from configuration import config
from train_resnet import ResNet, SPECS
from prune_eval import apply_vnm_to_model, conv_geometry, named_convs
from prune_obs import calib_batch, obs_prune_model
from vnm_pack import conv_to_2d


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="venom_cifar100")
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--models", default="r20w58,r56w32,r18")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--vs", default="8,16,32,64,128")
    p.add_argument("--ms", default="4,8,16,32")
    p.add_argument("--ft-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--src", default="resnet_baselines")
    p.add_argument("--out", default="vnm_grid_resnet")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--method", default="mag", choices=["mag", "obs"], help="mag: magnitude; obs: second-order OBS/SparseGPT-style")
    p.add_argument("--nsamples", type=int, default=1024)
    p.add_argument("--layers-for-v", default="32,64,128")
    args = p.parse_args()

    setup_config(args.run, args.dataset)
    device = config.get_device()
    data = Data(args.dataset, device)
    out_dir = os.path.join("runs", args.run, args.out)
    os.makedirs(out_dir, exist_ok=True)
    rec_path = os.path.join(out_dir, "results.jsonl")
    done = set()
    if os.path.exists(rec_path):
        done = {(r["bp"], r["v"], r["m"]) for r in map(json.loads, open(rec_path))}
    layer_vs = {int(x) for x in args.layers_for_v.split(",") if x}

    names = args.models.split(",")
    top = [dict(bp=n, final_acc=0.0) for n in names]
    k, total = map(int, args.shard.split("/"))
    tasks = [(rank, r, int(v), int(m)) for rank, r in enumerate(top, 1)
             for v in args.vs.split(",") for m in args.ms.split(",")]
    my = tasks[k::total]
    print("models:", [r["bp"] for r in top], "tasks for this shard:", len(my), flush=True)

    def build(name):
        return ResNet(*SPECS[name]).to(device)

    cache = {}
    calib = calib_batch(data, args.nsamples) if args.method == "obs" else None

    def load(rank, r):
        if r["bp"] not in cache:
            model = build(r["bp"])
            sd = torch.load(os.path.join("runs", args.run, args.src, f"{r['bp']}.pt"), map_location=device)
            model.load_state_dict(sd)
            model.to(device).eval()
            n_params = sum(p.numel() for p in model.parameters())
            convs = sum(c.weight.numel() * 4 for _, c in named_convs(model))
            other = sum(p.numel() for p in model.parameters()) * 4 - convs
            sample = data._prep(data.xte[:2], False)
            geom = conv_geometry(model, sample)
            cache[r["bp"]] = (sd, model, n_params, convs, other, geom, data.test_acc(model))
        return cache[r["bp"]]

    for rank, r, v, m in my:
        sd, teacher, n_params, convs, other, geom, dense_acc = load(rank, r)
        if k == 0 and (r["bp"], 0, 0) not in done:
            append(rec_path, dict(run=args.run, rank=rank, bp=r["bp"], params=n_params, m=0, v=0, acc_zero=dense_acc,
                                  acc_ft=dense_acc, dense_acc=dense_acc, dense_conv_bytes_fp32=convs,
                                  vnm_conv_bytes=convs, other_bytes_fp32=other))
            done.add((r["bp"], 0, 0))
        if (r["bp"], v, m) in done:
            continue
        model = build(r["bp"])
        model.load_state_dict(sd)
        model.to(device)
        if args.method == "obs":
            model.eval()
            masks, dense_b, sparse_b = obs_prune_model(model, calib, v, m)
        else:
            masks, _, dense_b, sparse_b = apply_vnm_to_model(model, v=v, m=m)
        acc0 = data.test_acc(model)
        print(f"[rank {rank} bp {r['bp']}] V={v} M={m} dense {dense_acc * 100:.2f} zero-shot {acc0 * 100:.2f}", flush=True)
        acc_ft, _ = train(model, data, args.ft_epochs, args.lr, wd=0.01, bs=args.bs, warmup=1, teacher=teacher,
                          masks=masks, eval_every=10, log=lambda s: print("   " + s, flush=True))
        tag = f"bp-{r['bp']}_v{v}_m{m}_ft{args.ft_epochs}"
        layers_file = ""
        if v in layer_vs:
            layers = []
            for name, conv in named_convs(model):
                g = dict(geom[name])
                g["w2d"] = conv_to_2d(conv.weight.detach().cpu()).half()
                layers.append(g)
            layers_file = f"{tag}_layers.pt"
            torch.save(layers, os.path.join(out_dir, layers_file))
        append(rec_path, dict(run=args.run, rank=rank, bp=r["bp"], params=n_params, m=m, v=v, acc_zero=acc0, acc_ft=acc_ft,
                              dense_acc=dense_acc, dense_conv_bytes_fp32=dense_b, vnm_conv_bytes=sparse_b,
                              other_bytes_fp32=other, ft_epochs=args.ft_epochs, layers_file=layers_file))
        print(f"[rank {rank} bp {r['bp']}] V={v} M={m} -> {acc_ft * 100:.2f}  ({dense_b / max(sparse_b, 1):.2f}x conv)", flush=True)


if __name__ == "__main__":
    main()
