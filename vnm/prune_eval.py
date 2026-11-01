"""Load the CoDeepNEAT winner, apply V:N:M magnitude pruning, optional fine-tune, report acc + bytes.

Run from the CoDeepNEAT repo root after copying this folder next to it or onto sys.path:

    python ../vnm/prune_eval.py --run colab_baseline --ft-epochs 8
"""

from __future__ import annotations

import argparse
import json
import re
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "CoDeepNEAT"))
if os.path.isdir(ROOT):
    os.chdir(ROOT)
    sys.path[:0] = [ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "runs"), os.path.join(ROOT, "configuration")]

import torch
from torch import nn

from configuration import config
from configuration.config_loader import load_simple_config
from src.analysis.run import get_run
from src.phenotype.neural_network.evaluator.data_loader import load_data, load_transform
from src.phenotype.neural_network.evaluator.fully_train import _create_model
from src.phenotype.neural_network.evaluator.evaluator import test_nn, train_epoch
from src.analysis.reporters.reporter_set import ReporterSet

sys.path.insert(0, os.path.dirname(__file__))
from vnm_pack import conv_from_2d, conv_to_2d, pack_vnm_2d, packed_nbytes_tight, prune_vnm_2d, unpack_vnm_2d


def named_convs(model: nn.Module):
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.weight is not None:
            yield name, mod


def apply_vnm_to_model(model: nn.Module, v: int, m: int):
    masks = {}
    stats = []
    dense_bytes = 0
    sparse_bytes = 0
    for name, conv in named_convs(model):
        w = conv.weight.data
        c_out, c_in, kh, kw = w.shape
        w2d = conv_to_2d(w)
        dense_bytes += w.numel() * 4  # fp32 stored in the .model
        pruned, mask = prune_vnm_2d(w2d, v=v, m=m, n=2)
        packed = pack_vnm_2d(pruned, v=v, m=m, n=2)
        rebuilt = unpack_vnm_2d(packed)
        if not torch.allclose(rebuilt, pruned, atol=1e-5):
            print("WARNING unpack mismatch", name)
        sparse_bytes += packed_nbytes_tight(packed)
        conv.weight.data.copy_(conv_from_2d(pruned, c_out, c_in, kh, kw))
        masks[name] = conv_from_2d(mask.float(), c_out, c_in, kh, kw)
        kept = mask.float().mean().item()
        stats.append((name, tuple(w.shape), 1.0 - kept, packed_nbytes_tight(packed)))
    return masks, stats, dense_bytes, sparse_bytes


def conv_geometry(model: nn.Module, sample: torch.Tensor):
    """Record each conv's geometry and output size from one forward pass (used by the Spatha benchmark)."""
    info, hooks = {}, []
    for name, conv in named_convs(model):
        def hook(mod, inp, out, name=name):
            info[name] = dict(
                name=name, c_out=mod.out_channels, c_in=mod.in_channels, kh=mod.kernel_size[0], kw=mod.kernel_size[1],
                stride=mod.stride[0], padding=mod.padding[0], groups=mod.groups, h_in=inp[0].shape[2],
                w_in=inp[0].shape[3], h_out=out.shape[2], w_out=out.shape[3],
            )
        hooks.append(conv.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(sample)
    for h in hooks:
        h.remove()
    return info


def reapply_masks(model: nn.Module, masks: dict):
    by_name = dict(named_convs(model))
    for name, mask in masks.items():
        conv = by_name[name]
        conv.weight.data.mul_(mask.to(conv.weight.device))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="colab_baseline")
    p.add_argument("--run", default="colab_baseline")
    p.add_argument("--vs", default="128", help="comma-separated V values")
    p.add_argument("--ms", default="4,8,16")
    p.add_argument("--out", default="vnm", help="results folder name under runs/<run>/")
    p.add_argument("--layers-for-v", default="64,128", help="save layer files (for Spatha timing) only for these V")
    p.add_argument("--ft-epochs", type=int, default=8)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--model", default="", help="a fully_trained_models/bp-<id>_fm-<k>-best-<r>.model file (overrides --top-k)")
    args = p.parse_args()

    load_simple_config(args.config, lambda: None, lambda: None, ngpus=1, use_wandb_override=False)
    config.fully_train = True
    config.use_wandb = False
    config.evolve_da = False
    device = config.get_device()

    run = get_run(args.run)
    from src.phenotype.neural_network.evaluator.data_loader import get_data_shape
    in_size = get_data_shape()

    test_loader = load_data(load_transform(), "test")
    train_loader = load_data(load_transform(), "train")
    out_dir = os.path.join("runs", args.run, args.out)
    os.makedirs(out_dir, exist_ok=True)
    records_path = os.path.join(out_dir, "results.jsonl")
    done = set()
    if os.path.exists(records_path):
        done = {(r["rank"], r["v"], r["m"]) for r in map(json.loads, open(records_path))}
    vs = [int(x) for x in args.vs.split(",")]
    layer_vs = {int(x) for x in args.layers_for_v.split(",") if x}

    if args.model:  # one checkpoint, possibly at feature multiplier > 1
        bp_id, fm, rank = map(int, re.search(r"bp-(\d+)_fm-(\d+)-best-(\d+)", os.path.basename(args.model)).groups())
        found = [(b, g) for b, g in run.get_most_accurate_blueprints(300) if b.id == bp_id]
        targets = [(rank, *found[0], fm, args.model)]
    else:
        targets = [(rank, bp, g, 1, os.path.join("runs", args.run, "fully_trained_models", f"bp-{bp.id}_fm-1-best-{rank}.model"))
                   for rank, (bp, g) in enumerate(run.get_most_accurate_blueprints(args.top_k), 1)]
    for rank, bp, gen_num, fm, model_path in targets:
        model = _create_model(run, bp, gen_num, in_size, fm)
        if not os.path.exists(model_path):
            print("missing", model_path, flush=True)
            continue
        sd = torch.load(model_path, map_location=device)
        model.load_state_dict(sd)
        model.to(device)
        n_params = model.size()
        print(f"[{args.run}] rank {rank} bp {bp.id} params {n_params}", flush=True)

        dense_acc = test_nn(model, test_loader)
        print(f"dense test acc: {dense_acc * 100:.3f}%", flush=True)
        sample = next(iter(test_loader))[0][:2].to(device)
        geometry = conv_geometry(model, sample)
        convs_fp32 = sum(c.weight.numel() * 4 for _, c in named_convs(model))
        other_bytes = (sum(p.numel() for p in model.parameters()) * 4) - convs_fp32
        if (rank, 0, 0) not in done:
            _append(records_path, dict(run=args.run, rank=rank, bp=bp.id, params=n_params, m=0, v=0, acc_zero=dense_acc,
                                       acc_ft=dense_acc, dense_conv_bytes_fp32=convs_fp32, vnm_conv_bytes=convs_fp32,
                                       other_bytes_fp32=other_bytes))

        print(f"{'M':>4} {'sparsity':>10} {'acc_zero':>10} {'acc_ft':>10} {'dense_B':>12} {'vnm_B':>12} {'ratio':>8}")
        for v, m in [(v, int(x)) for v in vs for x in args.ms.split(",")]:
            if (rank, v, m) in done:
                print(f"skip rank {rank} V={v} M={m} (done)", flush=True)
                continue
            model.load_state_dict(sd)
            model.to(device)
            masks, stats, dense_b, sparse_b = apply_vnm_to_model(model, v=v, m=m)
            acc0 = test_nn(model, test_loader)
            acc_ft = acc0
            if args.ft_epochs > 0:
                model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                for epoch in range(args.ft_epochs):
                    loss = train_epoch(model, train_loader, None, device, ReporterSet())
                    reapply_masks(model, masks)
                    print(f"  M={m} epoch {epoch + 1}/{args.ft_epochs} loss={loss:.4f}", flush=True)
                acc_ft = test_nn(model, test_loader)
            sparsity = 1.0 - 2.0 / m
            line = (
                f"{m:4d} {sparsity:10.2%} {acc0 * 100:10.2f} {acc_ft * 100:10.2f} "
                f"{dense_b:12d} {sparse_b:12d} {dense_b / max(sparse_b, 1):8.2f}x"
            )
            print(line, flush=True)

            tag = f"bp-{bp.id}_v{v}_m{m}_ft{args.ft_epochs}{args.tag}"
            if v in layer_vs:
                torch.save(model.state_dict(), os.path.join(out_dir, f"{tag}.pt"))
            # pruned 2D conv weights + geometry for vnm/spatha_bench.py (loads in any torch version)
            if v in layer_vs:
                layers = []
                for name, conv in named_convs(model):
                    g = dict(geometry[name])
                    g["w2d"] = conv_to_2d(conv.weight.detach().cpu()).half()
                    layers.append(g)
                torch.save(layers, os.path.join(out_dir, f"{tag}_layers.pt"))
            _append(records_path, dict(run=args.run, rank=rank, bp=bp.id, params=n_params, m=m, v=v,
                                       acc_zero=acc0, acc_ft=acc_ft, dense_conv_bytes_fp32=dense_b,
                                       vnm_conv_bytes=sparse_b, dense_acc=dense_acc, ft_epochs=args.ft_epochs,
                                       other_bytes_fp32=other_bytes, layers_file=f"{tag}_layers.pt" if v in layer_vs else ""))
            with open(os.path.join(out_dir, "results.txt"), "a") as f:
                f.write(f"bp{bp.id} rank{rank} v{v} " + line + "\n")


def _append(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
