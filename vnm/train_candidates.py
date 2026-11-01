"""Retrain the best evolved architectures from scratch with the modern recipe (vnm/recipe.py).

    python vnm/train_candidates.py --run venom_cifar100 --dataset cifar100 --n 6 --epochs 200 --shard 0/3
"""
import argparse
import json
import os
import time

import recipe  # noqa: F401  (sets cwd / sys.path)
import torch
from recipe import Data, setup_config, train
from configuration import config
from src.analysis.run import get_run
from src.phenotype.neural_network.evaluator.fully_train import _create_model


def candidates(run, in_shape, n, max_params, pool=80, only=None):
    seen, out = set(), []
    for bp, gen in run.get_most_accurate_blueprints(pool):
        if bp.id in seen:
            continue
        seen.add(bp.id)
        if only and bp.id not in only:
            continue
        try:
            model = _create_model(run, bp, gen, in_shape, 1)
        except Exception as exc:  # architecture invalid at this resolution
            print("skip", bp.id, str(exc)[:80])
            continue
        params = model.size()
        del model
        if params <= max_params:
            out.append((bp, gen, params))
        if len(out) == n:
            break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--max-params", type=float, default=15e6)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--out", default="retrained")
    p.add_argument("--bps", default="", help="comma-separated blueprint ids to train (overrides ranking)")
    p.add_argument("--cutmix", type=float, default=0.0)
    p.add_argument("--ema", type=float, default=0.0)
    args = p.parse_args()

    setup_config(args.run, args.dataset)
    device = config.get_device()
    data = Data(args.dataset, device)
    run = get_run(args.run)
    out_dir = os.path.join("runs", args.run, args.out)
    os.makedirs(out_dir, exist_ok=True)
    k, total = map(int, args.shard.split("/"))
    only = {int(x) for x in args.bps.split(",")} if args.bps else None
    cands = candidates(run, data.input_shape, args.n, args.max_params, only=only)
    print("candidates:", [(bp.id, params) for bp, _, params in cands], flush=True)

    for bp, gen, params in cands[k::total]:
        path = os.path.join(out_dir, f"bp-{bp.id}.pt")
        if os.path.exists(path):
            continue
        torch.manual_seed(bp.id)
        model = _create_model(run, bp, gen, data.input_shape, 1)
        print(f"== bp {bp.id} params {params} evo_acc {bp.max_acc:.4f}", flush=True)
        t0 = time.time()
        final, best = train(model, data, args.epochs, args.lr, bs=args.bs, cutmix=args.cutmix, ema=args.ema, log=lambda s: print(s, flush=True))
        torch.save(model.state_dict(), path)
        with open(os.path.join(out_dir, "results.jsonl"), "a") as f:
            f.write(json.dumps(dict(run=args.run, bp=bp.id, gen=gen, params=params, evo_acc=bp.max_acc, final_acc=final,
                                    best_acc=best, epochs=args.epochs, cutmix=args.cutmix, ema=args.ema, secs=time.time() - t0)) + "\n")
        print(f"== bp {bp.id} done final {final * 100:.2f} best {best * 100:.2f}", flush=True)


if __name__ == "__main__":
    main()
