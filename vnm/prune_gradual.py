"""Gradual second-order V:N:M pruning (VENOM / oBERT style): the V:N:M target is reached in several OBS steps with a
short masked fine-tune (and a fresh Hessian) between them. Steps: keep M/2, M/4, ... 4 columns per V x M block (column
pruning only), then the final 2:4 inside the kept columns, then a long fine-tune. KD from the dense model throughout.

    python vnm/prune_gradual.py --run venom_cifar100_res --dataset cifar100 --top 3 --src retrained_v2 --shard 0/3
"""
import argparse
import json
import os

import recipe  # noqa: F401  (sets cwd / sys.path)
import torch
from recipe import Data, setup_config, train
from configuration import config
from src.analysis.run import get_run
from src.phenotype.neural_network.evaluator.fully_train import _create_model
from prune_eval import named_convs
from prune_obs import calib_batch, obs_prune_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--vs", default="8,32,128")
    p.add_argument("--ms", default="8,16,32")
    p.add_argument("--step-epochs", type=int, default=5)
    p.add_argument("--final-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--src", default="retrained_v2")
    p.add_argument("--out", default="vnm_gridres_gradual")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--nsamples", type=int, default=1024)
    args = p.parse_args()

    setup_config(args.run, args.dataset)
    device = config.get_device()
    data = Data(args.dataset, device)
    run = get_run(args.run)
    out_dir = os.path.join("runs", args.run, args.out)
    os.makedirs(out_dir, exist_ok=True)
    rec_path = os.path.join(out_dir, "results.jsonl")
    done = set()
    if os.path.exists(rec_path):
        done = {(r["bp"], r["v"], r["m"]) for r in map(json.loads, open(rec_path))}
    rows = [json.loads(l) for l in open(os.path.join("runs", args.run, args.src, "results.jsonl"))]
    top = sorted(rows, key=lambda r: -r["final_acc"])[: args.top]
    config.final_relu = any(r.get("final_relu", False) for r in top)
    k, total = map(int, args.shard.split("/"))
    tasks = [(rank, r, int(v), int(m)) for rank, r in enumerate(top, 1)
             for v in args.vs.split(",") for m in args.ms.split(",")]
    my = tasks[k::total]
    print("models:", [(r["bp"], round(r["final_acc"], 4)) for r in top], "tasks:", len(my), flush=True)
    run_bps = {}
    for bp, gen in run.get_most_accurate_blueprints(200):
        run_bps.setdefault(bp.id, (bp, gen))
    calib = calib_batch(data, args.nsamples)
    log = lambda s: print("   " + s, flush=True)

    def build(r):
        model = _create_model(run, *run_bps[r["bp"]], data.input_shape, r.get("fm", 1))
        model.load_state_dict(torch.load(os.path.join("runs", args.run, args.src, f"bp-{r['bp']}.pt"), map_location=device))
        return model.to(device)

    for rank, r, v, m in my:
        if (r["bp"], v, m) in done:
            continue
        teacher = build(r).eval()
        dense_acc = data.test_acc(teacher)
        model = build(r)
        convs = sum(c.weight.numel() * 4 for _, c in named_convs(model))
        other = sum(q.numel() for q in model.parameters()) * 4 - convs
        steps = []
        kc = m // 2
        while kc > 4:
            steps.append((kc, kc))
            kc //= 2
        steps.append((4, 4))      # column pruning done: 4 of M columns kept, rows still dense
        steps.append((4, 2))      # final 2:4 inside the kept columns
        acc_steps = []
        for si, (kc, n) in enumerate(steps):
            model.eval()
            masks, dense_b, sparse_b = obs_prune_model(model, calib, v, m, kcols=kc, n=n)
            acc0 = data.test_acc(model)
            last = si == len(steps) - 1
            ep = args.final_epochs if last else args.step_epochs
            print(f"[bp {r['bp']}] V={v} M={m} step {si + 1}/{len(steps)} kcols={kc} n={n} zero-shot {acc0 * 100:.2f}", flush=True)
            acc, _ = train(model, data, ep, args.lr, wd=0.01, bs=args.bs, warmup=1, teacher=teacher, masks=masks,
                           eval_every=10, log=log)
            acc_steps.append((acc0, acc))
        with open(rec_path, "a") as f:
            f.write(json.dumps(dict(run=args.run, bp=r["bp"], rank=rank, params=model.size(), m=m, v=v, acc_zero=acc_steps[-1][0],
                                    acc_ft=acc, dense_acc=dense_acc, dense_conv_bytes_fp32=dense_b, vnm_conv_bytes=sparse_b,
                                    other_bytes_fp32=other, steps=len(steps), step_epochs=args.step_epochs,
                                    final_epochs=args.final_epochs, acc_steps=acc_steps)) + "\n")
        print(f"[bp {r['bp']}] V={v} M={m} -> {acc * 100:.2f} ({dense_b / max(sparse_b, 1):.2f}x conv)", flush=True)


if __name__ == "__main__":
    main()
