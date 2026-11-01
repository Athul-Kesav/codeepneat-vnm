"""From-scratch CIFAR ResNet baselines trained with the exact recipe used for the evolved models (vnm/recipe.py).

    python vnm/train_resnet.py --name r20w58 --epochs 300 --cutmix 0.5 --ema 0.999
Results -> runs/venom_cifar100/resnet_baselines/results.jsonl
"""
import argparse
import json
import os
import time

import recipe  # noqa: F401  (sets cwd / sys.path)
import torch
import torch.nn as nn
import torch.nn.functional as F
from recipe import Data, setup_config, train
from configuration import config

# name -> (blocks per stage, base width); CIFAR-style 3-stage basic-block ResNet, 3x3 stem, no maxpool
SPECS = {"r20w58": ([3, 3, 3], 58), "r56w32": ([9, 9, 9], 32), "r110w24": ([18, 18, 18], 24), "r18": ([2, 2, 2, 2], 64)}


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = None if stride == 1 and cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        return F.relu(y + (x if self.sc is None else self.sc(x)))


class ResNet(nn.Module):
    def __init__(self, blocks, w, classes=100):
        super().__init__()
        widths = [w * 2 ** i for i in range(len(blocks))]
        layers, cin = [nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU()], w
        for i, (n, cout) in enumerate(zip(blocks, widths)):
            for j in range(n):
                layers.append(Block(cin, cout, 2 if (j == 0 and i > 0) else 1)); cin = cout
        self.body, self.fc = nn.Sequential(*layers), nn.Linear(cin, classes)

    def forward(self, x):
        return F.log_softmax(self.fc(F.adaptive_avg_pool2d(self.body(x), 1).flatten(1)), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, choices=SPECS)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--cutmix", type=float, default=0.5)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--count", action="store_true")
    args = p.parse_args()
    blocks, w = SPECS[args.name]
    model = ResNet(blocks, w)
    n_params = sum(p.numel() for p in model.parameters())
    print(args.name, "params", n_params, flush=True)
    if args.count:
        return
    setup_config("venom_cifar100", "cifar100")
    device = config.get_device()
    data = Data("cifar100", device)
    torch.manual_seed(0)
    model = model.to(device)
    t0 = time.time()
    final, best = train(model, data, args.epochs, args.lr, cutmix=args.cutmix, ema=args.ema, log=lambda s: print(s, flush=True))
    out = os.path.join("runs", "venom_cifar100", "resnet_baselines")
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, f"{args.name}.pt"))
    with open(os.path.join(out, "results.jsonl"), "a") as f:
        f.write(json.dumps(dict(name=args.name, params=n_params, final_acc=final, best_acc=best, epochs=args.epochs,
                                cutmix=args.cutmix, ema=args.ema, secs=time.time() - t0)) + "\n")
    print(f"== {args.name} done final {final * 100:.2f} best {best * 100:.2f}", flush=True)


if __name__ == "__main__":
    main()
