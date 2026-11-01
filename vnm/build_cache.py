"""Pre-decode ImageNet-100 into uint8 arrays so retraining can augment on the GPU.

train: short side -> 144, centre crop 144x144 (random 128 crops are taken at train time)
test : short side -> 146, centre crop 128x128 (standard 0.875 crop ratio)
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CoDeepNEAT", "data", "Datasets", "imagenet100")
OUT = os.path.join(os.environ.get("VENOM_BUILD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "venom_build")), "cache", "in100")
os.makedirs(OUT, exist_ok=True)


def load(args):
    path, short, crop = args
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = short / min(w, h)
    im = im.resize((max(short, round(w * s)), max(short, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - crop) // 2, (h - crop) // 2
    return np.asarray(im.crop((l, t, l + crop, t + crop)), dtype=np.uint8)


def build(split, short, crop):
    classes = sorted(os.listdir(f"{SRC}/{split}"))
    items = [(os.path.join(f"{SRC}/{split}", c, f), ci) for ci, c in enumerate(classes)
             for f in sorted(os.listdir(f"{SRC}/{split}/{c}"))]
    x = np.lib.format.open_memmap(f"{OUT}/{split}_x.npy", mode="w+", dtype=np.uint8, shape=(len(items), crop, crop, 3))
    with Pool(24) as pool:
        for i, arr in enumerate(pool.imap(load, [(p, short, crop) for p, _ in items], chunksize=64)):
            x[i] = arr
            if i % 20000 == 0:
                print(split, i, len(items), flush=True)
    x.flush()
    np.save(f"{OUT}/{split}_y.npy", np.array([c for _, c in items], dtype=np.int64))
    print(split, "done", len(items), flush=True)


if __name__ == "__main__":
    build("test", 146, 128)
    build("train", 144, 144)
