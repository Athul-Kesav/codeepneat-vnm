"""Turn the Hugging Face ImageNet-100 parquet into ImageFolder train/ and test/."""

from __future__ import annotations

import io
from pathlib import Path

from datasets import load_dataset
from PIL import Image

PARQUET = Path(__file__).resolve().parents[1] / "CoDeepNEAT" / "data" / "Datasets" / "imagenet100_hf" / "data"
OUT = Path(__file__).resolve().parents[1] / "CoDeepNEAT" / "data" / "Datasets" / "imagenet100"


def as_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    raw = value["bytes"] if isinstance(value, dict) else value
    return Image.open(io.BytesIO(raw)).convert("RGB")


def export(split_name: str, folder_name: str, files: list[str]) -> None:
    ds = load_dataset("parquet", data_files={split_name: files}, split=split_name)
    names = ds.features["label"].names
    root = OUT / folder_name
    root.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(ds):
        label = names[row["label"]]
        dest = root / label
        dest.mkdir(exist_ok=True)
        path = dest / f"{i:06d}.jpg"
        if not path.exists():
            as_image(row["image"]).save(path, quality=90)
        if i and i % 10000 == 0:
            print(folder_name, i, flush=True)
    print(folder_name, "done", len(ds), "classes", len(names), flush=True)


def main() -> None:
    train = sorted(str(p) for p in PARQUET.glob("train-*.parquet"))
    val = sorted(str(p) for p in PARQUET.glob("validation-*.parquet"))
    if len(train) != 17 or len(val) != 1:
        raise SystemExit(f"expected 17 train shards and 1 val shard, got {len(train)} and {len(val)}")
    export("train", "train", train)
    export("validation", "test", val)


if __name__ == "__main__":
    main()
