from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from nutrition5k_cbam.data import (
    TARGET_COLUMNS,
    load_dish_metadata,
    read_id_file,
    write_id_file,
    write_manifest,
)
from nutrition5k_cbam.utils import save_json, seed_everything


def stratified_val_ids(train_ids: list[str], records, val_ratio: float, seed: int) -> set[str]:
    rows = []
    for dish_id in train_ids:
        if dish_id not in records:
            continue
        target = records[dish_id].targets
        rows.append(
            {
                "dish_id": dish_id,
                "calories": target["calories"],
                "mass": target["mass"],
                "num_ingrs": len(records[dish_id].ingredients),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return set()

    for col in ["calories", "mass", "num_ingrs"]:
        bins = min(5, df[col].nunique())
        df[f"{col}_bin"] = pd.qcut(df[col], q=bins, duplicates="drop").astype(str)
    df["strata"] = df[["calories_bin", "mass_bin", "num_ingrs_bin"]].agg("|".join, axis=1)

    rng = random.Random(seed)
    val_ids: set[str] = set()
    target_n = round(len(df) * val_ratio)
    for _, group in df.groupby("strata"):
        ids = group["dish_id"].tolist()
        rng.shuffle(ids)
        take = max(1, round(len(ids) * val_ratio)) if len(ids) >= 10 else 0
        val_ids.update(ids[:take])

    if len(val_ids) > target_n:
        val_ids = set(rng.sample(sorted(val_ids), target_n))
    elif len(val_ids) < target_n:
        remaining = [dish_id for dish_id in train_ids if dish_id not in val_ids and dish_id in records]
        val_ids.update(rng.sample(remaining, target_n - len(val_ids)))
    return val_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-side-frames", type=int, default=4)
    args = parser.parse_args()

    seed_everything(args.seed)
    records = load_dish_metadata(args.dataset_dir)
    official_train = read_id_file(args.dataset_dir / "dish_ids" / "splits" / "rgb_train_ids.txt")
    official_test = read_id_file(args.dataset_dir / "dish_ids" / "splits" / "rgb_test_ids.txt")

    val_ids = stratified_val_ids(official_train, records, args.val_ratio, args.seed)
    train_ids = [dish_id for dish_id in official_train if dish_id not in val_ids and dish_id in records]
    val_ids_list = [dish_id for dish_id in official_train if dish_id in val_ids and dish_id in records]
    test_ids = [dish_id for dish_id in official_test if dish_id in records]

    split_dir = args.output_dir / "splits"
    manifest_dir = args.output_dir / "manifests"
    write_id_file(split_dir / "train_ids.txt", train_ids)
    write_id_file(split_dir / "val_ids.txt", val_ids_list)
    write_id_file(split_dir / "test_ids.txt", test_ids)

    manifest_counts = {
        "train": write_manifest(
            manifest_dir / "train.jsonl", args.dataset_dir, train_ids, records, args.max_side_frames
        ),
        "val": write_manifest(
            manifest_dir / "val.jsonl", args.dataset_dir, val_ids_list, records, args.max_side_frames
        ),
        "test": write_manifest(
            manifest_dir / "test.jsonl", args.dataset_dir, test_ids, records, args.max_side_frames
        ),
    }

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "max_side_frames_per_dish": args.max_side_frames,
        "dish_counts": {"train": len(train_ids), "val": len(val_ids_list), "test": len(test_ids)},
        "image_counts": manifest_counts,
        "target_columns": TARGET_COLUMNS,
    }
    save_json(args.output_dir / "split_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

