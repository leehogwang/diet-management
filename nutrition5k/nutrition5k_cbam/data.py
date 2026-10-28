from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TARGET_COLUMNS = ["calories", "mass", "fat", "carb", "protein"]


@dataclass(frozen=True)
class DishRecord:
    dish_id: str
    targets: dict[str, float]
    ingredients: list[str]


def read_id_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_id_file(path: Path, ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n")


def load_dish_metadata(dataset_dir: Path) -> dict[str, DishRecord]:
    records: dict[str, DishRecord] = {}
    for csv_path in [
        dataset_dir / "metadata" / "dish_metadata_cafe1.csv",
        dataset_dir / "metadata" / "dish_metadata_cafe2.csv",
    ]:
        with csv_path.open(newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                dish_id = row[0]
                targets = {name: float(row[i + 1]) for i, name in enumerate(TARGET_COLUMNS)}
                ingredients: list[str] = []
                for i in range(6, len(row), 7):
                    if i + 1 < len(row) and row[i + 1]:
                        ingredients.append(row[i + 1])
                records[dish_id] = DishRecord(dish_id, targets, ingredients)
    return records


def available_images(dataset_dir: Path, dish_id: str, max_side_frames: int = 4) -> list[Path]:
    images: list[Path] = []
    overhead_rgb = dataset_dir / "imagery" / "realsense_overhead" / dish_id / "rgb.png"
    if overhead_rgb.exists():
        images.append(overhead_rgb)

    side_dir = dataset_dir / "imagery" / "side_angles" / dish_id / "frames_sampled5"
    side_images = sorted(side_dir.glob("*.jpeg")) if side_dir.exists() else []
    if max_side_frames > 0 and len(side_images) > max_side_frames:
        if max_side_frames == 1:
            side_images = [side_images[len(side_images) // 2]]
        else:
            step = (len(side_images) - 1) / (max_side_frames - 1)
            side_images = [side_images[round(i * step)] for i in range(max_side_frames)]
    images.extend(side_images)
    return images


def write_manifest(
    output_path: Path,
    dataset_dir: Path,
    dish_ids: Iterable[str],
    records: dict[str, DishRecord],
    max_side_frames: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w") as f:
        for dish_id in dish_ids:
            record = records.get(dish_id)
            if record is None:
                continue
            for image_path in available_images(dataset_dir, dish_id, max_side_frames):
                item = {
                    "dish_id": dish_id,
                    "image_path": str(image_path),
                    "targets": record.targets,
                    "ingredients": record.ingredients,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
    return count


def load_manifest(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]

