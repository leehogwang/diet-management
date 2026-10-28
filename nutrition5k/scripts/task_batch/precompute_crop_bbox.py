"""Precompute plate-ROI bounding boxes for every usable overhead dish -> JSON cache.
Used by the cropped-image retrain (Task4 empirical test)."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/students/cs/202520906/nutrition5k_pipeline")
from scripts.task_batch.preprocess_plate_roi import process, OV
from scripts.task_batch.train_grams import usable

OUT = Path("/home/students/cs/202520906/nutrition5k_pipeline/outputs/task_runs/task4_final/crop_bbox.json")


def main():
    dishes = set()
    for sp in ["train", "val", "test"]:
        d, _ = usable(sp)
        dishes.update(d)
    bb = {}
    for i, d in enumerate(sorted(dishes)):
        try:
            _, _, _, box, _ = process(d)
            bb[d] = [int(v) for v in box]
        except Exception:
            bb[d] = None
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(dishes)}", flush=True)
    OUT.write_text(json.dumps(bb))
    n_ok = sum(1 for v in bb.values() if v)
    print(f"wrote {OUT}  ({n_ok}/{len(bb)} with bbox)")


if __name__ == "__main__":
    main()
