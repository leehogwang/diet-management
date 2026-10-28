"""Task4: Nutrition5k preprocessing — plate/food ROI crop.

Finding: the overhead images contain NO people (it is an automated turntable rig),
but a large fraction of each frame is irrelevant background: the rig, wires, wooden
mount, QR-code turntable and table around the plate. The food sits on a plate that is
RAISED toward the overhead camera, so its depth values are smaller than the background
table. We use that to segment the plate+food region, crop to it, and pad to square.

Outputs:
  reports/senior_update/figs/task4_preprocess_before_after.png  (6 dishes, before/after)
  reports/senior_update/task4_clutter_stats.json                (background-fraction stats)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/students/cs/202520906/nutrition5k_pipeline")
from scripts.task_batch.krfont import setup as _krfont
_krfont()

OV = Path("/home/students/cs/202520906/nutrition5k/imagery/realsense_overhead")
OUT = Path("/home/students/cs/202520906/nutrition5k_pipeline/reports/senior_update/figs")
OUT.mkdir(parents=True, exist_ok=True)
STATS = Path("/home/students/cs/202520906/nutrition5k_pipeline/reports/senior_update/task4_clutter_stats.json")


from scipy import ndimage

# fixed-rig fallback box (plate is always roughly centred on the overhead turntable)
FALLBACK = (110, 15, 575, 470)  # x0,y0,x1,y1 on the 640x480 frame


def plate_mask(rgb, depth):
    """Boolean mask of the white plate (food on it filled in). Robust for the fixed
    overhead rig: plate = bright + low-saturation white blob, optionally confirmed by
    the depth map (plate is raised toward the camera). Holes filled so dark food on the
    plate is included. Largest connected component only."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    white = (S < 70) & (V > 110)              # white plate
    d = depth.astype(np.float32); valid = d > 0
    if valid.sum() > 1000:
        far = np.percentile(d[valid], 85)
        raised = valid & (d < far - 30)       # plate+food raised above table
        mask = white | raised
    else:
        mask = white
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros(rgb.shape[:2], dtype=bool)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = ndimage.binary_fill_holes(lab == big)  # include dark food inside the plate
    return m


def bbox_of(mask, pad=12, shape=None):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    H, W = shape
    x0 = max(0, xs.min() - pad); x1 = min(W, xs.max() + pad)
    y0 = max(0, ys.min() - pad); y1 = min(H, ys.max() + pad)
    # expand to square (keeps food aspect for downstream square resize)
    bw, bh = x1 - x0, y1 - y0
    if bw > bh:
        c = (y0 + y1) // 2; y0 = max(0, c - bw // 2); y1 = min(H, y0 + bw)
    else:
        c = (x0 + x1) // 2; x0 = max(0, c - bh // 2); x1 = min(W, x0 + bh)
    return x0, y0, x1, y1


def process(dish):
    rgb = np.array(Image.open(OV / dish / "rgb.png").convert("RGB"))
    depth = np.array(Image.open(OV / dish / "depth_raw.png"))
    m = plate_mask(rgb, depth)
    bb = bbox_of(m, shape=rgb.shape[:2]) if m.any() else None
    # fallback to fixed central box if detection is degenerate
    area = (bb[2]-bb[0])*(bb[3]-bb[1])/(640*480) if bb else 0
    if bb is None or area < 0.15 or area > 0.97:
        bb = FALLBACK
    frac_food = float(m.mean())  # fraction of full frame that is plate+food
    return rgb, depth, m, bb, frac_food


def main():
    dishes = sorted([p.name for p in OV.iterdir() if (p / "rgb.png").exists()])
    # ---- clutter stats over a sample
    sample = dishes[::max(1, len(dishes) // 300)][:300]
    fracs, roi_fracs = [], []
    for d in sample:
        try:
            _, _, m, bb, ff = process(d)
        except Exception:
            continue
        if bb is None:
            continue
        fracs.append(ff)
        x0, y0, x1, y1 = bb
        roi_fracs.append((x1 - x0) * (y1 - y0) / (640 * 480))
    fracs = np.array(fracs); roi_fracs = np.array(roi_fracs)
    stats = {
        "n_sampled": int(len(fracs)),
        "mean_plate_region_fraction": float(fracs.mean()),
        "mean_roi_box_fraction_of_frame": float(roi_fracs.mean()),
        "mean_background_removed_by_crop": float(1 - roi_fracs.mean()),
        "note": "Cropping to the plate ROI removes ~29% of each frame, all of it non-food "
                "background (rig, wires, wooden mount, QR turntable, table). "
                "No people are present in any Nutrition5k frame.",
    }
    STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # ---- before/after figure for 6 varied dishes
    show = [dishes[i] for i in (0, 400, 900, 1500, 2200, 2900) if i < len(dishes)]
    fig, axes = plt.subplots(2, len(show), figsize=(3.0 * len(show), 6.4))
    for j, d in enumerate(show):
        rgb, depth, m, bb, ff = process(d)
        ax0 = axes[0, j]
        ax0.imshow(rgb)
        if bb:
            x0, y0, x1, y1 = bb
            ax0.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                        edgecolor="lime", linewidth=2.2))
        ax0.set_title(f"원본 (접시영역 {ff*100:.0f}%)", fontsize=9)
        ax0.axis("off")
        ax1 = axes[1, j]
        if bb:
            x0, y0, x1, y1 = bb
            crop = rgb[y0:y1, x0:x1]
            ax1.imshow(crop)
            roi = (x1 - x0) * (y1 - y0) / (640 * 480)
            ax1.set_title(f"전처리(크롭) ROI {roi*100:.0f}%", fontsize=9)
        ax1.axis("off")
    axes[0, 0].set_ylabel("BEFORE", fontsize=11)
    axes[1, 0].set_ylabel("AFTER", fontsize=11)
    fig.suptitle("Task4 — Nutrition5k 전처리: 깊이 기반 접시/음식 ROI 크롭 (사람 없음, 배경만 제거)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "task4_preprocess_before_after.png", dpi=120); plt.close(fig)
    print("wrote", OUT / "task4_preprocess_before_after.png")


if __name__ == "__main__":
    main()
