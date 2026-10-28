"""배터리8 실험3 — '공간 프레임 벡터' 사전계산.

배경/동기
---------
챔피언 파이프라인은 접시 ROI로 **크롭한 뒤 384x384로 리사이즈**한다. 이 과정에서
"접시가 원본 프레임 어디에 얼마나 크게 찍혔는가"라는 절대 스케일 정보가 완전히 사라진다.
같은 음식이라도 접시가 크게 찍히면 실제로 더 큰 접시일 가능성이 높으므로, 이 정보를
벡터로 되돌려 넣으면 대용량 접시 오차가 줄 수 있다는 가설이다.

주의(사전 점검에서 발견)
------------------------
crop_bbox.json 의 bbox 는 3261개 접시 중 2870개(88.0%)가 FALLBACK 고정박스
(110,15,575,470) 로 동일하다. 즉 **bbox 4좌표만으로는 88%가 같은 벡터**라 학습 신호가
거의 없다. 그래서 여기서는 bbox 뿐 아니라 실제로 분산이 존재하는 성분까지 계산한다:

  [0] cx           bbox 중심 x / W          ) 위치
  [1] cy           bbox 중심 y / H          )
  [2] bw           bbox 너비 / W            ) 크기
  [3] bh           bbox 높이 / H            )
  [4] barea        bbox 면적 / (W*H)        )
  ------------------------------------------- 위 5개 = 'bbox only' 아블레이션 대상
  [5] mask_area    접시+음식 마스크 면적 / 프레임      ) 실제 접시 크기
  [6] mask_cx      마스크 무게중심 x / W               ) 실제 접시 위치
  [7] mask_cy      마스크 무게중심 y / H               )
  [8] mask_rg      마스크 관성반경 / W (퍼짐 정도)      )
  [9] valid_frac   크롭 영역 내 유효깊이 픽셀 비율
  [10] depth_mean  마스크 내 평균 깊이(mm)/1000  = 카메라~접시면 절대거리
  [11] depth_spread 마스크 내 (p85-p15) 깊이(mm)/100 = 음식 높이 편차(부피 단서)

[10][11] 은 4번째 입력채널이 접시별 z-score 로 정규화되면서 버려지는 **절대 높이** 정보라
이미지로부터 복원 불가능하다 = 진짜 신규 정보.

flip 증강 시 x 좌표 성분([0],[6])은 1-v 로 미러링해야 한다(FLIP_X_IDX).

출력: outputs/task_runs/task_b8/spatial_feats.npz  (dishes, feats(N,12))
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/students/cs/202520906/nutrition5k_pipeline")
sys.path.insert(0, str(ROOT))
from scripts.task_batch.preprocess_plate_roi import OV, plate_mask, bbox_of, FALLBACK  # noqa: E402
from scripts.task_batch.train_grams import usable  # noqa: E402

OUT = ROOT / "outputs/task_runs/task_b8/spatial_feats.npz"
DIM = 12
FLIP_X_IDX = [0, 6]           # 좌우반전 시 v -> 1-v
BBOX_ONLY_IDX = [0, 1, 2, 3, 4]
NAMES = ["cx", "cy", "bw", "bh", "barea", "mask_area", "mask_cx", "mask_cy",
         "mask_rg", "valid_frac", "depth_mean", "depth_spread"]


def spatial_of(dish):
    """접시 하나의 12차원 공간 프레임 벡터. 실패해도 예외 대신 fallback 벡터를 낸다."""
    v = np.zeros(DIM, np.float32)
    try:
        rgb = np.array(Image.open(OV / dish / "rgb.png").convert("RGB"))
    except Exception:
        return dish, v
    H, W = rgb.shape[:2]
    try:
        depth = np.array(Image.open(OV / dish / "depth_raw.png")).astype(np.float32)
    except Exception:
        depth = np.zeros((H, W), np.float32)

    try:
        m = plate_mask(rgb, depth)
    except Exception:
        m = np.zeros((H, W), bool)

    # --- bbox: train_grams_crop 이 실제로 쓰는 것과 동일한 규칙(fallback 포함)
    bb = bbox_of(m, shape=(H, W)) if m.any() else None
    area = (bb[2] - bb[0]) * (bb[3] - bb[1]) / float(H * W) if bb else 0.0
    if bb is None or area < 0.15 or area > 0.97:
        bb = FALLBACK
    x0, y0, x1, y1 = bb
    v[0] = (x0 + x1) / 2.0 / W
    v[1] = (y0 + y1) / 2.0 / H
    v[2] = (x1 - x0) / float(W)
    v[3] = (y1 - y0) / float(H)
    v[4] = v[2] * v[3]

    # --- 마스크 기하 (bbox 와 달리 fallback 이 없어 분산이 살아있다)
    if m.any():
        ys, xs = np.nonzero(m)
        v[5] = m.mean()
        mcx, mcy = xs.mean(), ys.mean()
        v[6] = mcx / W
        v[7] = mcy / H
        v[8] = float(np.sqrt(((xs - mcx) ** 2 + (ys - mcy) ** 2).mean())) / W
    else:  # 마스크 실패 시 fallback bbox 기하로 대체
        v[5] = v[4]; v[6] = v[0]; v[7] = v[1]; v[8] = v[2] / 2.0

    # --- 깊이 통계 (절대 스케일: 채널 z-score 로 사라지는 정보)
    crop_d = depth[y0:y1, x0:x1]
    valid = crop_d > 0
    v[9] = float(valid.mean()) if crop_d.size else 0.0
    dm = depth[m] if m.any() else crop_d.reshape(-1)
    dm = dm[dm > 0]
    if dm.size > 50:
        v[10] = float(dm.mean()) / 1000.0
        v[11] = float(np.percentile(dm, 85) - np.percentile(dm, 15)) / 100.0
    return dish, v


def main():
    dishes = set()
    for sp in ["train", "val", "test"]:
        d, _ = usable(sp)
        dishes.update(d)
    dishes = sorted(dishes)
    print(f"computing spatial frame vectors for {len(dishes)} dishes ...", flush=True)

    feats = np.zeros((len(dishes), DIM), np.float32)
    idx = {d: i for i, d in enumerate(dishes)}
    with Pool(16) as pool:
        for n, (d, v) in enumerate(pool.imap_unordered(spatial_of, dishes, chunksize=8), 1):
            feats[idx[d]] = v
            if n % 500 == 0:
                print(f"  {n}/{len(dishes)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, dishes=np.array(dishes), feats=feats, names=np.array(NAMES))
    print(f"SAVED {OUT}")
    print("\n--- 성분별 분포 (표준편차가 0에 가까우면 정보 없음) ---")
    for i, nm in enumerate(NAMES):
        col = feats[:, i]
        uniq = len(np.unique(np.round(col, 6)))
        print(f"  [{i:2d}] {nm:13s} mean {col.mean():8.4f}  std {col.std():8.4f}  "
              f"min {col.min():8.4f}  max {col.max():8.4f}  unique {uniq}")


if __name__ == "__main__":
    main()
