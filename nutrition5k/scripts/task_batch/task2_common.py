"""TASK 1~5 공통 유틸 — 5타깃(4영양소+중량) 로딩, PMA/RMSE 동시 계산, 구간 정의.

배경: 지금까지 학습 타깃은 4영양소(calories/fat/carb/protein)뿐이고 중량(mass)은 빠져 있었다.
여기서는 mass를 5번째 타깃으로 다시 꺼내 쓰되, 기존 실험과 비교 가능하도록
mean_PMA 는 항상 '4영양소 평균'으로만 계산하고 중량은 별도 지표로 보고한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/students/cs/202520906/nutrition5k_pipeline")
sys.path.insert(0, str(ROOT))
from nutrition5k_cbam.data import load_manifest  # noqa: E402
from scripts.task_batch.train_depth import OV, _ok, MANIFEST_DIR  # noqa: E402

NUTR4 = ["calories", "fat", "carb", "protein"]
TGT5 = NUTR4 + ["mass"]

# 구간 정의 (TASK 2 '약점 지도'용). (라벨, 하한, 상한)
WEIGHT_BINS = [("0-100g\n(소량)", 0, 100), ("100-300g\n(중간)", 100, 300),
               ("300-500g\n(표준)", 300, 500), ("500g+\n(대용량)", 500, np.inf)]
CAL_BINS = [("0-300\n(저칼로리)", 0, 300), ("300-600\n(중간)", 300, 600),
            ("600-900\n(고칼로리)", 600, 900), ("900+\n(초고칼로리)", 900, np.inf)]
# 커버리지 부족 구간 = 합성증강이 노리는 표적
LARGE_MASS = 500.0
HIGH_CAL = 900.0


def dish_targets5(split):
    """dish_id -> [calories, fat, carb, protein, mass] (접시당 1벡터)."""
    out = {}
    for it in load_manifest(ROOT / f"{MANIFEST_DIR}/{split}.jsonl"):
        out.setdefault(it["dish_id"], [float(it["targets"][c]) for c in TGT5])
    return out


def usable5(split):
    """rgb/depth가 모두 존재하는 접시만 (기존 실험과 동일 집합)."""
    tg = dish_targets5(split)
    d = [k for k in tg if _ok(OV / k / "rgb.png") and _ok(OV / k / "depth_raw.png")]
    return d, tg


def metrics(pred, tgt, names=TGT5):
    """PMA(%)와 RMSE를 함께 계산.

    PMA_i   = 100 * mean|err| / mean(정답)      (기존 프로젝트 지표, 낮을수록 좋음)
    RMSE_i  = sqrt(mean(err^2))                 (절대 오차 크기, 큰 오차에 민감)
    nRMSE_i = 100 * RMSE_i / mean(정답)         (타깃 간 비교 가능하게 정규화한 RMSE)
    mean_PMA / mean_nRMSE 는 '4영양소'만 평균 → 과거 실험과 직접 비교 가능.
    """
    pred = np.asarray(pred, np.float64); tgt = np.asarray(tgt, np.float64)
    err = pred - tgt
    abs_err = np.abs(err).mean(0)
    rmse = np.sqrt((err ** 2).mean(0))
    denom = np.maximum(tgt.mean(0), 1e-8)
    out = {}
    for i, n in enumerate(names):
        out[f"{n}_PMA"] = float(100.0 * abs_err[i] / denom[i])
        out[f"{n}_RMSE"] = float(rmse[i])
        out[f"{n}_nRMSE"] = float(100.0 * rmse[i] / denom[i])
        out[f"{n}_MAE"] = float(abs_err[i])
    k4 = [i for i, n in enumerate(names) if n in NUTR4]
    out["mean_PMA"] = float(np.mean([100.0 * abs_err[i] / denom[i] for i in k4]))
    out["mean_nRMSE"] = float(np.mean([100.0 * rmse[i] / denom[i] for i in k4]))
    out["n"] = int(len(tgt))
    return out


def bin_index(v, bins):
    for i, (_, lo, hi) in enumerate(bins):
        if lo <= v < hi:
            return i
    return len(bins) - 1


def bucket_metrics(pred, tgt, values, bins, col):
    """구간별 PMA/RMSE. values=구간 기준값(중량 또는 칼로리), col=평가할 타깃 인덱스."""
    pred = np.asarray(pred, np.float64); tgt = np.asarray(tgt, np.float64)
    rows = []
    for bi, (label, lo, hi) in enumerate(bins):
        m = np.array([lo <= v < hi for v in values])
        if m.sum() == 0:
            rows.append({"bin": label, "n": 0, "PMA": np.nan, "RMSE": np.nan, "mean_target": np.nan})
            continue
        e = pred[m, col] - tgt[m, col]
        rows.append({"bin": label, "n": int(m.sum()),
                     "PMA": float(100.0 * np.abs(e).mean() / max(tgt[m, col].mean(), 1e-8)),
                     "RMSE": float(np.sqrt((e ** 2).mean())),
                     "mean_target": float(tgt[m, col].mean())})
    return rows
