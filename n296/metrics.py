"""PMAE 계산. 체크포인트 val_metrics 와 같은 키를 쓴다.

PMAE(%) = 100 × MAE / mean(GT)   [추정: Nutrition5k 논문 정의. 샘플별 |오차|/GT 평균일 수도 있음]
mean_pmae = (cal + fat + prot) / 3  (체크포인트 수치로 확인됨)
"""
from __future__ import annotations

import numpy as np

from .common import MEAN_PMAE_KEYS, TARGETS, atwater_carb


def pmae(pred: np.ndarray, gt: np.ndarray) -> float:
    ok = ~np.isnan(gt)
    if not ok.any():
        return float("nan")
    return float(100 * np.abs(pred[ok] - gt[ok]).mean() / (np.abs(gt[ok]).mean() + 1e-8))


def compute_metrics(pred: dict[str, np.ndarray], gt: dict[str, np.ndarray]) -> dict:
    m = {
        "cal": pmae(pred["calories"], gt["calories"]),
        "fat": pmae(pred["fat"], gt["fat"]),
        "prot": pmae(pred["protein"], gt["protein"]),
        "mass": pmae(pred["mass"], gt["mass"]),
        "carb_head": pmae(pred["carb"], gt["carb"]),
        "carb_atwater": pmae(atwater_carb(pred["calories"], pred["fat"], pred["protein"]), gt["carb"]),
    }
    m["mean_pmae"] = float(np.mean([m[k] for k in MEAN_PMAE_KEYS]))
    return m


def collect(outputs: list[dict], stats: dict) -> tuple[dict, dict]:
    """배치별 정규화된 예측을 원래 단위로 되돌려 모은다."""
    pred = {k: [] for k in TARGETS}
    gt = {k: [] for k in TARGETS}
    for o in outputs:
        for j, k in enumerate(TARGETS):
            p = o["pred"][k].squeeze(1).float().cpu().numpy() * stats[k]["std"] + stats[k]["mean"]
            g = o["y"][:, j].cpu().numpy().copy()
            g[~o["mask"][:, j].cpu().numpy()] = np.nan
            pred[k].append(p)
            gt[k].append(g)
    return ({k: np.concatenate(v) for k, v in pred.items()}, {k: np.concatenate(v) for k, v in gt.items()})
