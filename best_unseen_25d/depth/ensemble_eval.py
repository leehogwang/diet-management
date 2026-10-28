"""fold별 test_preds.npz 를 모아 5-fold 앙상블 평가 (CPU). 지표 A/B/G, 음식별 오차, 분량별 암시 배율, 음식별 게이트 평균.
usage: python depth/ensemble_eval.py --root OUT   (OUT/fold0..4/test_preds.npz)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_aihub122_multimodal import load_aihub122_dataset, PORTION_FACTORS


def metrics(P, T):
    e = np.abs(P - T)
    return {"A": round(float((e / (np.abs(T) + 1)).mean(0)[:4].mean() * 100), 2),
            "B": round(float((e / np.clip(T, 1, None)).mean(0)[:4].mean() * 100), 2),
            "G": round(float((e.sum(0) / T.sum(0))[:4].mean() * 100), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    recs = load_aihub122_dataset(emb_cache_path=ROOT / "text_baseline_the_best/emb_cache_with_ingredients.json")[0]["test"]
    foods = np.array([r["food_name"] for r in recs])
    q = np.array([r["portion"] for r in recs])
    zs = [np.load(f) for f in sorted(root.glob("fold*/test_preds.npz"))]
    T = zs[0]["target"]
    assert all(np.allclose(z["target"], T) for z in zs) and len(T) == len(recs), "fold 간 테스트 순서 불일치"
    res = {f"fold{k}": metrics(z["pred"], T) for k, z in enumerate(zs)}
    fa = [v["A"] for v in res.values()]
    res["folds_A_mean_std"] = [round(float(np.mean(fa)), 2), round(float(np.std(fa)), 2)]
    ens = np.expm1(np.log1p(np.stack([z["pred"] for z in zs])).mean(0))
    res["ensemble_logmean"] = metrics(ens, T)
    base = T / np.array([PORTION_FACTORS[x] for x in q])[:, None]
    res["portion_factor_median"] = {Q: round(float(np.median(ens[q == Q, 0] / base[q == Q, 0])), 3) for Q in sorted(PORTION_FACTORS)}
    G = np.stack([z["gate"] for z in zs]).mean(0)  # [N,K]
    res["gate_mean"] = [round(float(x), 3) for x in G.mean(0)]
    per = lambda k: float((np.abs(ens[k] - T[k]) / (np.abs(T[k]) + 1)).mean(0)[:4].mean() * 100)
    res["per_food"] = {f: {"A": round(per(foods == f), 2), "gate": [round(float(x), 3) for x in G[foods == f].mean(0)], "n": int((foods == f).sum())}
                       for f in sorted(set(foods), key=lambda f: -per(foods == f))}
    print(json.dumps({k: v for k, v in res.items() if k != "per_food"}, ensure_ascii=False, indent=1))
    (root / "ensemble_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
