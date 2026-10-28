"""stage3_3 파이프라인의 Nutrition5k test 평가 (접시당 사진 1장).

입력
  runs/eval_n5k/pipeline_results.csv  : n296.pipeline_infer 결과 (selected_images.csv 의 709장)
  ../n296_stage3_4/data/n5k/manifest_test.csv : 정답 (접시 전체의 실제 값)
  ../n296_stage3_4/runs/eval_n5k_ingr/image_results.csv : stage3_4 모델의 사진별 예측 (같은 사진으로 비교)

출력: runs/eval_n5k/summary.json, dish_results.csv, detail_metrics.csv

사용 예 (n296_stage3_3 에서)
  ~/.conda/envs/diet/bin/python runs/eval_n5k/eval_n5k_stage3_3.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from n296.common import TARGETS  # noqa: E402
from n296.metrics import compute_metrics  # noqa: E402

S4 = ROOT.parent / "n296_stage3_4"
OUT = ROOT / "runs/eval_n5k"
KEYS = ("cal", "mass", "fat", "carb_head", "prot", "mean_pmae")
MAX_KCAL = 3000


def pmae(df, prefix):
    m = compute_metrics({k: df[f"{prefix}_{k}"].values for k in TARGETS}, {k: df[k].values for k in TARGETS})
    return {k: round(float(m[k]), 2) for k in KEYS}


def detail(df, prefix):
    rows = []
    for k in TARGETS:
        g, p = df[k].values, df[f"{prefix}_{k}"].values
        e = p - g
        ae = np.abs(e)
        rel = ae[g > 0] / g[g > 0]
        rows.append({"model": prefix, "target": k, "gt_mean": g.mean(), "pred_mean": p.mean(), "MAE": ae.mean(),
                     "PMAE": 100 * ae.mean() / g.mean(), "MedAE": np.median(ae), "RMSE": np.sqrt((e ** 2).mean()),
                     "bias": e.mean(), "R2": 1 - (e ** 2).sum() / ((g - g.mean()) ** 2).sum(),
                     "within20": 100 * (rel <= 0.2).mean()})
    return rows


def main():
    sel = pd.read_csv(OUT / "selected_images.csv")
    res = pd.read_csv(OUT / "pipeline_results.csv")
    gt = pd.read_csv(S4 / "data/n5k/manifest_test.csv").drop_duplicates("dish_id")
    d = sel.merge(res, on="image_path", how="inner").merge(gt[["dish_id", "ingredients"] + TARGETS].rename(
        columns={"ingredients": "gt_ingredients"}), on="dish_id")
    for k in TARGETS:
        d[f"stage3_3_{k}"] = d[f"pred_{k}"]          # 양 보정 후 (실제 파이프라인 출력)
        d[f"stage3_3_q4_{k}"] = d[f"pred_{k}_q4"]    # 양 보정 전 (1인분 기준)
        d[f"trainmean_{k}"] = np.nan
    s4 = pd.read_csv(S4 / "runs/eval_n5k_ingr/image_results.csv")
    s4 = s4.set_index("image_path")
    for k in TARGETS:
        d[f"stage3_4_{k}"] = d["image_path"].map(s4[f"pipeline_{k}"])
    tr = pd.read_csv(S4 / "data/n5k/manifest.csv")
    tr = tr[tr.split == "train"].drop_duplicates("dish_id")
    for k in TARGETS:
        d[f"trainmean_{k}"] = tr[k].mean()
    clean = d[(d.calories > 0) & (d.calories <= MAX_KCAL)]

    models = {"stage3_3": "stage3_3 (양 보정 후)", "stage3_3_q4": "stage3_3 (양 보정 전, 1인분)",
              "stage3_4": "stage3_4 pipeline (같은 사진 1장)", "trainmean": "평균값 예측"}
    summary = {"images": len(d), "selected": len(sel), "clean_dishes": len(clean),
               "views": d.view.value_counts().to_dict(),
               "name_source": d.name_source.value_counts().to_dict(),
               "quantity_applied": int(d.q_applied.sum()),
               "q_level": d.q_level.value_counts().sort_index().to_dict(),
               "all": {m: pmae(d, m) for m in models}, "clean": {m: pmae(clean, m) for m in models},
               "labels": models}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([r for m in models for r in detail(clean, m)]).round(3).to_csv(OUT / "detail_metrics.csv", index=False)
    cols = ["dish_id", "view", "image_path", "cls_name", "cls_prob", "name_source", "food_name", "ingredients", "category",
            "q_level", "q_applied", "gt_ingredients"] + TARGETS + [f"{m}_{k}" for m in models if m != "trainmean" for k in TARGETS]
    d[cols].to_csv(OUT / "dish_results.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pd.read_csv(OUT / "detail_metrics.csv").to_string(index=False))


if __name__ == "__main__":
    main()
