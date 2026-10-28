"""파이프라인 전체 평가: 이미지 → (분류기 / Gemini) 음식명 → Gemini 재료·대분류 → 영양 모델 × 양 비율(사진 + 대분류).

테스트 세트 (Gemini 무료 등급 한도 때문에 음식마다 일부만 뽑는다)
  aihub122 : 분류기 test 분할(data/food_cls_122/split.csv)에서 음식마다 --n122 장. 분류기가 아는 109종.
             정답 = 영양DB 1인분. nut_seen = 영양 모델 학습에 쓴 음식인지 (False 면 영양 모델에는 처음 보는 음식)
  test296  : test_dataset_296 에서 음식마다 --n296 장 (서로 다른 샘플). 분류기가 모르는 19종.
             정답 = 296 라벨의 1인분 값 (영양DB 와 1인분 기준이 다르다)
  quantity122 : 양추정 test 분할(data/quantity_122/split.csv, 공식 Validation)에서 분류기가 아는 음식만,
             음식 × Q 등급마다 --nq 장. 정답 = 영양DB × Q 비율 (사진마다 실제 양이 다름).
             split.csv 가 없으면 (양추정 이미지 미준비) 건너뛴다.

aihub122·test296 의 정답은 1인분(Q4)을 가정한 값이라 양 보정이 오히려 오차를 만들 수 있다.
그래서 모든 세트에서 양 보정 후(pipeline)와 보정 전(pipeline_q4) 을 함께 보고한다.
양 보정 효과는 실제 양 정답이 있는 quantity122 에서 판단한다.

비교 기준 (aihub122·quantity122)
  oracle   : 정답 음식명 + 영양DB 재료(data/food_ingredients_122.json) 텍스트로 같은 영양 모델 예측.
             양 비율은 pipeline 과 같은 예측값을 쓴다. pipeline 과 oracle 의 차이 = 음식명·재료 인식에서 생긴 오차.
  oracle_q : (quantity122 만) oracle 에 정답 Q 비율을 곱한 값. oracle 과의 차이 = 양추정에서 생긴 오차.

category_acc : Gemini 대분류가 사람이 정한 대분류(food_category.FOOD_CATEGORY)와 같은 비율.

출력: <out-dir>/results.csv (사진별), summary.json, pipeline_raw.jsonl (Gemini 결과 캐시, 다시 실행 시 재사용)

사용 예
  CUDA_VISIBLE_DEVICES=0 python -m n296.eval_pipeline --config configs/cls109_nutrition.yaml \
      --nutrition-ckpt runs/nutrition_cls109/model_stage3_296_final.pth --out-dir runs/pipeline_eval
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_manifest_122 import load_db
from .common import TARGETS, load_config, resolve
from .food_category import FOOD_CATEGORY
from .metrics import compute_metrics
from .pipeline_infer import (DEFAULT_TEMPLATE, add_gate_args, apply_quantity, estimate_quantity, predict_nutrition,
                             recognize, render_text)
from .quantity_estimator import DATA_DIR as QUANTITY_DATA_DIR

T296 = "data/manifest_test296.csv"
ORACLE_SETS = ("aihub122", "quantity122")


def sample_122(cfg, n: int, seed: int) -> pd.DataFrame:
    sp = pd.read_csv(resolve("data/food_cls_122/split.csv"), dtype={"code": str})
    t = sp[sp["split"] == "test"].drop_duplicates("group")
    t = t.groupby("name", group_keys=False).apply(lambda g: g.sample(min(n, len(g)), random_state=seed))
    db = load_db(Path(cfg["data"]["db_path_122"])).set_index("food_name")
    nut_train = set(pd.read_csv(resolve(cfg["paths"]["manifest"])).query("source_split == 'Training'")["food_name"])
    out = pd.DataFrame({"set": "aihub122", "image_path": t["path"].values, "gt_name": t["name"].values})
    for k in TARGETS:
        out[f"gt_{k}"] = out["gt_name"].map(db[k])
    out["nut_seen"] = out["gt_name"].isin(nut_train)
    return out


def sample_quantity(cfg, n: int, seed: int, known_foods: set[str]) -> pd.DataFrame:
    path = resolve(QUANTITY_DATA_DIR) / "split.csv"
    if n <= 0 or not path.exists():
        if n > 0:
            print(f"⚠ {path} 가 없어 quantity122 세트를 건너뜁니다 (python -m n296.quantity_estimator prepare 먼저).")
        return pd.DataFrame()
    sp = pd.read_csv(path)
    t = sp[(sp["split"] == "test") & sp["food_name"].isin(known_foods)].drop_duplicates("split_group")
    t = t.groupby(["food_name", "q_level"], group_keys=False).apply(lambda g: g.sample(min(n, len(g)), random_state=seed))
    nut_train = set(pd.read_csv(resolve(cfg["paths"]["manifest"])).query("source_split == 'Training'")["food_name"])
    out = pd.DataFrame({"set": "quantity122", "image_path": t["path"].values, "gt_name": t["food_name"].values,
                        "gt_q_level": t["q_level"].values, "gt_q_ratio": t["q_ratio"].values})
    for k in TARGETS:
        out[f"gt_{k}"] = t[k].values  # 영양DB × Q 비율 (build_manifest_122)
    out["nut_seen"] = out["gt_name"].isin(nut_train)
    return out


def sample_296(n: int, seed: int) -> pd.DataFrame:
    m = pd.read_csv(resolve(T296))
    m["sample"] = m["file_name"].str.extract(r"_(\d+)_\d+\.\w+$")[0]
    m = m.drop_duplicates(["food_name", "sample"])
    t = m.groupby("food_name", group_keys=False).apply(lambda g: g.sample(min(n, len(g)), random_state=seed))
    out = pd.DataFrame({"set": "test296", "image_path": t["image_path"].values, "gt_name": t["food_name"].values})
    for k in TARGETS:
        out[f"gt_{k}"] = t[k].values
    out["nut_seen"] = False
    return out


def pmae(df: pd.DataFrame, prefix: str = "pred") -> dict:
    if df.empty:
        return {}
    m = compute_metrics({k: df[f"{prefix}_{k}"].values for k in TARGETS}, {k: df[f"gt_{k}"].values for k in TARGETS})
    return {k: round(float(v), 1) for k, v in m.items() if k in ("cal", "fat", "prot", "mass", "carb_head", "mean_pmae")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--nutrition-ckpt", required=True)
    ap.add_argument("--out-dir", default="runs/pipeline_eval")
    ap.add_argument("--n122", type=int, default=1, help="aihub122 음식당 사진 수")
    ap.add_argument("--n296", type=int, default=3, help="test296 음식당 사진 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nq", type=int, default=1, help="quantity122 음식 × Q 등급당 사진 수 (0 = 사용 안 함)")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--quantity-ckpt", default="runs/quantity_122_cat/best.pth")
    ap.add_argument("--quantity-ratio", choices=["argmax", "expected"], default="argmax")
    ap.add_argument("--skip-quantity", action="store_true")
    add_gate_args(ap)
    ap.add_argument("--gemini-model", default="gemini-3.5-flash-lite")
    ap.add_argument("--key-file", default=str(Path(__file__).resolve().parents[1] / ".gemini_key"))
    ap.add_argument("--rpm", type=float, default=14)
    args = ap.parse_args()
    args.classifier_ckpt = "runs/food_cls_122/best.pth"
    args.max_retries, args.raw_output, args.stats_json = 5, False, None
    cfg = load_config(args.config)
    template = cfg["data"]["text_template"]
    if template != DEFAULT_TEMPLATE:
        print(f"⚠ config text_template {template!r} 이 파이프라인 기본값과 다릅니다. config 값을 씁니다.")

    s122 = sample_122(cfg, args.n122, args.seed)
    known = set(pd.read_csv(resolve("data/food_cls_122/split.csv"))["name"])  # 분류기가 아는 음식
    samples = pd.concat([s122, sample_296(args.n296, args.seed), sample_quantity(cfg, args.nq, args.seed, known)],
                        ignore_index=True)
    print(samples.groupby("set").agg(images=("image_path", "size"), foods=("gt_name", "nunique")).to_string())

    recs = recognize([Path(p) for p in samples["image_path"]], args)
    got = {r["image_path"] for r in recs}
    if len(got) < len(samples):
        print(f"⚠ Gemini 결과가 없는 사진 {len(samples) - len(got)}장은 제외 (일일 한도 등). 같은 명령으로 이어서 실행 가능.")
    samples = samples[samples["image_path"].isin(got)].reset_index(drop=True)
    by_path = {r["image_path"]: r for r in recs}
    recs = [by_path[p] for p in samples["image_path"]]
    estimate_quantity(recs, args)
    for r in recs:
        r["text"] = render_text(r, template)
    predict_nutrition(recs, cfg, args)
    apply_quantity(recs)

    # oracle: 정답 음식명 + 영양DB 재료 (aihub122·quantity122). 양 비율은 pipeline 예측값 (deepcopy 로 이어받음)
    ing = json.loads(resolve(cfg["data"]["ingredients_json"]).read_text(encoding="utf-8"))
    oracle = []
    for (_, s), r in zip(samples.iterrows(), recs):
        if s["set"] in ORACLE_SETS:
            o = copy.deepcopy(r)
            o.update(food_name=s["gt_name"], ingredients=ing[s["gt_name"]])
            o["text"] = render_text(o, template)
            oracle.append(o)
    if oracle:
        predict_nutrition(oracle, cfg, args)
        apply_quantity(oracle)
    oracle_by = {o["image_path"]: o for o in oracle}

    res = samples.copy()
    for col in ("cls_name", "cls_prob", "name_source", "food_name", "text", "category", "q_level", "q_prob", "q_applied",
                "scale"):
        res[col] = [r[col] for r in recs]
    res["ingredients"] = [" | ".join(r["ingredients"]) for r in recs]
    for k in TARGETS:
        res[f"pred_{k}"] = [r[f"pred_{k}"] for r in recs]
        res[f"q4_{k}"] = [r[f"pred_{k}_q4"] for r in recs]  # 양 보정 전 (1인분 기준)
        res[f"oracle_{k}"] = [oracle_by[p][f"pred_{k}"] if p in oracle_by else np.nan for p in res["image_path"]]
        if "gt_q_ratio" in res:  # 정답 음식명·재료 + 정답 양
            res[f"oracle_q_{k}"] = np.array([oracle_by[p][f"pred_{k}_q4"] if p in oracle_by else np.nan
                                             for p in res["image_path"]]) * res["gt_q_ratio"].values
    res["name_correct"] = res["food_name"] == res["gt_name"]
    res["gt_category"] = res["gt_name"].map(FOOD_CATEGORY)  # 사람이 정한 대분류 (test296 은 없음)
    res["category_correct"] = res["category"] == res["gt_category"]
    res["cls_correct"] = res["cls_name"] == res["gt_name"]

    summary = {}
    for name, g in res.groupby("set"):
        s = {"images": len(g), "foods": int(g["gt_name"].nunique()),
             "to_gemini": round(float((g["name_source"] == "gemini").mean()), 3),
             "name_acc": round(float(g["name_correct"].mean()), 3),
             "category_acc": round(float(g.loc[g["gt_category"].notna(), "category_correct"].mean()), 3)
             if g["gt_category"].notna().any() else None,
             "pipeline": pmae(g), "pipeline_q4": pmae(g, "q4")}
        if not args.skip_quantity:
            s["q_level_counts"] = g["q_level"].value_counts().sort_index().to_dict()
            s["q_applied"] = round(float(g["q_applied"].mean()), 3)
        for src, gg in g.groupby("name_source"):
            s[f"by_source_{src}"] = {"images": len(gg), "name_acc": round(float(gg["name_correct"].mean()), 3), **pmae(gg)}
        if name == "quantity122":
            qi = {q: i for i, q in enumerate(["Q1", "Q2", "Q3", "Q4", "Q5"])}
            if not args.skip_quantity:
                s["q_acc"] = round(float((g["q_level"] == g["gt_q_level"]).mean()), 3)
                s["q_acc_within1"] = round(float((g["q_level"].map(qi) - g["gt_q_level"].map(qi)).abs().le(1).mean()), 3)
            s["oracle_q"] = pmae(g, "oracle_q")
            for q, gg in g.groupby("gt_q_level"):
                s[f"gt_{q}"] = {"images": len(gg), "pipeline": pmae(gg), "pipeline_q4": pmae(gg, "q4")}
        if name in ORACLE_SETS:
            s["oracle"] = pmae(g, "oracle")
            for seen, gg in g.groupby("nut_seen"):
                key = "nut_seen" if seen else "nut_unseen"
                s[key] = {"images": len(gg), "pipeline": pmae(gg), "oracle": pmae(gg, "oracle")}
            for ok, gg in g.groupby("name_correct"):
                s["name_correct" if ok else "name_wrong"] = {"images": len(gg), "pipeline": pmae(gg), "oracle": pmae(gg, "oracle")}
        summary[name] = s

    out = resolve(args.out_dir)
    res.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"저장: {out}/results.csv, summary.json")


if __name__ == "__main__":
    main()
