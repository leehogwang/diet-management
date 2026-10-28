"""AIHub 122 음식분류 109종 이미지(food_classifier 데이터) → 영양 모델 학습 manifest.

정답: 영양DB(Q4 = 1인분) 값. 음식분류 사진은 양 정보가 없어 모든 사진을 1인분으로 가정한다 [추정].
     같은 음식의 사진은 정답이 모두 같다.

텍스트: 파이프라인(pipeline_infer)과 같은 "{food_name} (재료: {ingredients})" 형식.
     재료는 영양DB 400종마다 Gemini 에 음식명만 주고 한 번 생성해 data/food_ingredients_122.json 에 저장한다.
     (파이프라인은 사진 + 음식명으로 재료를 받으므로 표현이 조금 다를 수 있다.)

분할 (음식 단위로 떼어 두어 처음 보는 음식 성능도 잰다)
  - 대분류마다 음식을 무작위로 골라 val 음식 ~10%, test_unseen 음식 ~15%
  - 나머지(train 음식)는 food_classifier 의 split(train/val/test) 을 따른다
      manifest_cls109.csv          : train 음식의 train·val 사진 = source_split Training,
                                     val 음식의 모든 사진 = Validation (조기 종료 기준 = 처음 보는 음식)
      manifest_cls109_test_seen.csv   : train 음식의 test 사진 (아는 음식)
      manifest_cls109_test_unseen.csv : test_unseen 음식의 모든 사진 (처음 보는 음식)
      manifest_cls109_db.csv       : 영양DB 텍스트 전용 샘플 (val·test_unseen 음식 제외)

사용 예
  python -m n296.build_manifest_cls109 --config configs/cls109_nutrition.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_manifest_122 import load_db
from .common import TARGETS, load_config, resolve

INGREDIENT_PROMPT = ("다음 한국 음식 각각에 일반적으로 들어가는 재료를 한국어로 나열하세요. "
                     "food_name 은 주어진 이름을 그대로 쓰세요.\n{foods}")
INGREDIENT_SCHEMA = {
    "type": "ARRAY",
    "items": {"type": "OBJECT",
              "properties": {"food_name": {"type": "STRING"},
                             "ingredients": {"type": "ARRAY", "items": {"type": "STRING"}}},
              "required": ["food_name", "ingredients"]},
}


def generate_ingredients(names: list[str], out_path: Path, args) -> dict[str, list[str]]:
    """음식명 → 재료 목록 (Gemini, 여러 음식을 한 번에 요청). 이미 있는 음식은 건너뛴다."""
    from google import genai
    from google.genai import types

    from .predict_food_gemini import RateLimiter, clean_list, gemini_json, get_api_key

    table = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    todo = [n for n in names if n not in table]
    print(f"재료: 음식 {len(names)}종 중 새로 생성 {len(todo)}종 (model={args.gemini_model})")
    if not todo:
        return table
    client = genai.Client(api_key=get_api_key(args.key_file), http_options=types.HttpOptions(timeout=120_000))
    limiter = RateLimiter(args.rpm)
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        prompt = INGREDIENT_PROMPT.format(foods="\n".join(f"- {n}" for n in chunk))
        parsed, _, err = gemini_json(client, args.gemini_model, None, prompt, INGREDIENT_SCHEMA, 0.0, 5, limiter)
        if parsed is None:
            raise SystemExit(f"Gemini 재료 생성 실패: {err}")
        got = {str(x.get("food_name", "")).strip(): clean_list(x.get("ingredients")) for x in parsed}
        missing = [n for n in chunk if not got.get(n)]
        table.update({n: got[n] for n in chunk if got.get(n)})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {i + len(chunk)}/{len(todo)}  누락 {len(missing)} {missing[:3]}")
    still = [n for n in names if n not in table]
    if still:
        raise SystemExit(f"재료가 없는 음식 {len(still)}종 (예: {still[:5]}). 다시 실행하면 누락분만 요청합니다.")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--val-food-ratio", type=float, default=0.10)
    ap.add_argument("--test-food-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gemini-model", default="gemini-3.5-flash-lite")
    ap.add_argument("--key-file", default=str(Path(__file__).resolve().parents[1] / ".gemini_key"))
    ap.add_argument("--rpm", type=float, default=14)
    ap.add_argument("--batch", type=int, default=50, help="Gemini 요청 1건에 넣을 음식 수")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]

    db = load_db(Path(d["db_path_122"]))
    ing = generate_ingredients(db["food_name"].tolist(), resolve(d["ingredients_json"]), args)
    db["ingredients"] = db["food_name"].map(lambda n: ", ".join(ing[n]))
    nut = db.set_index("food_name")

    img = pd.read_csv(resolve(d["classifier_split"]), dtype={"code": str})
    img = img.rename(columns={"path": "image_path", "name": "food_name", "split": "cls_split"})
    img["category"] = img["code"].str[:2]
    missing = sorted(set(img["food_name"]) - set(nut.index))
    if missing:
        raise SystemExit(f"영양DB 에 없는 음식 {missing}")

    rng = np.random.default_rng(args.seed)
    role = {}
    for _, g in img.groupby("category"):
        foods = sorted(g["food_name"].unique())
        rng.shuffle(foods)
        n_val = max(1, round(len(foods) * args.val_food_ratio))
        n_test = max(1, round(len(foods) * args.test_food_ratio))
        for i, f in enumerate(foods):
            role[f] = "val" if i < n_val else "test_unseen" if i < n_val + n_test else "train"
    img["food_role"] = img["food_name"].map(role)
    for k in TARGETS + ["ingredients"]:
        img[k] = img["food_name"].map(nut[k])
    img["food_code"] = img["food_name"]  # evaluate.py 음식별 집계 키
    img["split_group"] = img["group"]

    train_foods = img["food_role"] == "train"
    main_df = img[(train_foods & img["cls_split"].isin(["train", "val"])) | (img["food_role"] == "val")].copy()
    main_df["source_split"] = np.where(main_df["food_role"] == "val", "Validation", "Training")
    seen = img[train_foods & (img["cls_split"] == "test")]
    unseen = img[img["food_role"] == "test_unseen"]
    held = set(img.loc[img["food_role"] != "train", "food_name"])
    text_only = db[~db["food_name"].isin(held)].copy()
    text_only["food_code"] = text_only["food_name"]
    text_only["image_path"] = ""

    cols = ["image_path", "food_name", "food_code", "ingredients", *TARGETS, "source_split", "split_group", "category", "food_role"]
    p = cfg["paths"]
    main_df[cols].to_csv(resolve(p["manifest"]), index=False)
    test_cols = [c for c in cols if c != "source_split"]
    seen[test_cols].to_csv(resolve(d["test_seen_manifest"]), index=False)
    unseen[test_cols].to_csv(resolve(d["test_unseen_manifest"]), index=False)
    text_only[["image_path", "food_name", "food_code", "ingredients", *TARGETS]].to_csv(resolve(p["text_only_manifest"]), index=False)

    summary = pd.DataFrame({
        "음식 수": [main_df.loc[main_df.source_split == "Training", "food_name"].nunique(),
                   main_df.loc[main_df.source_split == "Validation", "food_name"].nunique(),
                   seen["food_name"].nunique(), unseen["food_name"].nunique(), len(text_only)],
        "사진 수": [(main_df.source_split == "Training").sum(), (main_df.source_split == "Validation").sum(),
                   len(seen), len(unseen), 0],
    }, index=["train", "val (처음 보는 음식)", "test_seen", "test_unseen", "텍스트 전용(DB)"])
    print(summary.to_string())
    print("val 음식:", sorted(f for f, r in role.items() if r == "val"))
    print("test_unseen 음식:", sorted(f for f, r in role.items() if r == "test_unseen"))


if __name__ == "__main__":
    main()
