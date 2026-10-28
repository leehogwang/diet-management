"""이미지 → 음식명 → 재료·대분류 → Gemini 임베딩 → 영양성분 / 양 등급(이미지 + 대분류) → 양 보정 (추론 파이프라인).

  이미지
   ├─ 1.   음식명: 음식 분류기(n296.food_classifier) top1
   │              1-1. 확신도(보정된 softmax) < --threshold(0.7) 이면 Gemini(--gemini-model)가 사진만 보고 음식명 예측
   ├─ 2.   재료·대분류: Gemini 가 사진 + 1 의 음식명을 보고 재료와 대분류(food_category.CATEGORIES 중 하나)를 함께 예측
   ├─ 3.   텍스트: --text-template 로 음식명·재료를 합쳐 Gemini 임베딩 (캐시 사용)
   ├─ 4-1. 영양:   Stage3NutritionModel(텍스트 임베딩 + 이미지·주파수 특징) → calories/mass/fat/carb/protein
   │              영양 모델은 1인분(Q4) 값으로 학습했으므로 이 값은 1인분 기준이다 → pred_<k>_q4
   ├─ 4-2. 양:     양추정 모델(n296.quantity_estimator)에 사진 + 2 의 대분류 → Q1~Q5
   └─ 5.   보정:   pred_<k> = pred_<k>_q4 × 양 비율 (Q1 0.25 · Q2 0.5 · Q3 0.75 · Q4 1.0 · Q5 1.25)
                  --quantity-ratio expected 면 Q 확률 가중 평균 비율. --skip-quantity 면 비율 1.0
                  --quantity-gate (기본 known_category): 대분류가 기타가 아닐 때만 곱한다 (처음 보는 음식도 대분류로 보정).
                  known_food 면 양추정 모델이 학습한 음식일 때만, always 면 항상. 양 확신도 < --quantity-min-prob 이면 곱하지 않음.

주의: 영양 모델은 학습 때와 같은 텍스트 형식으로 임베딩해야 한다. config 의 data.text_template 과
--text-template 이 다르면 경고를 출력한다 (예: 음식명만으로 학습한 모델에 "음식명 + 재료" 텍스트를 넣는 경우).

Gemini 결과(1·2)는 <out-dir>/pipeline_raw.jsonl 에 한 장씩 저장하고, 다시 실행하면 재사용한다 (무료 등급 한도 절약).
저장된 결과의 음식명 경로(분류기/Gemini)가 지금 --threshold 로 정한 경로와 다르면 그 사진만 다시 처리한다.
양추정(1-1)은 로컬 모델이라 캐시하지 않고 매번 계산한다.

사용 예
  CUDA_VISIBLE_DEVICES=0 python -m n296.pipeline_infer --config configs/cls109_nutrition.yaml \
      --nutrition-ckpt runs/nutrition_cls109/model_stage3_296_final.pth \
      --images <이미지 또는 폴더 ...> --out-dir runs/pipeline_demo
  # 영양 예측 없이 음식명·재료만
  python -m n296.pipeline_infer --config configs/cls109_nutrition.yaml --skip-nutrition --images ... --out-dir ...
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from tqdm import tqdm

from .common import TARGETS, load_config, resolve
from .embeddings import ensure_embeddings
from .food_category import CATEGORIES, CATEGORY_NAMES, UNKNOWN, normalize_category
from .food_classifier import IMAGE_EXTS, RUN_DIR, FoodClassifier
from .quantity_estimator import RUN_DIR as QUANTITY_RUN_DIR
from .predict_food_gemini import RateLimiter, clean_list, gemini_json, get_api_key, is_daily_quota

NAME_PROMPT = "이 사진 속 음식의 이름을 예측하세요.\n- food_name: 음식 이름 (한국어)"
NAME_SCHEMA = {"type": "OBJECT", "properties": {"food_name": {"type": "STRING"}}, "required": ["food_name"]}
ING_PROMPT = ("이 사진 속 음식은 '{food_name}'입니다. 이 음식에 들어간 재료와 음식의 대분류를 예측하세요.\n"
              "- ingredients: 재료 목록 (한국어)\n"
              "- category: 음식의 대분류. 아래 중 하나를 그대로 고르세요 (괄호는 예시).\n"
              + "\n".join(f"  {c} ({ex})" for c, ex in CATEGORIES.items()))
ING_SCHEMA = {"type": "OBJECT",
              "properties": {"ingredients": {"type": "ARRAY", "items": {"type": "STRING"}},
                             "category": {"type": "STRING", "enum": CATEGORY_NAMES}},
              "required": ["ingredients", "category"]}
DEFAULT_TEMPLATE = "{food_name} (재료: {ingredients})"


def list_images(items: list[str]) -> list[Path]:
    out = []
    for it in items:
        if not str(it).strip():  # 빈 문자열은 Path("") = 현재 폴더가 되어 폴더 전체를 처리하게 된다
            raise SystemExit("--images 에 빈 경로가 있습니다.")
        p = Path(it)
        if not p.exists():
            raise SystemExit(f"이미지 경로가 없습니다: {p}")
        out += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]
    return out


def jpeg_bytes(img: Image.Image, max_side: int = 1024) -> bytes:
    """Gemini 로 보낼 이미지. 원본(최대 4000px)을 그대로 보내면 느리고 토큰이 많아 줄인다."""
    import io

    img = img.copy()
    img.thumbnail((max_side, max_side), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def route_matches(rec: dict, threshold: float) -> bool:
    """저장된 결과의 음식명 경로가 현재 threshold 로 정해지는 경로와 같은가.
    대분류(category)가 없는 예전 결과(stage3_2 이전 프롬프트)는 다시 처리한다."""
    want = "classifier" if rec["cls_prob"] >= threshold else "gemini"
    return rec.get("name_source") == want and "category" in rec


def recognize(paths: list[Path], args) -> list[dict]:
    """1·2 단계: 음식명(분류기 → 필요 시 Gemini) + 재료·대분류(Gemini). jsonl 로 이어서 실행 가능."""
    from google import genai
    from google.genai import types

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "pipeline_raw.jsonl"
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            # 같은 사진이 여러 줄이면 마지막 줄이 최신 (threshold 변경으로 다시 처리한 경우)
            if not r.get("error") and route_matches(r, args.threshold):
                done[r["image_path"]] = r
    todo = [p for p in paths if str(p) not in done]
    print(f"이미지 {len(paths)}장, 이전 결과 재사용 {len(paths) - len(todo)}장, 새로 처리 {len(todo)}장")
    if not todo:
        return [done[str(p)] for p in paths]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = FoodClassifier(resolve(args.classifier_ckpt), device)
    client = genai.Client(api_key=get_api_key(args.key_file), http_options=types.HttpOptions(timeout=90_000))
    limiter = RateLimiter(args.rpm)
    call = lambda img_bytes, prompt, schema: gemini_json(client, args.gemini_model, img_bytes, prompt, schema,
                                                         0.0, args.max_retries, limiter)
    with open(raw_path, "a", encoding="utf-8") as f:
        for p in tqdm(todo, desc="recognize"):
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            top = clf.predict(img, topk=5)
            rec = {"image_path": str(p), "cls_top5": top, "cls_name": top[0]["name"], "cls_prob": top[0]["prob"],
                   "threshold": args.threshold, "gemini_name": None, "error": None}
            img_bytes = jpeg_bytes(img)
            if top[0]["prob"] >= args.threshold:
                rec["name_source"], rec["food_name"] = "classifier", top[0]["name"]
            else:
                parsed, _, err = call(img_bytes, NAME_PROMPT, NAME_SCHEMA)
                rec["name_source"] = "gemini"
                rec["gemini_name"] = rec["food_name"] = (parsed or {}).get("food_name", "").strip() or None
                rec["error"] = err if parsed is None else None
            if rec["food_name"]:
                parsed, _, err = call(img_bytes, ING_PROMPT.format(food_name=rec["food_name"]), ING_SCHEMA)
                rec["ingredients"] = clean_list((parsed or {}).get("ingredients"))
                rec["category"] = normalize_category((parsed or {}).get("category"))
                rec["error"] = rec["error"] or (err if parsed is None else None)
            else:
                rec["ingredients"], rec["category"] = [], UNKNOWN
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if rec["error"]:
                print(f"\n  {p.name}: {rec['error'][:160]}")
                if is_daily_quota(rec["error"]):
                    print("Gemini 일일 한도 도달 — 여기까지 저장하고 중지합니다. 한도 초기화 후 같은 명령으로 이어서 실행하세요.")
                    break
            else:
                done[str(p)] = rec
    return [done[str(p)] for p in paths if str(p) in done]


def estimate_quantity(recs: list[dict], args) -> None:
    """4-2 단계: 사진 + 대분류 → 양 등급 Q1~Q5. rec 에 q_level·q_prob·q_ratio·q_ratio_expected·q_probs·q_category,
    보정 적용 여부 q_applied 와 영양값에 곱할 scale 을 기록한다."""
    if args.skip_quantity:
        for r in recs:
            r.update(q_level=None, q_prob=None, q_ratio=None, q_ratio_expected=None, q_probs=None,
                     q_applied=False, scale=1.0)
        return
    from .quantity_estimator import QuantityEstimator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    est = QuantityEstimator(resolve(args.quantity_ckpt), device)
    for r in tqdm(recs, desc="quantity"):
        r.update(est.predict(Image.open(r["image_path"]), r.get("category")))
        known = {"always": True, "known_food": r["food_name"] in est.foods,
                 "known_category": r["q_category"] != UNKNOWN}[args.quantity_gate]
        r["q_applied"] = known and r["q_prob"] >= args.quantity_min_prob
        ratio = r["q_ratio_expected"] if args.quantity_ratio == "expected" else r["q_ratio"]
        r["scale"] = ratio if r["q_applied"] else 1.0


def add_gate_args(ap) -> None:
    ap.add_argument("--quantity-gate", choices=["known_category", "known_food", "always"], default="known_category",
                    help="known_category: 대분류가 기타가 아닐 때만 보정, known_food: 양추정 모델이 학습한 음식일 때만, "
                         "always: 항상 보정")
    ap.add_argument("--quantity-min-prob", type=float, default=0.0, help="양 확신도가 이 값 미만이면 보정하지 않음")


def apply_quantity(recs: list[dict]) -> None:
    """5 단계: 1인분(Q4) 기준 영양 예측에 양 비율을 곱한다. 보정 전 값은 pred_<k>_q4 로 남긴다."""
    for r in recs:
        for k in TARGETS:
            r[f"pred_{k}_q4"] = r[f"pred_{k}"]
            r[f"pred_{k}"] = r[f"pred_{k}_q4"] * r["scale"]


def render_text(rec: dict, template: str) -> str:
    return template.format(food_name=rec["food_name"], ingredients=", ".join(rec["ingredients"])).strip()


def predict_nutrition(recs: list[dict], cfg: dict, args) -> None:
    from .model import Stage3NutritionModel

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.nutrition_ckpt, map_location="cpu", weights_only=False)
    if args.raw_output:
        stats = {k: {"mean": 0.0, "std": 1.0} for k in TARGETS}
    elif args.stats_json:
        stats = json.loads(Path(args.stats_json).read_text())
    elif "target_stats" in ck:
        stats = ck["target_stats"]
    else:
        raise SystemExit("영양 체크포인트에 target_stats 가 없습니다. --raw-output 또는 --stats-json 을 지정하세요.")

    if os.environ.get(cfg["embedding"].get("api_key_env", "GEMINI_KEY")) is None:
        os.environ[cfg["embedding"].get("api_key_env", "GEMINI_KEY")] = get_api_key(args.key_file)
    texts = [r["text"] for r in recs]
    cache = ensure_embeddings(texts, resolve(cfg["paths"]["emb_cache"]), cfg["embedding"])

    model = Stage3NutritionModel(cfg, pretrained=False).to(device).eval()
    model.load_state_dict(ck.get("state_dict", ck))
    size = cfg["data"]["image_size"]
    tf = transforms.Compose([transforms.Resize(int(size * 256 / 224)), transforms.CenterCrop(size), transforms.ToTensor()])
    bs = 16
    for i in tqdm(range(0, len(recs), bs), desc="nutrition"):
        chunk = recs[i:i + bs]
        image = torch.stack([tf(ImageOps.exif_transpose(Image.open(r["image_path"])).convert("RGB")) for r in chunk]).to(device)
        emb = torch.tensor(np.array([cache[r["text"]] for r in chunk]), dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(image, emb, torch.ones(len(chunk), dtype=torch.bool, device=device), image_mode="full")
        for j, r in enumerate(chunk):
            for k in TARGETS:
                r[f"pred_{k}"] = float(pred[k][j, 0]) * stats[k]["std"] + stats[k]["mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="영양 모델 config (embedding·model·data 설정)")
    ap.add_argument("--images", nargs="+", required=True, help="이미지 파일 또는 폴더")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--classifier-ckpt", default=f"{RUN_DIR}/best.pth")
    ap.add_argument("--threshold", type=float, default=0.7, help="분류기 확신도가 이 값 미만이면 Gemini 로 음식명 예측")
    ap.add_argument("--quantity-ckpt", default=f"{QUANTITY_RUN_DIR}/best.pth")
    ap.add_argument("--quantity-ratio", choices=["argmax", "expected"], default="argmax",
                    help="argmax: 가장 확률 높은 Q 의 비율, expected: Q 확률로 가중 평균한 비율")
    ap.add_argument("--skip-quantity", action="store_true", help="양 보정 없이 1인분(Q4) 기준 값 그대로 출력")
    add_gate_args(ap)
    ap.add_argument("--gemini-model", default="gemini-3.5-flash-lite")
    ap.add_argument("--key-file", default=str(Path(__file__).resolve().parents[1] / ".gemini_key"))
    ap.add_argument("--rpm", type=float, default=14)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--text-template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--nutrition-ckpt", default=None)
    ap.add_argument("--raw-output", action="store_true", help="영양 체크포인트 출력을 그대로 원래 단위로 간주")
    ap.add_argument("--stats-json", default=None)
    ap.add_argument("--skip-nutrition", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = list_images(args.images)[: args.limit]
    recs = recognize(paths, args)
    estimate_quantity(recs, args)
    for r in recs:
        r["text"] = render_text(r, args.text_template)

    if not args.skip_nutrition:
        if not args.nutrition_ckpt:
            raise SystemExit("--nutrition-ckpt 가 필요합니다 (영양 예측을 건너뛰려면 --skip-nutrition).")
        if cfg["data"].get("text_template") != args.text_template:
            print(f"⚠ 영양 모델 config 의 text_template({cfg['data'].get('text_template')!r}) 과 "
                  f"--text-template({args.text_template!r}) 이 다릅니다. 학습 때와 다른 텍스트로 임베딩하므로 예측이 부정확할 수 있습니다.")
        predict_nutrition(recs, cfg, args)
        apply_quantity(recs)

    out_dir = resolve(args.out_dir)
    (out_dir / "pipeline_results.json").write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
    df = pd.DataFrame(recs)
    if not df.empty:
        df["ingredients"] = df["ingredients"].map(" | ".join)
        df["cls_top5"] = df["cls_top5"].map(lambda t: " | ".join(f"{x['name']}:{x['prob']:.2f}" for x in t))
        df["q_probs"] = df["q_probs"].map(lambda d: " | ".join(f"{q}:{p:.2f}" for q, p in d.items()) if d else "")
    df.to_csv(out_dir / "pipeline_results.csv", index=False, encoding="utf-8-sig")
    if not df.empty:
        print(df["name_source"].value_counts().to_string())
        print(df["category"].value_counts().to_string())
        if not args.skip_quantity:
            print(df["q_level"].value_counts().sort_index().to_string())
            print(f"양 보정 적용 {int(df['q_applied'].sum())}/{len(df)}장")
    print(f"저장: {out_dir}/pipeline_results.json, .csv ({len(recs)}/{len(paths)}장)")


if __name__ == "__main__":
    main()
