"""122 텍스트를 파이프라인 형식 "{음식명} (재료: {재료1}, {재료2}, …)" (n296.pipeline_infer.DEFAULT_TEMPLATE)로 만들고, 대분류를 함께 준다.
test 조건 (cond_inputs):
  gt     : 정답 음식명 + data/food_ingredients_122.json 재료 (nutrition_cls109 학습과 같은 재료) / 정답표 대분류(food_category.FOOD_CATEGORY)
  gemini : Gemini 음식명(사진만) + Gemini 재료·대분류(사진 + 그 음식명, 파이프라인 2단계)
  pipe   : 파이프라인 그대로. 음식명 = 분할별 분류기 top1 (보정 확신도 >= 0.7) 아니면 Gemini 음식명 → Gemini 재료·대분류(사진 + 그 음식명)
  pipe_osr : pipe 와 같되 분류기 사용 여부를 open-set 판정(cls_osr_<split>_baseline 의 route)으로 정함
학습은 항상 gt. N5k 텍스트는 기존과 같다 (재료 목록, 대분류 기타). 임베딩은 EXP/emb_cache_ing.npz (texts, emb float32).
Gemini 재료·대분류는 EXP/gemini_ing.jsonl ({"id", "food", "ingredients", "category"}), (id, 음식명) 쌍마다 한 줄."""
import json, random, sys
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parents[1]
REPO = Path("/home/leehogwang/food_b")
sys.path.insert(0, str(REPO))
from n296.food_category import CATEGORY_NAMES, FOOD_CATEGORY, UNKNOWN  # noqa: E402

TEMPLATE = "{food_name} (재료: {ingredients})"
THRESHOLD = 0.7  # pipeline_infer --threshold 기본값
ING = json.loads((REPO / "data/food_ingredients_122.json").read_text())


def render(name, ingredients): return TEMPLATE.format(food_name=name, ingredients=", ".join(ingredients)).strip()


def gt_text(r):
    if r["ds"] == "122": return render(r["food"], ING[r["food"]])
    ing = list(dict.fromkeys(r["ingr"])); random.Random(r["name"]).shuffle(ing); return ", ".join(ing)


def gt_cat(r):  # train_quantity.category 와 같다
    if r["ds"] != "122": return UNKNOWN
    return FOOD_CATEGORY.get(r["food"]) or FOOD_CATEGORY.get(r["raw_food"]) or UNKNOWN


def cat_index(c): return CATEGORY_NAMES.index(c if c in CATEGORY_NAMES else UNKNOWN)


def gemini_names():
    out = {}
    for l in open(EXP / "gemini_pred.jsonl"):
        j = json.loads(l)
        if "food" in j: out[j["id"]] = j["food"]
    return out


def pipe_names(split, osr=False):
    """{id: (음식명, 출처)}. 분류기 test_pred 가 없으면 빈 dict.
    osr=False: 기존 분류기 runs/cls_<split> + softmax 확신도 >= 0.7 규칙.
    osr=True : 같은 분류기를 open-set 점수로 판정한 runs/cls_osr_<split>_baseline 의 route (Mahalanobis + val 균형 threshold,
               처음 보는 음식 val 이 없는 분할은 known val 99% 통과). 분할 test 사진만 쓴다 (test_pred 에는 train 에서 떼어낸 known test 도 있음)."""
    f = EXP / "runs" / (f"cls_osr_{split}_baseline" if osr else f"cls_{split}") / "test_pred.json"
    if not f.exists(): return {}
    cls = json.loads(f.read_text()); gem = gemini_names(); out = {}
    test = set(json.loads((EXP / "splits.json").read_text())[split]["test"]) if osr else None
    for i, p in cls.items():
        i = int(i)
        if test is not None and i not in test: continue
        use = p["route"] == "classifier" if osr else p["prob"][0] >= THRESHOLD
        out[i] = (p["top"][0], "classifier") if use else (gem[i], "gemini")
    return out


def ing_records():
    f = EXP / "gemini_ing.jsonl"; out = {}
    if f.exists():
        for l in open(f):
            j = json.loads(l); out[(j["id"], j["food"])] = j
    return out


def n5k_pred():
    out = {}
    for l in open(EXP / "gemini_pred.jsonl"):
        j = json.loads(l)
        if "ingr" in j: out[j["id"]] = (", ".join(dict.fromkeys(j["ingr"])), UNKNOWN)
    return out


def cond_inputs(cond, split):
    """{id: (텍스트, 대분류)} — Gemini 결과가 아직 없는 사진은 빠진다. gt 는 None (레코드에서 바로 만든다)."""
    if cond == "gt": return None
    names = {i: n for i, n in gemini_names().items()} if cond == "gemini" else {i: n for i, (n, _) in pipe_names(split, osr=cond == "pipe_osr").items()}
    ing = ing_records(); out = n5k_pred()
    for i, n in names.items():
        j = ing.get((i, n))
        if j: out[i] = (render(n, j["ingredients"]), j["category"])
    return out


def all_pred_texts():
    """임베딩해야 할 Gemini 텍스트 전체 (두 분할, gemini + pipe)."""
    t = {v[0] for v in n5k_pred().values()}
    for (i, n), j in ing_records().items(): t.add(render(n, j["ingredients"]))
    return t


def load_emb():
    f = EXP / "emb_cache_ing.npz"
    if not f.exists(): return {}
    d = np.load(f, allow_pickle=False); return dict(zip(d["texts"].tolist(), d["emb"]))
