"""Gemini 에게 음식명만 주고 1인분(1회 제공량) 영양값을 직접 추정시킨다 (영양 모델 대신 외부 지식으로 1인분 값 공급하는 실험).
대상 음식명: 122 두 분할 test 의 pipe_osr 음식명 + 정답 음식명 (중복 제거, 이름 단위 호출 — 사진 없음).
모델 gemini-3.8-flash, temperature 0, thinking 0 (파이프라인 다른 단계와 같음).
출력 EXP/gemini_serving.jsonl ({"food", "calories", "fat", "protein", "carb", "mass"}). 다시 실행하면 남은 이름만.
--prompt mfds: 1인분 기준을 식약처 식품영양성분 DB 로 명시한 프롬프트 (출력 EXP/gemini_serving_mfds.jsonl).
--prompt ing : 음식명 + 파이프라인 2단계 재료(gemini_ing.jsonl)를 함께 준다. 대상은 pipe_osr 에서 Gemini 로 넘긴(처음 보는 음식) 사진의
               (음식명, 재료) 조합. 키 "food" = texts_ing.render(음식명, 재료) 텍스트 (출력 EXP/gemini_serving_ing.jsonl).
--train_foods: (default 프롬프트) 122_2_forunseen train·val 정답 음식명도 대상에 넣는다 (양추정 모델에 1인분 중량을 입력하는 실험용).
usage: python gemini_serving.py [--workers 16] [--prompt default|mfds|ing] [--train_foods]"""
import argparse, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ENV = Path("/home/leehogwang/Food-Nutrition-Estimation/.env")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts_ing  # noqa: E402

KEYS = ["calories", "fat", "protein", "carb", "mass"]
PROMPT = ("음식 '{food}' 의 한국 음식 영양성분 데이터베이스 기준 1인분(1회 제공량) 영양값을 추정하세요.\n"
          "- calories: 열량 (kcal)\n- fat: 지방 (g)\n- protein: 단백질 (g)\n- carb: 탄수화물 (g)\n- mass: 1인분 중량 (g)\n"
          "숫자만 답하세요.")
PROMPTS = {"default": PROMPT,
           "mfds": ("음식 '{food}' 의 1인분 영양값을 식품의약품안전처 식품영양성분 데이터베이스(국가표준식품성분표)에 등록된 "
                    "음식 1인분(1회 제공량) 기준으로 답하세요. 일반 가정·식당의 양이 아니라 DB 의 1인분 중량과 그 중량의 영양값입니다.\n"
                    "- calories: 열량 (kcal)\n- fat: 지방 (g)\n- protein: 단백질 (g)\n- carb: 탄수화물 (g)\n- mass: DB 1인분 중량 (g)\n"
                    "숫자만 답하세요.")}
PROMPTS["ing"] = ("음식 '{food}' 의 재료는 {ingredients} 입니다. 이 음식의 한국 음식 영양성분 데이터베이스 기준 1인분(1회 제공량) 영양값을 추정하세요.\n"
                  "- calories: 열량 (kcal)\n- fat: 지방 (g)\n- protein: 단백질 (g)\n- carb: 탄수화물 (g)\n- mass: 1인분 중량 (g)\n"
                  "숫자만 답하세요.")
OUT = {"default": "gemini_serving.jsonl", "mfds": "gemini_serving_mfds.jsonl", "ing": "gemini_serving_ing.jsonl"}
SCHEMA = {"type": "OBJECT", "properties": {k: {"type": "NUMBER"} for k in KEYS}, "required": KEYS}


def target_names():
    recs = json.loads((EXP / "records.json").read_text()); S = json.loads((EXP / "splits.json").read_text()); names = set()
    for sp in ("122_1_forseen", "122_2_forunseen"):
        names |= {n for n, _ in texts_ing.pipe_names(sp, osr=True).values()}
        names |= {recs[i]["food"] for i in S[sp]["test"]}
    return names


def ing_targets():
    """{render 텍스트: (음식명, 재료 목록)} — pipe_osr 에서 Gemini 로 넘긴 사진만."""
    ing = texts_ing.ing_records(); out = {}
    for sp in ("122_1_forseen", "122_2_forunseen"):
        for i, (n, src) in texts_ing.pipe_names(sp, osr=True).items():
            if src != "gemini": continue
            j = ing[(i, n)]; out[texts_ing.render(n, j["ingredients"])] = (n, j["ingredients"])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--prompt", choices=list(PROMPTS), default="default")
    ap.add_argument("--train_foods", action="store_true"); a = ap.parse_args()
    f = EXP / OUT[a.prompt]; done = {json.loads(l)["food"] for l in open(f)} if f.exists() else set()
    tg = ing_targets() if a.prompt == "ing" else None
    names = set(tg) if tg else target_names()
    if a.train_foods and not tg:
        recs = json.loads((EXP / "records.json").read_text()); S = json.loads((EXP / "splits.json").read_text())["122_2_forunseen"]
        names |= {recs[i]["food"] for i in S["train"] + S["val"]}
    todo = sorted(names - done)
    print("대상 이름", len(names), "완료", len(names & done), "새로 호출", len(todo), flush=True)
    if not todo: return
    from google import genai
    from google.genai import types
    key = next(l.split("=", 1)[1].strip().strip("\"' ") for l in open(ENV) if l.strip().startswith("GEMINI_KEY"))
    cl = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0, response_schema=SCHEMA,
                                      thinking_config=types.ThinkingConfig(thinking_budget=0))
    lock = threading.Lock()

    def ask(name):
        err = None
        for t in range(6):
            try:
                resp = cl.models.generate_content(model="gemini-3.8-flash", contents=[PROMPTS[a.prompt].format(food=tg[name][0], ingredients=", ".join(tg[name][1])) if tg else PROMPTS[a.prompt].format(food=name)], config=cfg)
                j = json.loads(resp.text); out = {"food": name, **{k: float(j[k]) for k in KEYS}}
                with lock: open(f, "a").write(json.dumps(out, ensure_ascii=False) + "\n")
                return
            except Exception as e:
                err = type(e).__name__; time.sleep(3 * (t + 1))
        print("실패", name, err, flush=True)  # 예외 문자열에 키가 섞일 수 있어 타입만

    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(ask, todo))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
