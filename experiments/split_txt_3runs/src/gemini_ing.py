"""파이프라인 2단계와 같은 Gemini 재료·대분류 예측 (122 test 사진, 두 분할).
프롬프트·스키마는 n296.pipeline_infer 의 ING_PROMPT / ING_SCHEMA 그대로. 음식명은 gemini_pred.jsonl 의 Gemini 음식명(사진만 보고 추정).
모델은 음식명 단계와 같은 gemini-3.8-flash (temperature 0, thinking 0). 사진은 gemini.py 와 같이 긴 변 768px JPEG.
출력 EXP/gemini_ing.jsonl ({"id", "food", "ingredients", "category"}). 다시 실행하면 남은 것만.
--names pipe: 분할별 분류기 → (확신도 < 0.7 이면 Gemini) 음식명으로 아직 없는 (id, 음식명) 쌍만 호출 (texts_ing.pipe_names).
usage: python gemini_ing.py [--workers 16] [--limit N] [--names gemini|pipe]"""
import argparse, io, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

EXP = Path(__file__).resolve().parents[1]
REPO = Path("/home/leehogwang/food_b")
ENV = Path("/home/leehogwang/Food-Nutrition-Estimation/.env")
sys.path.insert(0, str(REPO))
from n296.pipeline_infer import ING_PROMPT, ING_SCHEMA  # noqa: E402
from n296.predict_food_gemini import clean_list  # noqa: E402
from n296.food_category import normalize_category  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--names", choices=["gemini", "pipe", "pipe_osr"], default="gemini")
    a = ap.parse_args()
    recs = json.loads((EXP / "records.json").read_text()); S = json.loads((EXP / "splits.json").read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import texts_ing
    if a.names == "gemini":
        gem = texts_ing.gemini_names(); pairs = {(i, gem[i]) for sp in ("122_1_forseen", "122_2_forunseen") for i in S[sp]["test"]}
    else:
        pairs = set()
        for sp in ("122_1_forseen", "122_2_forunseen"):
            pn = texts_ing.pipe_names(sp, osr=a.names == "pipe_osr"); assert pn, f"분류기 test_pred 없음: {sp}"; pairs |= {(i, n) for i, (n, _) in pn.items()}
    f = EXP / "gemini_ing.jsonl"; done = {(j["id"], j["food"]) for j in map(json.loads, open(f))} if f.exists() else set()
    todo = sorted(pairs - done)
    if a.limit: todo = todo[:a.limit]
    print("대상 쌍", len(pairs), "완료", len(pairs & done), "새로 호출", len(todo), flush=True)
    if not todo: return
    from google import genai
    from google.genai import types
    key = next(l.split("=", 1)[1].strip().strip("\"' ") for l in open(ENV) if l.strip().startswith("GEMINI_KEY"))
    cl = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0, response_schema=ING_SCHEMA,
                                      thinking_config=types.ThinkingConfig(thinking_budget=0))
    lock = threading.Lock(); tok = [0, 0]; cnt = [0]

    def ask(pair):
        i, name = pair; r = recs[i]; err = None
        for t in range(6):
            try:
                im = Image.open(r["img"]).convert("RGB"); W, H = im.size; s = 768 / max(W, H)
                if s < 1: im = im.resize((round(W * s), round(H * s)), Image.BILINEAR)
                b = io.BytesIO(); im.save(b, format="JPEG", quality=90)
                part = types.Part.from_bytes(data=b.getvalue(), mime_type="image/jpeg")
                resp = cl.models.generate_content(model="gemini-3.8-flash", contents=[part, ING_PROMPT.format(food_name=name)], config=cfg)
                j = json.loads(resp.text); u = resp.usage_metadata
                out = {"id": i, "food": name, "ingredients": clean_list(j.get("ingredients")), "category": normalize_category(j.get("category"))}
                with lock:
                    open(f, "a").write(json.dumps(out, ensure_ascii=False) + "\n"); tok[0] += u.prompt_token_count or 0; tok[1] += u.candidates_token_count or 0
                    cnt[0] += 1
                    if cnt[0] % 500 == 0: print("진행", cnt[0], "토큰", tok, flush=True)
                return
            except Exception as e:
                err = type(e).__name__; time.sleep(3 * (t + 1))
        print("실패", i, err, flush=True)  # 예외 문자열에 키가 섞일 수 있어 타입만

    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(ask, todo))
    print("DONE 토큰", tok, flush=True)


if __name__ == "__main__":
    main()
