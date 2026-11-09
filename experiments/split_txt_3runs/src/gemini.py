"""test 사진의 Gemini 추정 텍스트(평가 조건 text_gemini). 실험 40 과 같은 모델·프롬프트(gemini-3.8-flash, temperature 0, thinking 0, JSON).
① 122 test 사진 → 음식명  ② N5k test 접시 → 재료 목록. 실험 40 gemini_pred.jsonl 에 같은 사진이 있으면 재사용, 없는 것만 호출.
출력 EXP/gemini_pred.jsonl ({"id", "food"|"ingr", "src": e40|new}). 다시 실행하면 남은 것만. 키는 Food-Nutrition-Estimation/.env 의 GEMINI_KEY.
usage: python gemini.py [--workers 16]"""
import argparse, io, json, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

EXP = Path(__file__).resolve().parents[1]
E40 = Path("/home/leehogwang/Food-Nutrition-Estimation/experiments/40_unified")
ENV = Path("/home/leehogwang/Food-Nutrition-Estimation/.env")
P122 = "이 사진 속 한국 음식의 이름을 하나만 답하세요. 일반적으로 쓰는 음식 이름(예: 김치찌개, 돈가스)으로 답하고 JSON 으로만 답하세요."
PN5K = "List the food ingredients visible on this plate, using short common English names (e.g. white rice, chicken, broccoli). Answer only in JSON."


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); a = ap.parse_args()
    recs = json.loads((EXP / "records.json").read_text()); S = json.loads((EXP / "splits.json").read_text())
    ids = sorted({i for t in S for i in S[t]["test"]})
    f = EXP / "gemini_pred.jsonl"; done = {json.loads(l)["id"] for l in open(f)} if f.exists() else set()
    old = {}
    for l in open(E40 / "gemini_pred.jsonl"):
        j = json.loads(l); old[j["id"]] = j
    with open(f, "a") as w:
        for i in ids:
            e = recs[i].get("e40_id")
            if i not in done and e is not None and e in old:
                j = {k: v for k, v in old[e].items() if k != "id"}; w.write(json.dumps({"id": i, **j, "src": "e40"}, ensure_ascii=False) + "\n"); done.add(i)
    todo = [i for i in ids if i not in done]; print("대상", len(ids), "재사용 포함 완료", len(done), "새로 호출", len(todo), flush=True)
    if not todo: return
    from google import genai
    from google.genai import types
    key = next(l.split("=", 1)[1].strip().strip("\"' ") for l in open(ENV) if l.strip().startswith("GEMINI_KEY"))
    cl = genai.Client(api_key=key)
    cfg = lambda schema: types.GenerateContentConfig(response_mime_type="application/json", temperature=0, response_schema=schema,
                                                     thinking_config=types.ThinkingConfig(thinking_budget=0))
    c122 = cfg({"type": "OBJECT", "properties": {"food_name": {"type": "STRING"}}, "required": ["food_name"]})
    cn5k = cfg({"type": "OBJECT", "properties": {"ingredients": {"type": "ARRAY", "items": {"type": "STRING"}}}, "required": ["ingredients"]})
    lock = threading.Lock(); tok = [0, 0]; cnt = [0]

    def ask(i):
        r = recs[i]; err = None
        for t in range(6):
            try:
                im = Image.open(r["img"]).convert("RGB"); W, H = im.size; s = 768 / max(W, H)
                if s < 1: im = im.resize((round(W * s), round(H * s)), Image.BILINEAR)
                b = io.BytesIO(); im.save(b, format="JPEG", quality=90)
                part = types.Part.from_bytes(data=b.getvalue(), mime_type="image/jpeg")
                n5k = r["ds"] == "n5k"
                resp = cl.models.generate_content(model="gemini-3.8-flash", contents=[part, PN5K if n5k else P122], config=cn5k if n5k else c122)
                j = json.loads(resp.text); u = resp.usage_metadata
                out = {"id": i, "ingr": j["ingredients"]} if n5k else {"id": i, "food": j["food_name"]}
                out["src"] = "new"
                with lock:
                    open(f, "a").write(json.dumps(out, ensure_ascii=False) + "\n"); tok[0] += u.prompt_token_count or 0; tok[1] += u.candidates_token_count or 0
                    cnt[0] += 1
                    if cnt[0] % 250 == 0: print("진행", cnt[0], "토큰", tok, flush=True)
                return
            except Exception as e:
                err = type(e).__name__; time.sleep(3 * (t + 1))
        print("실패", i, err, flush=True)  # 예외 문자열에 키가 섞일 수 있어 타입만

    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(ask, todo))
    print("DONE 토큰", tok, flush=True)


if __name__ == "__main__":
    main()
