"""텍스트 임베딩 (gemini-embedding-2, 3072차원, task_type SEMANTIC_SIMILARITY — n296 config·실험 40 과 같음).
텍스트: 122 = 음식명(정답 / Gemini 추정), N5k = 재료 목록 "a, b, c"(정답은 접시마다 고정 시드로 순서 섞음 / Gemini 추정).
실험 40 emb_cache.json 을 먼저 복사해 두고 없는 텍스트만 호출. 출력 EXP/emb_cache.json.  usage: python embed.py"""
import json, random, shutil, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
E40 = Path("/home/leehogwang/Food-Nutrition-Estimation/experiments/40_unified")
ENV = Path("/home/leehogwang/Food-Nutrition-Estimation/.env")


def gt_text(r):
    if r["ds"] == "122": return r["food"]
    ing = list(dict.fromkeys(r["ingr"])); random.Random(r["name"]).shuffle(ing); return ", ".join(ing)


def pred_texts():
    f = EXP / "gemini_pred.jsonl"; out = {}
    if f.exists():
        for l in open(f):
            j = json.loads(l); out[j["id"]] = j["food"] if "food" in j else ", ".join(dict.fromkeys(j["ingr"]))
    return out


def main():
    f = EXP / "emb_cache.json"
    if not f.exists(): shutil.copy(E40 / "emb_cache.json", f)
    cache = json.loads(f.read_text()); recs = json.loads((EXP / "records.json").read_text())
    texts = {gt_text(r) for r in recs} | set(pred_texts().values())
    todo = sorted(t for t in texts if t and t not in cache); print("텍스트", len(texts), "새로", len(todo), flush=True)
    if not todo: return
    from google import genai
    from google.genai import types
    key = next(l.split("=", 1)[1].strip().strip("\"' ") for l in open(ENV) if l.strip().startswith("GEMINI_KEY"))
    cl = genai.Client(api_key=key)

    def one(t):
        err = None
        for k in range(6):
            try:
                r = cl.models.embed_content(model="models/gemini-embedding-2", contents=t, config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"))
                return t, r.embeddings[0].values
            except Exception as e:
                err = type(e).__name__; time.sleep(5 * (k + 1))
        print("실패", err, flush=True); return t, None
    with ThreadPoolExecutor(16) as ex:
        for n, (t, v) in enumerate(ex.map(one, todo)):
            if v: cache[t] = v
            if n % 500 == 499: f.write_text(json.dumps(cache)); print("진행", n + 1, flush=True)
    f.write_text(json.dumps(cache)); miss = [t for t in texts if t not in cache]
    print("DONE", len(cache), "남은 미임베딩", len(miss), flush=True)


if __name__ == "__main__":
    main()
