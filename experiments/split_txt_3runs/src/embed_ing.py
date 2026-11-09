"""texts_ing 형식 텍스트 임베딩 (gemini-embedding-2, 3072차원, SEMANTIC_SIMILARITY — embed.py 와 같은 설정).
기존 emb_cache.json 에 이미 있는 텍스트(N5k 재료 목록)는 거기서 가져오고, 없는 것만 호출. 출력 EXP/emb_cache_ing.npz.
usage: python embed_ing.py [--gt_only]"""
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from texts_ing import EXP, all_pred_texts, gt_text, load_emb  # noqa: E402

ENV = Path("/home/leehogwang/Food-Nutrition-Estimation/.env")


def save(cache):
    t = sorted(cache); np.savez(EXP / "emb_cache_ing.npz", texts=np.array(t), emb=np.stack([np.asarray(cache[k], np.float32) for k in t]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gt_only", action="store_true"); a = ap.parse_args()
    recs = json.loads((EXP / "records.json").read_text())
    texts = {gt_text(r) for r in recs} | (set() if a.gt_only else all_pred_texts())
    cache = load_emb(); missing = {t for t in texts if t not in cache}
    if missing:
        old = json.loads((EXP / "emb_cache.json").read_text())
        for t in missing & old.keys(): cache[t] = np.asarray(old[t], np.float32)
        del old
    todo = sorted(t for t in texts if t and t not in cache); print("텍스트", len(texts), "새로", len(todo), flush=True)
    if todo:
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
                if n % 1000 == 999: save(cache); print("진행", n + 1, flush=True)
    save(cache); print("DONE", len(cache), "남은 미임베딩", len([t for t in texts if t not in cache]), flush=True)


if __name__ == "__main__":
    main()
