"""양추정 모델 확신도(temperature 보정 softmax 최대값)가 --conf 미만인 test 사진만 Gemini 가 사진을 보고 양 등급(Q1~Q5)을 추정한다.
음식명 = pipe_osr 음식명 (파이프라인 1단계와 같음). 1인분 기준 중량 = 파이프라인 1인분 값의 중량
  (Gemini 로 넘긴 사진: gemini_serving_ing.jsonl (음식명 + 재료), 분류기 통과 사진: gemini_serving.jsonl (음식명)).
Gemini 답: q_level (Q1~Q5 enum) + grams (사진 속 음식 중량 추정). 모델 gemini-3.8-flash, temperature 0, thinking 0, JSON.
출력 EXP/gemini_quantity.jsonl ({"id", "split", "food", "serving_g", "q_level", "grams"}). 다시 실행하면 남은 사진만.
--sample K: 확신도 미달 사진 중 (정답 음식, 정답 Q) 조합마다 K 장만 고정 시드로 뽑아 호출 (이미 받은 사진을 먼저 채움). 0 이면 전부.
usage: python gemini_quantity.py [--conf 0.7] [--workers 16] [--sample 5]"""
import argparse, io, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts_ing  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
ENV = Path("/home/leehogwang/Food-Nutrition-Estimation/.env")
SPLITS = ("122_2_forunseen", "122_1_forseen")
PROMPT = ("사진 속 음식은 '{food}' 입니다. 이 음식의 1인분(1회 제공량)은 약 {serv:.0f} g 입니다.\n"
          "사진에 담긴 양이 1인분의 몇 배인지 보고 양 등급을 고르세요.\n"
          "- Q1: 1/4 인분 (약 0.25배)\n- Q2: 1/2 인분 (약 0.5배)\n- Q3: 3/4 인분 (약 0.75배)\n- Q4: 1 인분 (약 1.0배)\n- Q5: 1과 1/4 인분 (약 1.25배)\n"
          "그릇 크기, 숟가락·포크·동전 같은 주변 물체, 그릇이 채워진 정도를 참고하세요.\n"
          "q_level 에 등급, grams 에 사진 속 음식 중량(g) 추정값을 JSON 으로만 답하세요.")
SCHEMA = {"type": "OBJECT", "properties": {"q_level": {"type": "STRING", "enum": ["Q1", "Q2", "Q3", "Q4", "Q5"]}, "grams": {"type": "NUMBER"}},
          "required": ["q_level", "grams"]}


def softmax_conf(sp):
    T = torch.load(EXP / "runs" / f"{sp}_quantity/best.pth", map_location="cpu", weights_only=False)["temperature"]
    Q = np.load(EXP / "runs" / f"{sp}_quantity/test_logits_pipe.npz"); L = Q["logits_pipe_osr"] / T
    P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    return Q["id"], P.max(1)


def serving_mass():
    """(split, id) → (음식명, 1인분 중량 g): 파이프라인 1인분 값과 같은 출처."""
    G = {j["food"]: j["mass"] for j in map(json.loads, open(EXP / "gemini_serving.jsonl"))}
    GI = {j["food"]: j["mass"] for j in map(json.loads, open(EXP / "gemini_serving_ing.jsonl"))}
    ING = texts_ing.ing_records(); out = {}
    for sp in SPLITS:
        for i, (n, src) in texts_ing.pipe_names(sp, osr=True).items():
            out[(sp, i)] = (n, GI[texts_ing.render(n, ING[(i, n)]["ingredients"])] if src == "gemini" else G[n])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--conf", type=float, default=0.7); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    recs = json.loads((EXP / "records.json").read_text()); SM = serving_mass()
    f = EXP / "gemini_quantity.jsonl"; done = {(j["split"], j["id"]) for j in map(json.loads, open(f))} if f.exists() else set()
    todo = []
    for sp in SPLITS:
        ids, conf = softmax_conf(sp); low = [(sp, int(i)) for i, c in zip(ids, conf) if c < a.conf]
        if a.sample:
            rng = np.random.default_rng(0); grp = {}
            for k in low: grp.setdefault((recs[k[1]]["food"], recs[k[1]]["q"]), []).append(k)
            low = []
            for g in sorted(grp):
                ks = grp[g]; old = [k for k in ks if k in done]; new = [k for k in ks if k not in done]
                low += (old + [new[j] for j in rng.permutation(len(new))])[:a.sample]
        print(sp, "test", len(ids), f"확신도 < {a.conf} 대상", len(low), flush=True); todo += [k for k in low if k not in done]
    print("새로 호출", len(todo), "이미 완료", len(done), flush=True)
    if not todo: return
    from google import genai
    from google.genai import types
    key = next(l.split("=", 1)[1].strip().strip("\"' ") for l in open(ENV) if l.strip().startswith("GEMINI_KEY"))
    cl = genai.Client(api_key=key)
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0, response_schema=SCHEMA,
                                      thinking_config=types.ThinkingConfig(thinking_budget=0))
    lock = threading.Lock(); tok = [0, 0]; cnt = [0]

    def ask(k):
        sp, i = k; food, serv = SM[k]; err = None
        for t in range(6):
            try:
                im = Image.open(recs[i]["img"]).convert("RGB"); W, H = im.size; s = 768 / max(W, H)
                if s < 1: im = im.resize((round(W * s), round(H * s)), Image.BILINEAR)
                b = io.BytesIO(); im.save(b, format="JPEG", quality=90)
                part = types.Part.from_bytes(data=b.getvalue(), mime_type="image/jpeg")
                resp = cl.models.generate_content(model="gemini-3.8-flash", contents=[part, PROMPT.format(food=food, serv=serv)], config=cfg)
                j = json.loads(resp.text); u = resp.usage_metadata
                out = {"id": i, "split": sp, "food": food, "serving_g": serv, "q_level": j["q_level"], "grams": j["grams"]}
                with lock:
                    open(f, "a").write(json.dumps(out, ensure_ascii=False) + "\n"); tok[0] += u.prompt_token_count or 0; tok[1] += u.candidates_token_count or 0
                    cnt[0] += 1
                    if cnt[0] % 250 == 0: print("진행", cnt[0], "토큰", tok, flush=True)
                return
            except Exception as e:
                err = type(e).__name__; time.sleep(3 * (t + 1))
        print("실패", sp, i, err, flush=True)  # 예외 문자열에 키가 섞일 수 있어 타입만

    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(ask, todo))
    print("DONE 토큰", tok, flush=True)


if __name__ == "__main__":
    main()
