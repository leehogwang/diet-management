"""양추정 확신도 < CONF 인 사진은 Gemini 양 등급(gemini_quantity.jsonl), 나머지는 양추정 모델 argmax 로 Q 를 정해 영양 G4 를 계산.
1인분 값은 eval_gemini_serving 의 r_ing 과 같음 (분류기 통과 → 영양 모델 1인분, Gemini 로 넘긴 사진 → Gemini 1인분(음식명 + 재료)).
조건:
  model       : 양추정 모델 argmax (= r_ing, 현재 최고)
  hybrid      : 확신도 < CONF → Gemini q_level, 아니면 모델              ← 요청한 규칙
  hybrid_g    : 확신도 < CONF → Gemini grams / 1인분 중량 에 가장 가까운 등급
  hybrid_cont : 확신도 < CONF → Gemini grams / 1인분 중량 을 0.25~1.25 로 자른 연속 비율
  oracle_q    : 정답 Q (상한)
출력 EXP/gemini_quantity_eval.json.  --sample K: gemini_quantity.py --sample K 로 뽑은 사진만 평가 ((정답 음식, 정답 Q) 조합마다 Gemini 답이 있는 사진 앞에서 K 장). 출력 gemini_quantity_eval_sampleK.json
usage: python eval_gemini_quantity.py [--conf 0.7] [--sample 5]"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts_ing  # noqa: E402
from compare_pipeline import EXP, KEYS, RATIO, metrics  # noqa: E402
from gemini_quantity import softmax_conf  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--conf", type=float, default=0.7); ap.add_argument("--sample", type=int, default=0); a = ap.parse_args()
    recs = json.loads((EXP / "records.json").read_text())
    GI = {j["food"]: [j[k] for k in KEYS] for j in map(json.loads, open(EXP / "gemini_serving_ing.jsonl"))}
    GQ = {(j["split"], j["id"]): j for j in map(json.loads, open(EXP / "gemini_quantity.jsonl"))}
    ING = texts_ing.ing_records(); out = {}
    for sp in ("122_2_forunseen", "122_1_forseen"):
        Q = np.load(EXP / "runs" / f"{sp}_quantity/test_logits_pipe.npz"); ids = Q["id"]; qt = Q["label"]; qm = Q["logits_pipe_osr"].argmax(1)
        cid, conf = softmax_conf(sp); assert (cid == ids).all()
        d = np.load(EXP / "runs" / f"{sp}_q4_ing_cat/preds_text_pipe_osr.npz"); pos = {int(i): k for k, i in enumerate(d["id"])}
        p_model = d["pred"][[pos[int(i)] for i in ids]]
        pn = texts_ing.pipe_names(sp, osr=True); known = np.array([pn[int(i)][1] == "classifier" for i in ids])
        p_gi = np.array([GI[texts_ing.render(pn[int(i)][0], ING[(int(i), pn[int(i)][0])]["ingredients"])] if not k else [0.0] * 5
                         for i, k in zip(ids, known)])
        serv = np.where(known[:, None], p_model, p_gi)
        y = np.array([recs[int(i)]["y"] for i in ids], np.float64)
        if a.sample:  # 확신도 미달 + Gemini 답이 있는 사진을 (음식, Q) 조합마다 K 장
            cnt = {}; keep = np.zeros(len(ids), bool)
            for j, i in enumerate(ids):
                g_ = (recs[int(i)]["food"], recs[int(i)]["q"])
                if conf[j] < a.conf and (sp, int(i)) in GQ and cnt.get(g_, 0) < a.sample: keep[j] = True; cnt[g_] = cnt.get(g_, 0) + 1
            ids, qt, qm, conf, p_model, known, serv, y = (x[keep] for x in (ids, qt, qm, conf, p_model, known, serv, y))
        low = conf < a.conf; g = [GQ.get((sp, int(i))) for i in ids]; has = np.array([x is not None for x in g]); use = low & has
        qg = np.array([int(x["q_level"][1]) - 1 if x else -1 for x in g])
        rg = np.array([x["grams"] / max(x["serving_g"], 1e-6) if x else np.nan for x in g])
        qgg = np.where(has, np.abs(np.log(np.clip(np.nan_to_num(rg, nan=1), 1e-3, None))[:, None] - np.log(RATIO)[None]).argmin(1), -1)
        Qs = {"model": qm, "hybrid": np.where(use, qg, qm), "hybrid_g": np.where(use, qgg, qm), "oracle_q": qt}
        R = {k: RATIO[v] for k, v in Qs.items()}; R["hybrid_cont"] = np.where(use, np.clip(np.nan_to_num(rg, nan=1), .25, 1.25), RATIO[qm])
        res = {"n": int(len(ids)), "conf_lt": float(low.mean()), "gemini_answered": int(use.sum()), "missing_gemini": int((low & ~has).sum()), "conds": {}}
        for k, r in R.items():
            m = metrics(serv * r[:, None], y)
            if k in Qs:
                q = Qs[k]; m["q_acc"] = round(float((q == qt).mean()), 4); m["q_within1"] = round(float((abs(q - qt) <= 1).mean()), 4)
                m["q_acc_low"] = round(float((q[use] == qt[use]).mean()), 4) if use.any() else None
                m["confusion"] = np.bincount(qt * 5 + q, minlength=25).reshape(5, 5).tolist()
            res["conds"][k] = m
        foods = np.array([recs[int(i)]["food"] for i in ids])
        res["per_food_q_acc"] = {f: {"n": int((foods == f).sum()), "model": round(float((qm == qt)[foods == f].mean()), 3),
                                     "gemini": round(float((qg == qt)[(foods == f) & has].mean()), 3) if ((foods == f) & has).any() else None,
                                     "gemini_bias": round(float((qg - qt)[(foods == f) & has].mean()), 2) if ((foods == f) & has).any() else None}
                                 for f in sorted(set(foods))}
        out[sp] = res
        print(f"===== {sp} n {len(ids)} 확신도<{a.conf} {low.mean():.1%}, Gemini 답 {use.sum()}, 누락 {(low & ~has).sum()}")
        for k, m in res["conds"].items():
            print(f"  {k:12s} G4 {m['G4']:6.2f} A4 {m['A4']:6.2f} | Q acc {m.get('q_acc', '-')} ±1 {m.get('q_within1', '-')} | G {m['G']}")
    (EXP / (f"gemini_quantity_eval_sample{a.sample}.json" if a.sample else "gemini_quantity_eval.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
