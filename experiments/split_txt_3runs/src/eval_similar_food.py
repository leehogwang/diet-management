"""처음 보는 음식(분류기 open-set 판정에서 Gemini 로 넘긴 사진)의 1인분 값을 '분류기가 가장 비슷하다고 본 아는 음식' 의 1인분 값으로 채우는 실험 (122_2_forunseen).
아는 음식 1인분 값 = 그 분할 train 정답 / Q 비율 의 음식별 중앙값 (122 정답은 음식마다 일정). 분류기 = runs/cls_osr_122_2_forunseen_baseline/test_pred.json (top5 + 확률).
양(Q) = 양추정 모델 argmax 그대로. 분류기 통과 사진은 영양 모델 1인분 (r_ing 과 같음). 조건:
  r_ing          : Gemini 1인분 (음식명 + 재료)                            ← 현재 최고
  sim_top1       : 분류기 top1 아는 음식의 1인분 값
  sim_top5w      : top5 아는 음식 1인분 값의 확률 가중 평균
  avg_top1_gem   : (sim_top1 + r_ing) / 2
  avg_top5w_gem  : (sim_top5w + r_ing) / 2
  dens_top1_massgem : top1 의 g당 영양값 × Gemini 1인분 중량
  dens_gem_masstop1 : Gemini g당 영양값 × top1 1인분 중량
  oracle_best_known : 아는 음식 중 정답 1인분과 가장 가까운 음식 (참고하는 방식의 상한)
출력 EXP/similar_food_eval.json.  usage: python eval_similar_food.py"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts_ing  # noqa: E402
from compare_pipeline import EXP, KEYS, RATIO, metrics  # noqa: E402

SP = "122_2_forunseen"


def main():
    recs = json.loads((EXP / "records.json").read_text()); S = json.loads((EXP / "splits.json").read_text())[SP]
    tr = {}
    for i in S["train"]: tr.setdefault(recs[i]["food"], []).append(np.array(recs[i]["y"]) / RATIO[recs[i]["q"]])
    KS = {f: np.median(v, 0) for f, v in tr.items()}
    C = json.loads((EXP / "runs" / f"cls_osr_{SP}_baseline/test_pred.json").read_text())
    GI = {j["food"]: np.array([j[k] for k in KEYS]) for j in map(json.loads, open(EXP / "gemini_serving_ing.jsonl"))}
    ING = texts_ing.ing_records()
    Q = np.load(EXP / "runs" / f"{SP}_quantity/test_logits_pipe.npz"); ids = Q["id"]; qp = Q["logits_pipe_osr"].argmax(1); qt = Q["label"]
    d = np.load(EXP / "runs" / f"{SP}_q4_ing_cat/preds_text_pipe_osr.npz"); pos = {int(i): k for k, i in enumerate(d["id"])}
    p_model = d["pred"][[pos[int(i)] for i in ids]]
    pn = texts_ing.pipe_names(SP, osr=True); known = np.array([pn[int(i)][1] == "classifier" for i in ids])
    y = np.array([recs[int(i)]["y"] for i in ids], np.float64); y4 = y / RATIO[qt][:, None]
    gem = np.array([GI[texts_ing.render(pn[int(i)][0], ING[(int(i), pn[int(i)][0])]["ingredients"])] if not k else p_model[j]
                    for j, (i, k) in enumerate(zip(ids, known))])
    top1 = np.array([KS[C[str(int(i))]["top"][0]] for i in ids])
    def w5(i):
        c = C[str(int(i))]; p = np.array(c["prob"]); p = p / p.sum(); return (p[:, None] * np.array([KS[f] for f in c["top"]])).sum(0)
    top5 = np.array([w5(i) for i in ids])
    kn = np.array(list(KS.values())); kname = list(KS)
    best = np.array([kn[np.abs((kn[:, :4] - t[:4]) / np.maximum(t[:4], 1)).mean(1).argmin()] for t in y4])
    dens = lambda v: v / np.maximum(v[:, 4:5], 1e-6)
    cand = {"r_ing": gem, "sim_top1": top1, "sim_top5w": top5, "avg_top1_gem": (top1 + gem) / 2, "avg_top5w_gem": (top5 + gem) / 2,
            "dens_top1_massgem": dens(top1) * gem[:, 4:5], "dens_gem_masstop1": dens(gem) * top1[:, 4:5], "oracle_best_known": best}
    for k in ("dens_top1_massgem", "dens_gem_masstop1"): cand[k][:, 4] = cand[k][:, 4] if k.endswith("massgem") else top1[:, 4]
    out = {"n": int(len(ids)), "gemini_route": int((~known).sum()), "conds": {}}
    print(f"===== {SP} n {len(ids)} Gemini 경로 {(~known).sum()}  (양 = 양추정 argmax, 정답Q = 상한)")
    for k, v in cand.items():
        P = np.where(known[:, None], p_model, v)
        m, mo, ms = metrics(P * RATIO[qp][:, None], y), metrics(P * RATIO[qt][:, None], y), metrics(P, y4)
        out["conds"][k] = {"pipe": m, "oracle_q": mo, "serving": ms}
        print(f"  {k:20s} G4 {m['G4']:6.2f} A4 {m['A4']:6.2f} | 정답Q {mo['G4']:6.2f} | 1인분 G4 {ms['G4']:6.2f} | G {m['G']}")
    foods = np.array([recs[int(i)]["food"] for i in ids]); pf = {}
    for f in sorted(set(foods)):
        m = foods == f; t1 = Counter(C[str(int(i))]["top"][0] for i in ids[m]).most_common(2)
        e = lambda v: round(float(np.abs(v[m, :4] - y4[m, :4]).sum(0).sum() / y4[m, :4].sum() * 100), 1)
        pf[f] = {"top1_known": t1, "true_cal": round(float(y4[m, 0].mean()), 1), "top1_cal": round(float(top1[m, 0].mean()), 1),
                 "gem_cal": round(float(gem[m, 0].mean()), 1), "err_top1": e(top1), "err_gem": e(gem), "err_top5w": e(top5)}
    out["per_food"] = pf
    print(f"\n{'음식':10s} {'분류기 top1 (최빈)':28s} 정답kcal top1kcal Geminikcal | 1인분오차% top1 top5w Gemini")
    for f, v in sorted(pf.items(), key=lambda x: x[1]["err_top1"] - x[1]["err_gem"]):
        t = ", ".join(f"{a}({b})" for a, b in v["top1_known"])
        print(f"{f:10s} {t:28s} {v['true_cal']:7.0f} {v['top1_cal']:7.0f} {v['gem_cal']:7.0f} | {v['err_top1']:6.1f} {v['err_top5w']:6.1f} {v['err_gem']:6.1f}")
    (EXP / "similar_food_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
