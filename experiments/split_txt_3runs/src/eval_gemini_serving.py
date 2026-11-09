"""Gemini 1인분 영양값(gemini_serving.jsonl) × 양추정 비율 로 122 test 영양 G4 를 계산해 영양 모델(1인분 _q4_ing_cat)과 비교. 재학습 없음.
조건 (모두 1인분 값 × 양추정 argmax 비율 = pipe_always, 양추정 입력 대분류는 pipe_osr):
  model        : 영양 모델 1인분 예측 (pipe_osr 텍스트)                       ← 현재 최선
  gemini       : Gemini 1인분 값 (pipe_osr 음식명)
  routed       : 분류기 통과(아는 음식) → model, Gemini 로 넘긴 사진(처음 보는 음식) → gemini
  avg          : (model + gemini) / 2  (가중치 튜닝 없음, 참고용)
  gemini_gtname: Gemini 1인분 값 (정답 음식명)   ← 음식명이 완벽할 때
  *_oracle_q   : 같은 1인분 값 × 정답 Q 비율
  (아래는 routed 와 같이 분류기 통과 사진은 영양 모델, 나머지만 바꿈. 표의 oracle_q 열은 이 조건들에서는 의미 없음)
  r_ing             : Gemini 1인분 값을 음식명 + 재료(gemini_serving_ing.jsonl)로 받은 것 × 양추정 비율  ← 처음 보는 음식 기본안
  r_mfds            : Gemini 1인분 값을 식약처 DB 기준 프롬프트(gemini_serving_mfds.jsonl)로 받은 것 × 양추정 비율
  r_dens[_mfds]_direct : Gemini g당 밀도(영양값/1인분 중량) × 직접 예측 영양 모델(_ing_cat)의 실제 중량 예측  ← 1인분 정의를 쓰지 않음
  r_dens[_mfds]_q4     : Gemini g당 밀도 × (1인분 영양 모델의 1인분 중량 예측 × 양추정 비율)
  r_dens[_mfds]_truemass : Gemini g당 밀도 × 정답 중량 (중량이 완벽할 때 상한)
출력 EXP/gemini_serving_eval.json.  usage: python eval_gemini_serving.py"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts_ing  # noqa: E402
from compare_pipeline import EXP, KEYS, RATIO, metrics  # noqa: E402


def main():
    recs = json.loads((EXP / "records.json").read_text())
    G = {j["food"]: [j[k] for k in KEYS] for j in map(json.loads, open(EXP / "gemini_serving.jsonl"))}
    GI = {j["food"]: [j[k] for k in KEYS] for j in map(json.loads, open(EXP / "gemini_serving_ing.jsonl"))}
    ING = texts_ing.ing_records()
    GM = {j["food"]: [j[k] for k in KEYS] for j in map(json.loads, open(EXP / "gemini_serving_mfds.jsonl"))}
    out = {}
    for sp in ("122_2_forunseen", "122_1_forseen"):
        Q = np.load(EXP / "runs" / f"{sp}_quantity/test_logits_pipe.npz"); ids = Q["id"]; qt = Q["label"]; qp = Q["logits_pipe_osr"].argmax(1)
        d = np.load(EXP / "runs" / f"{sp}_q4_ing_cat/preds_text_pipe_osr.npz"); pos = {int(i): k for k, i in enumerate(d["id"])}
        p_model = d["pred"][[pos[int(i)] for i in ids]]
        pn = texts_ing.pipe_names(sp, osr=True)
        p_gem = np.array([G[pn[int(i)][0]] for i in ids]); p_gemgt = np.array([G[recs[int(i)]["food"]] for i in ids])
        known = np.array([pn[int(i)][1] == "classifier" for i in ids])
        p_route = np.where(known[:, None], p_model, p_gem)
        y = np.array([recs[int(i)]["y"] for i in ids], np.float64); y4 = y / RATIO[qt][:, None]
        foods = np.array([recs[int(i)]["food"] for i in ids]); uf = sorted(set(foods))
        S = {"model": p_model, "gemini": p_gem, "routed": p_route, "avg": (p_model + p_gem) / 2, "gemini_gtname": p_gemgt}
        res = {"n": int(len(ids)), "classifier_route_rate": float(known.mean()), "conds": {}}
        # 실제 양 기준 최종값을 바로 만드는 조건 (×양추정 비율을 하지 않음)
        dd = np.load(EXP / "runs" / f"{sp}_ing_cat/preds_text_pipe_osr.npz"); pd_ = {int(i): k for k, i in enumerate(dd["id"])}
        p_direct = dd["pred"][[pd_[int(i)] for i in ids]]; ratio = RATIO[qp][:, None]
        mass = {"direct": p_direct[:, 4], "q4": p_model[:, 4] * RATIO[qp], "truemass": y[:, 4]}
        final = {"model": p_model * ratio, "routed": p_route * ratio}
        p_gi = np.array([GI[texts_ing.render(pn[int(i)][0], ING[(int(i), pn[int(i)][0])]["ingredients"])] if not k else [0.0] * 5
                         for i, k in zip(ids, known)])
        final["r_ing"] = np.where(known[:, None], p_model, p_gi) * ratio
        S["routed_ing"] = np.where(known[:, None], p_model, p_gi)
        final["r_mfds"] = np.where(known[:, None], p_model, np.array([GM[pn[int(i)][0]] for i in ids])) * ratio
        for tag, src in (("", G), ("_mfds", GM)):
            gv = np.array([src[pn[int(i)][0]] for i in ids]); dens = gv / np.maximum(gv[:, 4:5], 1e-6)
            for mk, m in mass.items():
                final[f"r_dens{tag}_{mk}"] = np.where(known[:, None], p_model * ratio, dens * m[:, None])
        res["final"] = {k: metrics(v, y) for k, v in final.items()}
        res["mass_G"] = {k: round(float(np.abs(m - y[:, 4]).sum() / y[:, 4].sum() * 100), 2) for k, m in
                         {**mass, "gemini_serving_x_ratio": p_gem[:, 4] * RATIO[qp], "gemini_mfds_serving_x_ratio": np.array([GM[pn[int(i)][0]][4] for i in ids]) * RATIO[qp]}.items()}
        res["serving_mass_ratio_median"] = {k: round(float(np.median(np.array([src[recs[int(i)]["food"]][4] for i in ids]) / y4[:, 4])), 2) for k, src in (("default", G), ("mfds", GM))}
        for k, p in S.items():
            ft, fp = np.array([y4[foods == f].mean(0) for f in uf]), np.array([p[foods == f].mean(0) for f in uf])
            res["conds"][k] = {"pipe_always": metrics(p * RATIO[qp][:, None], y), "oracle_q": metrics(p * RATIO[qt][:, None], y),
                               "serving": metrics(p, y4),
                               "food_slope": {KEYS[j]: round(float(np.polyfit(ft[:, j], fp[:, j], 1)[0]), 2) for j in range(4)},
                               "food_corr": {KEYS[j]: round(float(np.corrcoef(ft[:, j], fp[:, j])[0, 1]), 2) for j in range(4)}}
        res["per_food"] = {f: {"true_serving_cal": round(float(y4[foods == f, 0].mean()), 1),
                               **{k: {"G4": metrics(S[k][foods == f] * RATIO[qp[foods == f]][:, None], y[foods == f])["G4"],
                                      "serving_cal": round(float(S[k][foods == f, 0].mean()), 1)} for k in ("model", "gemini", "routed_ing", "gemini_gtname")}}
                           for f in uf}
        out[sp] = res
        print(f"===== {sp} n {len(ids)} 분류기 통과 {known.mean():.4f}")
        for k, v in res["conds"].items():
            print(f"  {k:14s} G4 {v['pipe_always']['G4']:6.2f} A4 {v['pipe_always']['A4']:6.2f} | oracle_q {v['oracle_q']['G4']:6.2f} | 1인분 G4 {v['serving']['G4']:6.2f}"
                  f" | G {v['pipe_always']['G']} | 기울기 {v['food_slope']} 상관 {v['food_corr']}")
        print("  -- 최종값 조건 (routed 기준)")
        for k, v in res["final"].items(): print(f"  {k:24s} G4 {v['G4']:6.2f} A4 {v['A4']:6.2f} | G {v['G']}")
        print("  중량 G (%, 실제 중량 기준):", res["mass_G"], "| Gemini 1인분 중량 / DB 1인분 중량 (정답 음식명, 중앙값):", res["serving_mass_ratio_median"])
    (EXP / "gemini_serving_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
