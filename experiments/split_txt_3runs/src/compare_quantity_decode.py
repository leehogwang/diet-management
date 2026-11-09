"""양추정 출력 → 비율 변환 방식 비교 (재학습 없음). 대상 모델·최종값 계산은 compare_quantity_soft 와 같음 (r_ing, 대분류 pipe_osr).
  argmax   : 확률 최고 Q 의 비율 (지금)
  expected : Σ p_k × 비율_k  (확률 가중 평균, p 는 val 로 맞춘 temperature 적용)
  expected_T1 : 같은 것, temperature 없이 (raw softmax)
  min_rel  : Σ_k p_k |r_j - r_k| / r_k 가 가장 작은 비율 r_j (Q1~Q5 중 하나) — 상대 오차 기대값 최소
  min_rel_cont : 같은 기대값을 최소로 하는 연속 비율 (= 가중치 p_k/r_k 의 가중 중앙값)
  median / median_T1 : 확률 분포의 중앙값 Q 의 비율 (Σ|오차| 기대값 최소 = G 지표에 맞는 선택)
출력 EXP/quantity_decode_compare.json.  usage: python compare_quantity_decode.py"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts_ing  # noqa: E402
from compare_pipeline import EXP, KEYS, RATIO, metrics  # noqa: E402
from compare_quantity_soft import TAGS  # noqa: E402


def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def wmedian(p):  # 가중 중앙값: Σ p_k/r_k |r - r_k| 최소, 해는 r_k 중 하나
    w = p / RATIO[None]; c = np.cumsum(w, 1) / w.sum(1, keepdims=True); return RATIO[(c >= 0.5).argmax(1)]


def main(sp="122_2_forunseen"):
    recs = json.loads((EXP / "records.json").read_text())
    GI = {j["food"]: [j[k] for k in KEYS] for j in map(json.loads, open(EXP / "gemini_serving_ing.jsonl"))}
    ING = texts_ing.ing_records(); pn = texts_ing.pipe_names(sp, osr=True); out = {}
    for tag, label in TAGS.items():
        rd = EXP / "runs" / f"{sp}_quantity{tag}"; Q = np.load(rd / "test_logits_pipe.npz")
        T = json.loads((rd / "report_pipe.json").read_text())["temperature"]
        ids, qt, lg = Q["id"], Q["label"], Q["logits_pipe_osr"]
        d = np.load(EXP / "runs" / f"{sp}_q4_ing_cat/preds_text_pipe_osr.npz"); pos = {int(i): k for k, i in enumerate(d["id"])}
        p_model = d["pred"][[pos[int(i)] for i in ids]]; known = np.array([pn[int(i)][1] == "classifier" for i in ids])
        p_gi = np.array([GI[texts_ing.render(pn[int(i)][0], ING[(int(i), pn[int(i)][0])]["ingredients"])] if not k else [0.0] * 5
                         for i, k in zip(ids, known)])
        p1 = np.where(known[:, None], p_model, p_gi); y = np.array([recs[int(i)]["y"] for i in ids], np.float64)
        p, p1t = softmax(lg / T), softmax(lg); rt = RATIO[qt]
        rel = np.abs(RATIO[:, None] - RATIO[None]) / RATIO[None]  # [j 예측, k 정답]
        R = {"argmax": RATIO[lg.argmax(1)], "expected": p @ RATIO, "expected_T1": p1t @ RATIO,
             "min_rel": RATIO[(p @ rel.T).argmin(1)], "min_rel_cont": wmedian(p),
             "median": RATIO[(np.cumsum(p, 1) >= 0.5).argmax(1)], "median_T1": RATIO[(np.cumsum(p1t, 1) >= 0.5).argmax(1)]}
        out[tag or "ce"] = {"T": T, **{k: {"G4": metrics(p1 * r[:, None], y)["G4"], "A4": metrics(p1 * r[:, None], y)["A4"],
                                         "ratio_mae": round(float(np.abs(r - rt).mean()), 4),
                                         "ratio_rel": round(float((np.abs(r - rt) / rt).mean()), 4),
                                         "rel_by_trueQ": [round(float((np.abs(r - rt) / rt)[qt == q].mean()), 3) for q in range(5)]}
                                     for k, r in R.items()}}
        print(f"== {label} (T={T:.3f})")
        for k, v in out[tag or "ce"].items():
            if k != "T": print(f"  {k:13s} G4 {v['G4']:6.2f} A4 {v['A4']:6.2f} | 비율 MAE {v['ratio_mae']:.4f} 비율 상대오차 {v['ratio_rel']:.4f} | 정답 Q별 상대오차 {v['rel_by_trueQ']}")
    (EXP / "quantity_decode_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
