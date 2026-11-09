"""양추정 기존(CE, 대분류 pipe_osr) vs Gemini 1인분 중량 입력(serv_in, serv_reg) 비교. 최종값은 r_ing (compare_quantity_soft 와 같음).
1인분 중량 조건: pipe (파이프라인 Gemini 값) / db (DB 정답, 상한).  출력 EXP/quantity_serving_compare.json
usage: python compare_quantity_serving.py"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts_ing  # noqa: E402
from compare_pipeline import EXP, KEYS, RATIO, metrics  # noqa: E402

SP = "122_2_forunseen"


def main():
    recs = json.loads((EXP / "records.json").read_text())
    GI = {j["food"]: [j[k] for k in KEYS] for j in map(json.loads, open(EXP / "gemini_serving_ing.jsonl"))}
    ING = texts_ing.ing_records(); pn = texts_ing.pipe_names(SP, osr=True)
    base = np.load(EXP / "runs" / f"{SP}_quantity/test_logits_pipe.npz"); ids = base["id"]; qt = base["label"]
    d = np.load(EXP / "runs" / f"{SP}_q4_ing_cat/preds_text_pipe_osr.npz"); pos = {int(i): k for k, i in enumerate(d["id"])}
    p_model = d["pred"][[pos[int(i)] for i in ids]]; known = np.array([pn[int(i)][1] == "classifier" for i in ids])
    p_gi = np.array([GI[texts_ing.render(pn[int(i)][0], ING[(int(i), pn[int(i)][0])]["ingredients"])] if not k else [0.0] * 5 for i, k in zip(ids, known)])
    p1 = np.where(known[:, None], p_model, p_gi); y = np.array([recs[int(i)]["y"] for i in ids], np.float64)
    foods = np.array([recs[int(i)]["food"] for i in ids]); uf = np.unique(foods)
    preds = {"기존 (사진 + 대분류)": base["logits_pipe_osr"].argmax(1)}
    for m in ("in", "reg"):
        f = EXP / "runs" / f"{SP}_quantity_serv_{m}/test_pred.npz"
        if not f.exists(): print("없음:", f); continue
        z = np.load(f); assert (z["id"] == ids).all() and (z["label"] == qt).all()
        preds[f"serv_{m} (Gemini 1인분)"] = z["q_pipe"]; preds[f"serv_{m} (DB 1인분, 상한)"] = z["q_db"]
        if m == "reg":
            mt = np.exp(z["log_mass_true"]); mp = np.exp(z["log_mass"])
            print(f"serv_reg 사진 중량 예측 G {np.abs(mp - mt).sum() / mt.sum() * 100:.2f}% (실제 중량 기준)")
    out = {}
    for k, qp in preds.items():
        e = np.abs(qp - qt); bias = [float((qp - qt)[foods == f].mean()) for f in uf]
        fin = metrics(p1 * RATIO[qp][:, None], y)
        out[k] = {"acc": round(float((e == 0).mean()), 4), "err1": round(float((e == 1).mean()), 4), "err2plus": round(float((e >= 2).mean()), 4),
                  "mid_acc": round(float((e == 0)[(qt >= 1) & (qt <= 3)].mean()), 4),
                  "per_q_acc": [round(float((qp == qt)[qt == q].mean()), 3) for q in range(5)],
                  "food_bias_abs_mean": round(float(np.mean(np.abs(bias))), 3), "G4": fin["G4"], "A4": fin["A4"], "G": fin["G"]}
        r = out[k]
        print(f"{k:24s} 정확도 {r['acc']:.3f} Q2~4 {r['mid_acc']:.3f} | 1단계 {r['err1']:.3f} 2단계+ {r['err2plus']:.3f} | Q별 {r['per_q_acc']}"
              f" | 음식별 치우침 |평균| {r['food_bias_abs_mean']:.2f} | G4 {r['G4']:.2f} A4 {r['A4']:.2f}")
    (EXP / "quantity_serving_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
