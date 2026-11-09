"""양추정 soft label 비교 (재학습 없음, 학습된 모델의 test_logits_pipe.npz 사용).
대상: runs/<split>_quantity{tag}/  tag = "" (기존 CE + label smoothing 0.1), _soft05 (σ=0.5), _soft10 (σ=1.0)
양추정 입력 대분류는 pipe_osr (현재 최고 파이프라인과 같음). 최종 영양값은 eval_gemini_serving 의 r_ing 와 같은 방식:
  분류기 통과(아는 음식) → 1인분 영양 모델, 나머지 → Gemini 1인분(음식명 + 재료)  ×  양추정 argmax 비율
출력 EXP/quantity_soft_compare.json.  usage: python compare_quantity_soft.py [--split 122_2_forunseen]"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts_ing  # noqa: E402
from compare_pipeline import EXP, KEYS, RATIO, metrics  # noqa: E402

TAGS = {"": "CE (smoothing 0.1)", "_soft05": "soft σ=0.5", "_soft10": "soft σ=1.0", "_aug": "강한 색·질감 증강", "_grlaug": "GRL 음식 정체성 제거 + 강한 증강", "_grl05aug": "GRL λ0.5 + 강한 증강", "_grl20aug": "GRL λ2.0 + 강한 증강", "_grl": "GRL λ1.0 (기존 증강)"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="122_2_forunseen"); a = ap.parse_args(); sp = a.split
    recs = json.loads((EXP / "records.json").read_text())
    GI = {j["food"]: [j[k] for k in KEYS] for j in map(json.loads, open(EXP / "gemini_serving_ing.jsonl"))}
    ING = texts_ing.ing_records(); pn = texts_ing.pipe_names(sp, osr=True)
    out = {}
    for tag, label in TAGS.items():
        f = EXP / "runs" / f"{sp}_quantity{tag}" / "test_logits_pipe.npz"
        if not f.exists(): print("없음:", f); continue
        Q = np.load(f); ids = Q["id"]; qt = Q["label"]; qp = Q["logits_pipe_osr"].argmax(1)
        d = np.load(EXP / "runs" / f"{sp}_q4_ing_cat/preds_text_pipe_osr.npz"); pos = {int(i): k for k, i in enumerate(d["id"])}
        p_model = d["pred"][[pos[int(i)] for i in ids]]
        known = np.array([pn[int(i)][1] == "classifier" for i in ids])
        p_gi = np.array([GI[texts_ing.render(pn[int(i)][0], ING[(int(i), pn[int(i)][0])]["ingredients"])] if not k else [0.0] * 5
                         for i, k in zip(ids, known)])
        p1 = np.where(known[:, None], p_model, p_gi)
        y = np.array([recs[int(i)]["y"] for i in ids], np.float64)
        err = np.abs(qp - qt); conf = np.zeros((5, 5), int)
        for t, p in zip(qt, qp): conf[t, p] += 1
        fin = metrics(p1 * RATIO[qp][:, None], y)
        r = {"label": label, "n": int(len(ids)), "acc": round(float((err == 0).mean()), 4),
             "abs_err_rate": {str(k): round(float((err == k).mean()), 4) for k in range(5)},
             "within1": round(float((err <= 1).mean()), 4), "under": round(float((qp < qt).mean()), 4), "over": round(float((qp > qt).mean()), 4),
             "ratio_mae": round(float(np.abs(RATIO[qp] - RATIO[qt]).mean()), 4),
             "per_q_acc": [round(float(conf[q, q] / conf[q].sum()), 4) for q in range(5)], "confusion": conf.tolist(),
             "r_ing": fin, "r_ing_oracle_q": metrics(p1 * RATIO[qt][:, None], y)}
        out[tag or "ce"] = r
        print(f"{label:20s} acc {r['acc']:.3f} | ±1 {r['abs_err_rate']['1']:.3f} ±2 {r['abs_err_rate']['2']:.3f} ≥3 {1 - r['within1'] - r['abs_err_rate']['2']:.3f}"
              f" | 비율 MAE {r['ratio_mae']:.4f} | Q별 정확도 {r['per_q_acc']} | r_ing G4 {fin['G4']:.2f} A4 {fin['A4']:.2f} G {fin['G']}")
    (EXP / "quantity_soft_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
