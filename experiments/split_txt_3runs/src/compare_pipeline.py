"""직접 예측(영양 모델이 실제 양 기준 값을 바로 예측) vs stage3_3 파이프라인(1인분 영양 모델 × 양추정 비율) 비교. 122 두 분할.
같은 test 사진(id)으로 맞춰 다음 방식을 계산한다. 정답은 모두 실제 양 기준(영양DB 1인분 × 정답 Q 비율).
  direct              : runs/<split>/preds_text_*.npz
  pipe_q4             : 보정 전. 1인분 예측 그대로 (runs/<split>_q4)
  pipe_argmax         : 1인분 예측 × 양추정 argmax Q 비율 (stage3_3 기본 --quantity-ratio argmax, gate always)
  pipe_expected       : 1인분 예측 × Q 확률 가중 평균 비율 (--quantity-ratio expected)
  oracle_q            : 1인분 예측 × 정답 Q 비율 (양추정이 완벽할 때)
  oracle_food         : 정답 1인분 × 양추정 argmax 비율 (영양 모델이 완벽할 때)
텍스트·대분류 조건: gt = 정답 음식명 + 정답 대분류, gemini = Gemini 음식명 + 그 이름의 대분류(정답표에 없으면 기타).
출력 EXP/pipeline_compare.json, EXP/pipeline_compare_rows.csv(사진별).  --suffix _ing: 영양 모델 런 <split>_ing, <split>_q4_ing 을 읽고 출력 파일 이름 끝에 _ing (양추정 런은 같음).
usage: python compare_pipeline.py [--suffix _ing]"""
import argparse, csv, json
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parents[1]
RATIO = np.array([0.25, 0.5, 0.75, 1.0, 1.25])
KEYS = ["calories", "fat", "protein", "carb", "mass"]


def metrics(P, T):
    e = np.abs(P - T); G = e.sum(0) / T.sum(0) * 100; A = (e / (np.abs(T) + 1)).mean(0) * 100
    return {"G4": round(float(G[:4].mean()), 2), "A4": round(float(A[:4].mean()), 2), "mean_pmae": round(float(G[:3].mean()), 2),
            "G": dict(zip(KEYS, G.round(2).tolist())), "MAE": dict(zip(KEYS, e.mean(0).round(2).tolist())), "n": int(len(T))}


def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--suffix", default=""); X = ap.parse_args().suffix
    recs = json.loads((EXP / "records.json").read_text()); out, rows = {}, []
    for sp in ("122_1_forseen", "122_2_forunseen"):
        Q = np.load(EXP / "runs" / f"{sp}_quantity/test_logits.npz"); qrep = json.loads((EXP / "runs" / f"{sp}_quantity/report.json").read_text())
        T = qrep["temperature"]; ids = Q["id"]; pos = {int(i): k for k, i in enumerate(ids)}
        qtrue = Q["label"]; y = np.array([recs[int(i)]["y"] for i in ids], np.float64); y4 = y / RATIO[qtrue][:, None]
        res = {"quantity": {k: {kk: qrep[f"test_cat_{k}"][kk] for kk in ("acc", "acc_within1", "ratio_mae_argmax", "ratio_mae_expected")}
                            for k in ("gt", "gemini", "masked")},
               "quantity_val_acc": qrep["val"]["acc"], "quantity_best_epoch": qrep["best_epoch"], "gemini_category_known": qrep["test_gemini_category_known"]}
        for tm in ("gt", "gemini"):
            def load(run):
                d = np.load(EXP / "runs" / run / f"preds_text_{tm}.npz"); idx = np.array([pos[int(i)] for i in d["id"]])
                p = np.zeros((len(ids), 5)); p[idx] = d["pred"]; return p
            direct, p4 = load(f"{sp}{X}"), load(f"{sp}_q4{X}")
            prob = softmax(Q[f"logits_{tm}"] / T); qa = prob.argmax(1); r_arg = RATIO[qa]; r_exp = prob @ RATIO
            M = {"direct": direct, "pipe_q4": p4, "pipe_argmax": p4 * r_arg[:, None], "pipe_expected": p4 * r_exp[:, None],
                 "oracle_q": p4 * RATIO[qtrue][:, None], "oracle_food": y4 * r_arg[:, None]}
            res[f"text_{tm}"] = {k: metrics(v, y) for k, v in M.items()}
            res[f"text_{tm}_q4_target"] = metrics(p4, y4)                   # 1인분 영양 모델 자체 성능 (1인분 정답 기준)
            res[f"text_{tm}_by_q"] = {k: [round(float(metrics(v[qtrue == q], y[qtrue == q])["G4"]), 2) for q in range(5)] for k, v in M.items()}
            for j, i in enumerate(ids):
                rows.append({"split": sp, "text": tm, "id": int(i), "food": recs[int(i)]["food"], "view": recs[int(i)]["view"], "q_true": int(qtrue[j]) + 1,
                             "q_pred": int(qa[j]) + 1, "true_cal": round(y[j, 0], 2), "direct_cal": round(direct[j, 0], 2),
                             "q4_cal": round(p4[j, 0], 2), "pipe_cal": round(M["pipe_argmax"][j, 0], 2)})
        out[sp] = res
        print(sp, {tm: {k: v["G4"] for k, v in out[sp][f"text_{tm}"].items()} for tm in ("gt", "gemini")}, "| Q acc", res["quantity"]["gt"]["acc"])
    (EXP / f"pipeline_compare{X}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    with open(EXP / f"pipeline_compare_rows{X}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
