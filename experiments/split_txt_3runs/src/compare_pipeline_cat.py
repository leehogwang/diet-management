"""대분류 입력 영양 모델(_ing_cat) 기준 파이프라인 비교. 122 두 분할, test 조건 gt / gemini / pipe (texts_ing 참고).
정답은 실제 양 기준(영양DB 1인분 × 정답 Q 비율). 같은 조건의 텍스트·대분류를 영양 모델과 양추정 모델에 함께 넣는다.
  direct         : 영양 모델이 실제 양 기준 값을 바로 예측 (runs/<split>_ing_cat)
  pipe_q4        : 1인분 예측 그대로 (runs/<split>_q4_ing_cat)
  pipe_gated     : 1인분 × 양추정 argmax 비율, 대분류가 기타면 보정 안 함 (파이프라인 기본: --quantity-gate known_category --quantity-ratio argmax)
  pipe_always    : 1인분 × 양추정 argmax 비율, 항상 보정
  pipe_expected_gated : 1인분 × Q 확률 가중 평균 비율, 기타면 보정 안 함
  oracle_q       : 1인분 × 정답 Q 비율
텍스트가 없는(Gemini 결과·임베딩 없음) 사진은 그 조건에서 빼고 n 에 남긴다.
양추정 logits 는 runs/<split>_quantity/test_logits_pipe.npz (eval_quantity_cats.py).
출력 EXP/pipeline_compare_ing_cat.json, EXP/pipeline_compare_rows_ing_cat.csv.  usage: python compare_pipeline_cat.py"""
import csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts_ing  # noqa: E402
from compare_pipeline import EXP, KEYS, RATIO, metrics, softmax  # noqa: E402


def main():
    recs = json.loads((EXP / "records.json").read_text()); out, rows = {}, []
    for sp in ("122_1_forseen", "122_2_forunseen"):
        Q = np.load(EXP / "runs" / f"{sp}_quantity/test_logits_pipe.npz"); T = json.loads((EXP / "runs" / f"{sp}_quantity/report_pipe.json").read_text())["temperature"]
        ids = Q["id"]; pos = {int(i): k for k, i in enumerate(ids)}; qtrue = Q["label"]
        y = np.array([recs[int(i)]["y"] for i in ids], np.float64)
        pn, po = texts_ing.pipe_names(sp), texts_ing.pipe_names(sp, osr=True)
        names = {"gt": {int(i): recs[int(i)]["food"] for i in ids}, "gemini": texts_ing.gemini_names(), "pipe": {i: n for i, (n, _) in pn.items()},
                 "pipe_osr": {i: n for i, (n, _) in po.items()}}
        res = {"pipe_name_source_classifier": float(np.mean([pn[int(i)][1] == "classifier" for i in ids])),
               "pipe_osr_name_source_classifier": float(np.mean([po[int(i)][1] == "classifier" for i in ids]))}
        for tm in ("gt", "gemini", "pipe", "pipe_osr"):
            def load(run):
                d = np.load(EXP / "runs" / run / f"preds_text_{tm}.npz"); idx = np.array([pos[int(i)] for i in d["id"]])
                p = np.zeros((len(ids), 5)); ok = np.zeros(len(ids), bool); p[idx] = d["pred"]; ok[idx] = d["ok"]; return p, ok
            (direct, ok1), (p4, ok2) = load(f"{sp}_ing_cat"), load(f"{sp}_q4_ing_cat"); ok = ok1 & ok2
            prob = softmax(Q[f"logits_{tm}"] / T); qa = prob.argmax(1); cat = Q[f"cat_{tm}"]; known = cat != texts_ing.UNKNOWN
            r_arg, r_exp = RATIO[qa], prob @ RATIO
            M = {"direct": direct, "pipe_q4": p4, "pipe_gated": p4 * np.where(known, r_arg, 1.0)[:, None], "pipe_always": p4 * r_arg[:, None],
                 "pipe_expected_gated": p4 * np.where(known, r_exp, 1.0)[:, None], "oracle_q": p4 * RATIO[qtrue][:, None]}
            nm = names[tm]
            res[f"text_{tm}"] = {k: metrics(v[ok], y[ok]) for k, v in M.items()}
            res[f"text_{tm}_info"] = {"n_ok": int(ok.sum()), "n_missing": int((~ok).sum()), "known_category": round(float(known[ok].mean()), 4),
                                      "name_acc": round(float(np.mean([nm.get(int(i)) == recs[int(i)]["food"] for i in ids[ok]])), 4),
                                      "quantity_acc": round(float((qa[ok] == qtrue[ok]).mean()), 4)}
            res[f"text_{tm}_by_q"] = {k: [round(float(metrics(v[ok & (qtrue == q)], y[ok & (qtrue == q)])["G4"]), 2) for q in range(5)] for k, v in M.items()}
            for j, i in enumerate(ids):
                if not ok[j]: continue
                rows.append({"split": sp, "cond": tm, "id": int(i), "food": recs[int(i)]["food"], "name": nm.get(int(i)), "category": cat[j], "view": recs[int(i)]["view"],
                             "q_true": int(qtrue[j]) + 1, "q_pred": int(qa[j]) + 1, **{f"true_{k}": round(float(y[j, n]), 2) for n, k in enumerate(KEYS)},
                             **{f"{m}_{k}": round(float(M[m][j, n]), 2) for m in ("direct", "pipe_q4", "pipe_gated") for n, k in enumerate(KEYS)}})
        out[sp] = res
        print(sp, {tm: {k: v["G4"] for k, v in res[f"text_{tm}"].items()} for tm in ("gt", "gemini", "pipe", "pipe_osr")})
    (EXP / "pipeline_compare_ing_cat.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    with open(EXP / "pipeline_compare_rows_ing_cat.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
