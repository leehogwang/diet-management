"""Gemini 없이 대분류만으로 처음 보는 음식의 1인분을 정하는 실험 (재학습 없음). 분류기 통과(아는 음식) 사진은 영양 모델 그대로.
  cat_mean      : train 음식(분할 train, 음식 단위)의 대분류별 평균 1인분 영양값 × 양추정 비율
  cat_median    : 같은 것, 중앙값
  dens_x_catmass: 영양 모델 1인분 예측의 g당 밀도(영양값/중량) × 대분류별 1인분 중량 중앙값 × 양추정 비율
  model_x_catmass_ratio: 위와 같은 뜻 (영양 모델 1인분 × 대분류 중량/모델 중량)  ← dens_x_catmass 와 동일, 생략
대분류: test 는 pipe_osr 조건의 Gemini 대분류 (양추정 입력과 같음), *_gtcat 은 정답 대분류. train 에 없는 대분류는 전체 train 음식 평균/중앙값.
비교 기준: model (영양 모델 1인분), gemini_ing (gemini_serving_ing.jsonl).  usage: python eval_category_serving.py"""
import collections, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts_ing  # noqa: E402
from compare_pipeline import EXP, KEYS, RATIO, metrics  # noqa: E402


def main():
    recs = json.loads((EXP / "records.json").read_text()); S = json.loads((EXP / "splits.json").read_text())
    GI = {j["food"]: [j[k] for k in KEYS] for j in map(json.loads, open(EXP / "gemini_serving_ing.jsonl"))}; ING = texts_ing.ing_records()
    for sp in ("122_2_forunseen", "122_1_forseen"):
        # train 음식 단위 1인분 값 (정답 = DB 1인분, 음식 안에서 일정)
        tf = {}
        for i in S[sp]["train"]:
            r = recs[i]
            if r["ds"] == "122": tf[r["food"]] = (np.array(r["y"]) / RATIO[r["q"]], texts_ing.gt_cat(r))
        bycat = collections.defaultdict(list)
        for v, c in tf.values(): bycat[c].append(v)
        allv = np.array([v for v, _ in tf.values()])
        mean = {c: np.mean(v, 0) for c, v in bycat.items()}; med = {c: np.median(v, 0) for c, v in bycat.items()}
        Q = np.load(EXP / "runs" / f"{sp}_quantity/test_logits_pipe.npz"); ids = Q["id"]; qt = Q["label"]; ratio = RATIO[Q["logits_pipe_osr"].argmax(1)][:, None]
        d = np.load(EXP / "runs" / f"{sp}_q4_ing_cat/preds_text_pipe_osr.npz"); pos = {int(i): k for k, i in enumerate(d["id"])}
        pm = d["pred"][[pos[int(i)] for i in ids]]
        pn = texts_ing.pipe_names(sp, osr=True); ci = texts_ing.cond_inputs("pipe_osr", sp)
        known = np.array([pn[int(i)][1] == "classifier" for i in ids])
        cat_p = [ci[int(i)][1] for i in ids]; cat_g = [texts_ing.gt_cat(recs[int(i)]) for i in ids]
        y = np.array([recs[int(i)]["y"] for i in ids], np.float64)
        def look(tab, cats, fb): return np.array([tab.get(c, fb) for c in cats])
        dens = pm / np.maximum(pm[:, 4:5], 1e-6)
        U = {"model": pm,
             "cat_mean": look(mean, cat_p, allv.mean(0)), "cat_median": look(med, cat_p, np.median(allv, 0)),
             "cat_mean_gtcat": look(mean, cat_g, allv.mean(0)),
             "dens_x_catmass": dens * look(med, cat_p, np.median(allv, 0))[:, 4:5],
             "dens_x_catmass_gtcat": dens * look(med, cat_g, np.median(allv, 0))[:, 4:5],
             "train_mean_all": np.tile(allv.mean(0), (len(ids), 1)),
             "gemini_ing": np.array([GI[texts_ing.render(pn[int(i)][0], ING[(int(i), pn[int(i)][0])]["ingredients"])] if not k else pm[j]
                                     for j, (i, k) in enumerate(zip(ids, known))])}
        miss = sorted({c for c in cat_p if c not in mean})
        print(f"===== {sp} | train 음식 {len(tf)} | 대분류별 train 음식 수 {dict(sorted((c, len(v)) for c, v in bycat.items()))}")
        print(f"  test 대분류 중 train 에 없는 것 (전체 평균으로 대체): {miss} ({np.mean([c in miss for c in cat_p]):.3f})")
        for k, v in U.items():
            p = np.where(known[:, None], pm, v); m = metrics(p * ratio, y)
            print(f"  {k:22s} G4 {m['G4']:6.2f} A4 {m['A4']:6.2f} | G {m['G']}")


if __name__ == "__main__":
    main()
