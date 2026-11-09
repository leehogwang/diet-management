"""argmax vs expected_T1 (보정 전 확률 가중 평균) 차이의 신뢰도: 음식 단위·사진 단위 paired bootstrap (CE 양추정 모델, r_ing).
usage: python bootstrap_expected_t1.py"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts_ing  # noqa: E402
from compare_pipeline import EXP, KEYS, RATIO, metrics  # noqa: E402

sp = "122_2_forunseen"; recs = json.loads((EXP / "records.json").read_text())
GI = {j["food"]: [j[k] for k in KEYS] for j in map(json.loads, open(EXP / "gemini_serving_ing.jsonl"))}
ING = texts_ing.ing_records(); pn = texts_ing.pipe_names(sp, osr=True)
Q = np.load(EXP / "runs" / f"{sp}_quantity/test_logits_pipe.npz"); ids, lg = Q["id"], Q["logits_pipe_osr"]
d = np.load(EXP / "runs" / f"{sp}_q4_ing_cat/preds_text_pipe_osr.npz"); pos = {int(i): k for k, i in enumerate(d["id"])}
p_model = d["pred"][[pos[int(i)] for i in ids]]; known = np.array([pn[int(i)][1] == "classifier" for i in ids])
p_gi = np.array([GI[texts_ing.render(pn[int(i)][0], ING[(int(i), pn[int(i)][0])]["ingredients"])] if not k else [0.0] * 5 for i, k in zip(ids, known)])
p1 = np.where(known[:, None], p_model, p_gi); y = np.array([recs[int(i)]["y"] for i in ids], np.float64)
z = lg - lg.max(1, keepdims=True); p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
A, B = p1 * RATIO[lg.argmax(1)][:, None], p1 * (p @ RATIO)[:, None]
foods = np.array([recs[int(i)]["food"] for i in ids]); uf = np.unique(foods); idx = {f: np.where(foods == f)[0] for f in uf}
g4 = lambda P, T: (np.abs(P - T).sum(0) / T.sum(0))[:4].mean() * 100
print("전체: argmax", round(g4(A, y), 2), "expected_T1", round(g4(B, y), 2))
print("영양소별 G argmax", metrics(A, y)["G"], "\n           exp_T1", metrics(B, y)["G"])
rng = np.random.default_rng(0)
for name, draw in (("음식 단위", lambda: np.concatenate([idx[f] for f in rng.choice(uf, len(uf))])), ("사진 단위", lambda: rng.integers(0, len(y), len(y)))):
    dd = []
    for _ in range(2000):
        s = draw(); dd.append(g4(B[s], y[s]) - g4(A[s], y[s]))
    dd = np.array(dd); print(f"{name}: 차이(exp_T1 - argmax) 평균 {dd.mean():.2f}, 95% 구간 [{np.percentile(dd, 2.5):.2f}, {np.percentile(dd, 97.5):.2f}], exp_T1 이 나은 비율 {(dd < 0).mean():.2f}")
per = sorted(((g4(B[idx[f]], y[idx[f]]) - g4(A[idx[f]], y[idx[f]]), f) for f in uf))
print("음식별 차이 (음수 = exp_T1 이 나음):", [(f, round(v, 1)) for v, f in per])
print("exp_T1 이 나은 음식 수", sum(v < 0 for v, _ in per), "/", len(per))
