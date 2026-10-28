"""논문용 최종 성능표 (2026-10-02).

최종 챔피언 = side 3체인 `b8_sidefull_s{42,1,2}` (⑦ 순정 재학습은 10/02 판정에서 음성 → 유지).
test 507접시를 체크포인트에서 **다시 추론**해 시드별·앙상블·중량구간 수치를 같은 예측에서 낸다.
시드별 값은 각 런의 test_metrics.json 과 대조해 일치를 확인한다(불일치 시 중단).

조기종료 기준 전환 표는 7/25 TASK5 산출물 `results/task5/sota_rmse_ensemble.json` 에서 읽는다
(cs384_crop / mx_pma / mx_rmse 계열 — 최종 챔피언과는 다른 이전 계보의 실험).

  CUDA_VISIBLE_DEVICES=3 python -m scripts.task_batch.final_paper_table
"""
from __future__ import annotations

import json

import numpy as np

from scripts.task_batch import score_hybrid as S
from scripts.task_batch.task2_common import dish_targets5, metrics, NUTR4

import matplotlib.pyplot as plt  # krfont 는 score_hybrid 가 이미 설정

OUT = S.ROOT / "results/final"
CHAMP = "b8_sidefull_s{}"
S.ARMS = [("최종 챔피언 (side 3체인)", CHAMP, "#1f3a68", True)]
BINS = [("0~100g", 0, 100), ("100~300g", 100, 300), ("300~500g", 300, 500), ("500g+", 500, np.inf)]
ROWS = [("평균 PMA", "mean_PMA", "%"), ("칼로리 PMA", "calories_PMA", "%"), ("지방 PMA", "fat_PMA", "%"),
        ("탄수화물 PMA", "carb_PMA", "%"), ("단백질 PMA", "protein_PMA", "%"),
        ("칼로리 RMSE (kcal)", "calories_RMSE", ""), ("nRMSE (4영양소 평균)", "mean_nRMSE", "%")]


def msd(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1))


def main():
    dishes, T, per_seed, ens, meta = S.collect()
    label = S.ARMS[0][0]
    P = per_seed[label]                      # {seed: (507,4)}
    seeds = S.SEEDS
    t5 = dish_targets5("test")
    mass = np.array([t5[d][4] for d in dishes])

    # ── 시드별 지표 + test_metrics.json 대조
    seed_m = {}
    for s in seeds:
        m = metrics(P[s], T, NUTR4)
        ref = json.loads((S.RUNS / CHAMP.format(s) / "test_metrics.json").read_text())
        for _, k, _ in ROWS:
            if abs(m[k] - ref[k]) > 0.01:
                raise SystemExit(f"불일치 seed{s} {k}: 재추론 {m[k]:.4f} vs 기록 {ref[k]:.4f}")
        seed_m[s] = m
    E = metrics(ens[label], T, NUTR4)
    ref_ens = json.loads((OUT / "final_scores.json").read_text())["기준선 (side 3체인 · patience 10)"]["ensemble"]
    assert abs(E["calories_PMA"] - ref_ens["calories_PMA"]) < 0.01, "앙상블이 9/28 기록과 다름"

    overall = {k: {"seed_mean": msd([seed_m[s][k] for s in seeds])[0],
                   "seed_std": msd([seed_m[s][k] for s in seeds])[1],
                   "per_seed": {str(s): seed_m[s][k] for s in seeds},
                   "ensemble": E[k]} for _, k, _ in ROWS}

    # ── 중량 구간별 칼로리 PMA (구간 안에서 PMA = 100·mean|err| / mean(정답))
    bins = []
    for name, lo, hi in BINS:
        idx = np.where((mass >= lo) & (mass < hi))[0]
        ps = [metrics(P[s][idx], T[idx], NUTR4) for s in seeds]
        pe = metrics(ens[label][idx], T[idx], NUTR4)
        bins.append({"bin": name, "n": int(len(idx)),
                     "cal_PMA_ensemble": pe["calories_PMA"],
                     "cal_PMA_seed_mean": msd([p["calories_PMA"] for p in ps])[0],
                     "cal_PMA_seed_std": msd([p["calories_PMA"] for p in ps])[1],
                     "cal_RMSE_ensemble": pe["calories_RMSE"],
                     "mean_mass_g": float(mass[idx].mean())})
    assert sum(b["n"] for b in bins) == len(dishes)

    # ── 조기종료 기준 전환 (7/25 TASK5)
    t5j = json.loads((S.ROOT / "results/task5/sota_rmse_ensemble.json").read_text())
    arms = {"old": ("PMA기준(현행 챔피언)", "cs384_crop_seed{}"),
            "ctrl": ("이어학습 monitor=PMA(통제)", "mx_pma_seed{}"),
            "new": ("RMSE기준(신규)", "mx_rmse_seed{}")}
    mon = {}
    for key, (ek, pat) in arms.items():
        sd = [t5j[pat.format(s)] for s in seeds]
        mon[key] = {"ens_cal_PMA": t5j[ek]["calories_PMA"], "ens_cal_RMSE": t5j[ek]["calories_RMSE"],
                    "seed_cal_PMA": [x["calories_PMA"] for x in sd],
                    "seed_cal_RMSE": [x["calories_RMSE"] for x in sd]}
    dp = [a - b for a, b in zip(mon["new"]["seed_cal_PMA"], mon["ctrl"]["seed_cal_PMA"])]
    dr = [a - b for a, b in zip(mon["new"]["seed_cal_RMSE"], mon["ctrl"]["seed_cal_RMSE"])]
    mon["net"] = {"ens_cal_PMA": mon["new"]["ens_cal_PMA"] - mon["ctrl"]["ens_cal_PMA"],
                  "ens_cal_RMSE": mon["new"]["ens_cal_RMSE"] - mon["ctrl"]["ens_cal_RMSE"],
                  "seed_delta_cal_PMA": dp, "seed_delta_cal_RMSE": dr}

    res = {"_meta": {"date": "2026-10-02", "champion": "side 3체인 b8_sidefull_s{42,1,2}",
                     "n_test": len(dishes), "seeds": seeds,
                     "std": "시드간 표본표준편차(ddof=1)",
                     "ensemble": "3시드 예측 평균",
                     "PMA": "100·mean|pred−true| / mean(true)",
                     "nRMSE": "4영양소 nRMSE(=100·RMSE/mean(true)) 평균"},
           "overall": overall, "by_mass": bins, "monitor_switch": mon,
           "best_epochs": [m.get("best_epoch") for m in meta[label]],
           "epochs_run": [m.get("epochs_run") for m in meta[label]],
           "hit_epoch_cap": [bool(m.get("hit_epoch_cap")) for m in meta[label]]}
    (OUT / "FINAL_performance_table.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    np.savez(OUT / "FINAL_champion_preds.npz", dishes=np.array(dishes), T=T, mass=mass,
             **{f"P_seed{s}": P[s] for s in seeds}, P_ens=ens[label])
    write_md(res)
    write_png(res)
    print("저장", OUT / "FINAL_performance_table.md")


def fmt(v, unit, nd=2):
    return f"{v:.{nd}f}{unit}"


def write_md(r):
    o, b, m = r["overall"], r["by_mass"], r["monitor_switch"]
    nd = lambda k: 1 if k == "calories_RMSE" else 2  # noqa: E731
    L = ["# 최종 성능표 — Nutrition5k test 507접시 (2026-10-02)", "",
         "**최종 챔피언** = side 3체인 (ConvNeXt-S@384 RGB-D 4ch · 접시크롭 · 재료그램 aux · "
         "단계0 side 사전학습 → 단계1 → 단계2 monitor=nRMSE), 시드 42·1·2.  ",
         f"best epoch(단계2) {r['best_epochs']} / 실행 {r['epochs_run']} · 상한 도달 없음.  ",
         "⑦ 순정 재학습(pat25+mae_norm_bal)은 2026-10-02 판정에서 칼로리 부호 1/3 으로 음성 → 챔피언 유지.", "",
         "## 1. 전체 성능", "",
         "| 지표 | 수치 (3시드 평균±SD) | 3시드 앙상블 | 시드 42 / 1 / 2 |",
         "|------|------|------|------|"]
    for name, k, u in ROWS:
        x = o[k]; d = nd(k)
        ps = " / ".join(f"{x['per_seed'][str(s)]:.{d}f}" for s in r["_meta"]["seeds"])
        L.append(f"| {name} | {x['seed_mean']:.{d}f}±{x['seed_std']:.{d}f}{u} | {x['ensemble']:.{d}f}{u} | {ps} |")
    L += ["", "SD = 시드간 표본표준편차(ddof=1, n=3). PMA = 100·mean|예측−정답| / mean(정답). "
          "nRMSE = 4영양소 각각 100·RMSE/mean(정답) 의 평균.", "",
          "## 2. 중량 구간별 칼로리 PMA", "",
          "| 구간 | n | 칼로리 PMA (앙상블) | 3시드 평균±SD | 칼로리 RMSE (앙상블) |",
          "|------|---|-----------|------|------|"]
    for x in b:
        L.append(f"| {x['bin']} | {x['n']} | {x['cal_PMA_ensemble']:.2f}% | "
                 f"{x['cal_PMA_seed_mean']:.2f}±{x['cal_PMA_seed_std']:.2f}% | {x['cal_RMSE_ensemble']:.1f} |")
    L += ["", "구간은 정답 접시 중량 기준 [하한, 상한). 구간 PMA 의 분모는 그 구간 정답 칼로리 평균이라 "
          "구간 간 PMA 는 분모가 다르다(소량 접시는 분모가 작아 PMA 가 커진다).", "",
          "## 3. 조기종료 기준 전환 효과 (2026-07-25 TASK5)", "",
          "| 조건 | 칼로리 PMA | 칼로리 RMSE |", "|------|-----------|-----------|"]
    for key, name in [("old", "기존 챔피언 cs384_crop (monitor=PMA)"),
                      ("ctrl", "통제군 mx_pma (더 학습, monitor=PMA)"),
                      ("new", "신규 mx_rmse (더 학습, monitor=RMSE)")]:
        L.append(f"| {name} | {m[key]['ens_cal_PMA']:.2f}% | {m[key]['ens_cal_RMSE']:.1f} |")
    n = m["net"]
    L.append(f"| **순효과 (신규 − 통제군)** | **{n['ens_cal_PMA']:+.2f}%p** | **{n['ens_cal_RMSE']:+.1f}** |")
    dp, dr = n["seed_delta_cal_PMA"], n["seed_delta_cal_RMSE"]
    L += ["", "모든 행은 3시드 앙상블. 짝지은 시드별 순효과(시드 42 / 1 / 2): "
          f"칼로리 PMA {' / '.join(f'{v:+.2f}' for v in dp)} (평균 {np.mean(dp):+.2f}%p, 3/3 개선), "
          f"칼로리 RMSE {' / '.join(f'{v:+.2f}' for v in dr)} (평균 {np.mean(dr):+.2f}, 3/3 개선).", "",
          "⚠ 이 표는 최종 챔피언 이전 계보(cs384_crop, 단계0 side 사전학습 없음)에서 측정한 것이다. "
          "최종 챔피언은 이 결과를 받아 단계2 를 monitor=nRMSE 로 학습했지만, "
          "최종 챔피언 계보에서 PMA↔RMSE 기준만 바꾼 통제 비교는 따로 하지 않았다.", "",
          "원자료: `results/final/FINAL_performance_table.json` · 예측 `FINAL_champion_preds.npz` · "
          "생성기 `scripts/task_batch/final_paper_table.py`"]
    (OUT / "FINAL_performance_table.md").write_text("\n".join(L) + "\n")


def write_png(r):
    o, b, m = r["overall"], r["by_mass"], r["monitor_switch"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9.6), gridspec_kw={"height_ratios": [8, 5, 5]})
    navy, light = "#1f3a68", "#eef2f8"

    def table(ax, title, head, rows):
        ax.axis("off"); ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=navy)
        t = ax.table(cellText=rows, colLabels=head, loc="upper center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.45)
        for (i, j), c in t.get_celld().items():
            c.set_edgecolor("#c8ced8")
            if i == 0:
                c.set_facecolor(navy); c.set_text_props(color="white", fontweight="bold")
            elif i % 2 == 0:
                c.set_facecolor(light)

    rows = []
    for name, k, u in ROWS:
        d = 1 if k == "calories_RMSE" else 2
        rows.append([name, f"{o[k]['seed_mean']:.{d}f} ± {o[k]['seed_std']:.{d}f}{u}", f"{o[k]['ensemble']:.{d}f}{u}"])
    table(axes[0], "1. 전체 성능 (test 507접시)", ["지표", "3시드 평균 ± SD", "3시드 앙상블"], rows)
    table(axes[1], "2. 중량 구간별 칼로리 PMA", ["구간", "n", "앙상블", "3시드 평균 ± SD"],
          [[x["bin"], x["n"], f"{x['cal_PMA_ensemble']:.2f}%",
            f"{x['cal_PMA_seed_mean']:.2f} ± {x['cal_PMA_seed_std']:.2f}%"] for x in b])
    n = m["net"]
    table(axes[2], "3. 조기종료 기준 전환 효과 (3시드 앙상블, 이전 계보)", ["조건", "칼로리 PMA", "칼로리 RMSE"],
          [["기존 챔피언 (monitor=PMA)", f"{m['old']['ens_cal_PMA']:.2f}%", f"{m['old']['ens_cal_RMSE']:.1f}"],
           ["통제군 (더 학습, monitor=PMA)", f"{m['ctrl']['ens_cal_PMA']:.2f}%", f"{m['ctrl']['ens_cal_RMSE']:.1f}"],
           ["신규 (더 학습, monitor=RMSE)", f"{m['new']['ens_cal_PMA']:.2f}%", f"{m['new']['ens_cal_RMSE']:.1f}"],
           ["순효과 (신규 − 통제군)", f"{n['ens_cal_PMA']:+.2f}%p", f"{n['ens_cal_RMSE']:+.1f}"]])
    fig.text(0.01, 0.005, "최종 챔피언 = side 3체인 (ConvNeXt-S@384 RGB-D, 시드 42·1·2). SD = 시드간 표본표준편차.",
             fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "FINAL_performance_table.png", dpi=200, facecolor="white")


if __name__ == "__main__":
    main()
