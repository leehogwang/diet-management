"""STEP 2 채점 + 시각화 — 지방 절충안 배터리 (2026-09-20).

판정 규칙 v2 위에 **명세의 성공 기준 3가지**를 그대로 얹어 본다.
  ① 지방 PMA 3시드 평균 ≤ 22.41%(기준선)   ② 칼로리 PMA 3시드 평균 ≤ 15.24%
  ③ 지방 PMA 3시드 부호 일치(전부 개선 방향)

`analyze_capacity` 가 내는 mean_PMA 중심 표와 달리, 여기서는 **영양소별**(특히 지방)로
시드별 값·짝지은 부트스트랩 CI 를 전부 낸다. STEP 14 의 교훈(평균만 보면 원인을 놓친다).

  CUDA_VISIBLE_DEVICES=2 python -m scripts.task_batch.score_hybrid
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path("/home/students/cs/202520906/nutrition5k_pipeline")
sys.path.insert(0, str(ROOT))
from scripts.task_batch import krfont  # noqa: E402
krfont.setup()
import matplotlib.pyplot as plt  # noqa: E402

from scripts.task_batch.train_grams import parse_grams, usable, VOCAB_K  # noqa: E402
from scripts.task_batch.train_b8 import B8Net, SpatialCropDS  # noqa: E402
from scripts.task_batch.task2_common import dish_targets5, metrics, NUTR4  # noqa: E402
from scripts.task_batch.eval_infra import boot_idx, cal_rmse, mean_pma, paired_delta  # noqa: E402

RUNS = ROOT / "outputs/task_runs/task_b8/runs"
OUT = ROOT / "results/hybrid"
WEEK = ROOT / "results/weekly"
SEEDS = [42, 1, 2]
KOR = {"calories": "칼로리", "fat": "지방", "carb": "탄수", "protein": "단백"}
#         라벨                      런 이름 패턴        막대색      강조
ARMS = [("직접 예측 (기준선)",      "b8_sidefull_s{}", "#9aa5b1", False),
        ("재료 무게 경유",          "b8_dec196_s{}",   "#e8833a", False),
        ("절충안 (지방 전용 FC)",    "b8_hyb196_s{}",   "#2f6fd0", True),
        ("절충안 (지방=챔피언+보정)", "b8_hybres_s{}",   "#7fa9e8", False),
        ("대안B (지방 FC 3층)",     "b8_hybdeep_s{}",  "#5b4b8a", False)]
TARGET_FAT, TARGET_CAL = 22.41, 15.24
SCORE_NAME = "step2_scores"      # results/hybrid/<SCORE_NAME>.json
FIG_PREFIX = "step2"             # results/weekly/<FIG_PREFIX>_pma_all.png ...
LOSS_PAT = "b8_hyb196_s{}"       # 학습곡선을 그릴 arm
LOSS_TITLE = "절충안 (지방 전용 FC)"
STEP_NO = "STEP 2"        # 그림 제목 머리말


@torch.no_grad()
def predict(run, loader, dishes, device):
    ckp, meta = RUNS / run / "checkpoints/best.pt", RUNS / run / "test_metrics.json"
    if not (ckp.exists() and meta.exists()):
        return None, None
    st = torch.load(ckp, map_location=device, weights_only=False)
    cfg = st.get("cfg", {})
    m = B8Net(pool=cfg.get("pool", "avg"), spatial_dim=cfg.get("sp_dim", 0),
              sp_embed=cfg.get("sp_embed", 64), pretrained=False,
              arch=cfg.get("arch", "convnext_small"), vocab_k=cfg.get("vocab_k", VOCAB_K),
              decode=cfg.get("decode", "head"), fat_head=cfg.get("fat_head", "dedicated"))
    m.load_state_dict(st["model_state"]); m = m.to(device).eval()
    got = {}
    for x, _, _, dids, s in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out, _ = m(x.to(device), s.to(device))
        for d, pv in zip(dids, out.float().cpu().numpy()):
            got[d] = pv
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.stack([got[d] for d in dishes]), json.loads(meta.read_text())


def collect():
    device = torch.device(os.environ.get("SCORE_DEVICE", "cuda"))
    grams = parse_grams()
    tr_d, _ = usable("train"); te_d, te_t = usable("test")
    df = Counter()
    for d in tr_d:
        for n in grams.get(d, {}):
            df[n] += 1
    vocab = [n for n, _ in df.most_common(VOCAB_K)]
    G = np.array([[np.log1p(grams.get(d, {}).get(n, 0.0)) for n in vocab] for d in tr_d], np.float64)
    dishes = sorted(te_d)
    t5 = dish_targets5("test")
    T = np.stack([t5[d][:4] for d in dishes])
    ds = SpatialCropDS(dishes, te_t, grams, vocab, G.mean(0), G.std(0) + 1e-6, res=384,
                       train=False, spat_raw=None, sp_idx=None, smu=None, ssd=None)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)
    per_seed, ens, meta = {}, {}, {}
    print(f"test {len(dishes)}접시\n", flush=True)
    for label, pat, _, _ in ARMS:
        P, ok, mt = [], [], []
        for s in SEEDS:
            p, m = predict(pat.format(s), loader, dishes, device)
            if p is None:
                print(f"  [{pat.format(s):18s}] 런 없음"); continue
            P.append(p); ok.append(s); mt.append(m)
            print(f"  [{pat.format(s):18s}] 지방 {mean_pma(p,T,1):6.2f}%  칼로리 {mean_pma(p,T,0):6.2f}%"
                  f"  평균 {mean_pma(p,T):6.2f}%  ep{m.get('best_epoch','?')}/{m.get('epochs_run','?')}"
                  f"{'  [!]상한도달' if m.get('hit_epoch_cap') else ''}", flush=True)
        if not P:
            continue
        per_seed[label] = dict(zip(ok, P)); ens[label] = np.mean(P, 0); meta[label] = mt
        print(f"  => {label}: 앙상블 평균 {mean_pma(ens[label],T):.2f}% "
              f"지방 {mean_pma(ens[label],T,1):.2f}% 칼로리 {mean_pma(ens[label],T,0):.2f}%\n", flush=True)
    return dishes, T, per_seed, ens, meta


def score(T, per_seed, ens, meta, log):
    B = boot_idx(len(T))
    base = ARMS[0][0]
    R = {"_meta": {"n_test": int(len(T)), "seeds": SEEDS,
                   "rule": "명세 3조건(지방≤22.41 · 칼로리≤15.24 · 지방 3시드 부호일치) + 판정규칙 v2"}}
    log(f"{'arm':26s}" + "".join(f"{KOR[n]:>9}" for n in NUTR4) + f"{'평균':>9}{'칼RMSE':>9}")
    for label, _, _, _ in ARMS:
        if label not in ens:
            continue
        m = metrics(ens[label], T, NUTR4)
        R[label] = {"ensemble": m, "n_seeds": len(per_seed[label]),
                    "seeds": sorted(per_seed[label]),
                    "per_seed": {n: [float(mean_pma(per_seed[label][s], T, i))
                                     for s in sorted(per_seed[label])]
                                 for i, n in enumerate(NUTR4)},
                    "per_seed_mean_PMA": [float(mean_pma(per_seed[label][s], T))
                                          for s in sorted(per_seed[label])],
                    "best_epochs": [x.get("best_epoch") for x in meta[label]],
                    "hit_epoch_cap": [bool(x.get("hit_epoch_cap")) for x in meta[label]]}
        log(f"{label:26s}" + "".join(f"{m[n+'_PMA']:8.2f}%" for n in NUTR4)
            + f"{m['mean_PMA']:8.2f}%{m['calories_RMSE']:9.1f}")

    log("\n── 기준선 대비 Δ (짝지은 부트스트랩 95% CI · 같은 접시 재표집)")
    log(f"{'arm':26s}{'Δ지방':>9}{'95% CI':>17}{'Δ칼로리':>10}{'95% CI':>17}{'Δ평균':>9}")
    for label, _, _, _ in ARMS[1:]:
        if label not in ens:
            continue
        df_ = paired_delta(ens[label], ens[base], T, lambda P, t: mean_pma(P, t, 1), B)
        dc = paired_delta(ens[label], ens[base], T, lambda P, t: mean_pma(P, t, 0), B)
        dm = paired_delta(ens[label], ens[base], T, mean_pma, B)
        dr = paired_delta(ens[label], ens[base], T, cal_rmse, B)
        common = [s for s in SEEDS if s in per_seed[label] and s in per_seed[base]]
        sfat = [float(mean_pma(per_seed[label][s], T, 1) - mean_pma(per_seed[base][s], T, 1))
                for s in common]
        scal = [float(mean_pma(per_seed[label][s], T, 0) - mean_pma(per_seed[base][s], T, 0))
                for s in common]
        R[label].update({"delta_fat": df_, "delta_calories": dc, "delta_mean": dm,
                         "delta_cal_RMSE": dr, "per_seed_delta_fat": sfat,
                         "per_seed_delta_cal": scal})
        log(f"{label:26s}{df_['delta']:+9.2f}  [{df_['lo']:+5.2f},{df_['hi']:+5.2f}]"
            f"{dc['delta']:+10.2f}  [{dc['lo']:+5.2f},{dc['hi']:+5.2f}]{dm['delta']:+9.2f}")
        log(f"{'  시드별 Δ지방':26s}{str([round(v,2) for v in sfat]):>36}"
            f"   Δ칼로리 {[round(v,2) for v in scal]}")

    log("\n── 명세 성공 기준 (3개 모두 충족해야 채택)")
    for label, _, _, _ in ARMS[2:]:
        if label not in ens:
            continue
        pf = R[label]["per_seed"]["fat"]; pc = R[label]["per_seed"]["calories"]
        c1 = float(np.mean(pf)) <= TARGET_FAT
        c2 = float(np.mean(pc)) <= TARGET_CAL
        sf = R[label]["per_seed_delta_fat"]
        c3 = len(sf) == 3 and all(v < 0 for v in sf)
        ok = c1 and c2 and c3
        R[label]["spec_criteria"] = {"fat_mean_3seed": float(np.mean(pf)), "c1_fat<=22.41": c1,
                                     "cal_mean_3seed": float(np.mean(pc)), "c2_cal<=15.24": c2,
                                     "c3_fat_sign_consistent": c3, "adopt": ok}
        log(f"  {label}")
        log(f"    ① 지방 3시드 평균 {np.mean(pf):.2f}% ≤ 22.41  → {'통과' if c1 else '미달'}"
            f"   (시드별 {[round(v,2) for v in pf]})")
        log(f"    ② 칼로리 3시드 평균 {np.mean(pc):.2f}% ≤ 15.24 → {'통과' if c2 else '미달'}"
            f"   (시드별 {[round(v,2) for v in pc]})")
        log(f"    ③ 지방 Δ 3시드 부호 일치(개선)        → {'통과' if c3 else '미달'}"
            f"   (Δ {[round(v,2) for v in sf]})")
        log(f"    ⇒ 판정: {'★채택' if ok else '기각'}")
    return R


# ------------------------------------------------------------------ 그림
def fig_pma_all(T, ens, path):
    labels = [a[0] for a in ARMS if a[0] in ens]
    cols = {a[0]: a[2] for a in ARMS}; emph = {a[0]: a[3] for a in ARMS}
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    w = 0.8 / len(labels)
    xs = np.arange(4)
    for k, lb in enumerate(labels):
        m = metrics(ens[lb], T, NUTR4)
        v = [m[f"{n}_PMA"] for n in NUTR4]
        b = ax.bar(xs + (k - (len(labels)-1)/2) * w, v, width=w * 0.92, label=lb, color=cols[lb],
                   edgecolor=("#12448f" if emph[lb] else "white"), linewidth=(1.8 if emph[lb] else 0.8))
        for bb, vv in zip(b, v):
            ax.text(bb.get_x() + bb.get_width()/2, vv + 0.25, f"{vv:.1f}", ha="center",
                    va="bottom", fontsize=8.5,
                    fontweight=("bold" if emph[lb] else "normal"))
    ax.set_xticks(xs); ax.set_xticklabels([KOR[n] for n in NUTR4], fontsize=12)
    ax.set_ylabel("PMA (%)  ← 낮을수록 좋음")
    ax.set_title(f"{STEP_NO} ① 영양소별 PMA — arm 비교 (test 507접시 · 3시드 앙상블)")
    ax.legend(fontsize=9.5, ncol=2, loc="upper left"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(metrics(ens[l], T, NUTR4)[f"{n}_PMA"]
                       for l in labels for n in NUTR4) * 1.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    print(f"wrote {path}")


def fig_pma_key(T, ens, path):
    labels = [a[0] for a in ARMS if a[0] in ens]
    cols = {a[0]: a[2] for a in ARMS}; emph = {a[0]: a[3] for a in ARMS}
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    w = 0.8 / len(labels)
    xs = np.arange(2)
    for k, lb in enumerate(labels):
        m = metrics(ens[lb], T, NUTR4)
        v = [m["calories_PMA"], m["fat_PMA"]]
        b = ax.bar(xs + (k - (len(labels)-1)/2) * w, v, width=w * 0.92, label=lb, color=cols[lb],
                   edgecolor=("#12448f" if emph[lb] else "white"), linewidth=(2.0 if emph[lb] else 0.8))
        for bb, vv in zip(b, v):
            ax.text(bb.get_x() + bb.get_width()/2, vv + 0.3, f"{vv:.2f}%", ha="center",
                    va="bottom", fontsize=10, fontweight=("bold" if emph[lb] else "normal"))
    ax.axhline(TARGET_FAT, ls="--", color="#b3261e", lw=1.4)
    ax.axhline(TARGET_CAL, ls=":", color="#1b7f4b", lw=1.4)
    ax.set_xlim(-0.55, 1.95)
    ax.text(1.93, TARGET_FAT + 0.4, f"지방 목표 {TARGET_FAT}%", color="#b3261e",
            fontsize=10, ha="right")
    ax.text(1.93, TARGET_CAL + 0.4, f"칼로리 목표 {TARGET_CAL}%", color="#1b7f4b",
            fontsize=10, ha="right")
    ax.set_xticks(xs); ax.set_xticklabels(["칼로리", "지방"], fontsize=13)
    ax.set_ylabel("PMA (%)  ← 낮을수록 좋음")
    ax.set_title(f"{STEP_NO} ② 핵심 두 항목 — 칼로리와 지방 (점선 = 기준선 값)")
    ax.legend(fontsize=9.5, loc="upper left"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 34)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    print(f"wrote {path}")


def _pad(rows, n):
    """시드마다 에폭 수가 다르다. 짧은 시드는 NaN 으로 채워 **자르지 않고** 평균/범위를 낸다
    (자르면 ES 수직선이 그래프 밖으로 나간다)."""
    A = np.full((len(rows), n), np.nan)
    for i, r in enumerate(rows):
        A[i, :len(r)] = r
    return A


def fig_loss(path, pat="b8_hyb196_s{}", title="절충안 (지방 전용 FC)"):
    """학습 곡선. 학습기는 영양소별 train loss 를 따로 기록하지 않으므로(총 nutr_loss·aux_loss만),
    왼쪽 = 기록된 train loss, 오른쪽 = **에폭별 val PMA 를 영양소로 쪼갠 것**을 쓴다."""
    H = {}
    for s in SEEDS:
        f = RUNS / pat.format(s) / "training_history.json"
        if f.exists():
            H[s] = json.loads(f.read_text())
    if not H:
        print(f"skip {path} (history 없음)"); return
    n = max(len(v) for v in H.values())
    es = {s: json.loads((RUNS / pat.format(s) / "test_metrics.json").read_text())["best_epoch"]
          for s in H if (RUNS / pat.format(s) / "test_metrics.json").exists()}
    ep = np.arange(1, n + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    ax = axes[0]
    for key, lab, c in [("nutr_loss", "영양소 loss (SmoothL1, 4항목 합)", "#2f6fd0"),
                        ("aux_loss", "재료 그램 보조 loss × 30", "#e8833a")]:
        # 중단 후 로그에서 복원한 history 는 일부 키가 없을 수 있다 → NaN 으로 두고 nanmean 한다.
        A = _pad([[h.get(key, np.nan) * (30 if key == "aux_loss" else 1) for h in H[s]] for s in H], n)
        ax.plot(ep, np.nanmean(A, 0), color=c, lw=2.0, label=lab)
        ax.fill_between(ep, np.nanmin(A, 0), np.nanmax(A, 0), color=c, alpha=0.18)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("train loss (log)")
    ax.set_title(f"{title} — 학습 loss (3시드 평균 · 음영 = 시드 범위)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    v = lambda h, k: h.get("val", {}).get(k, np.nan)
    series = [("칼로리", lambda h: v(h, "calories_PMA"), "#1b7f4b"),
              ("지방 (직접 경로)", lambda h: v(h, "fat_PMA"), "#e8833a"),
              ("탄수+단백 (경유 경로)",
               lambda h: 0.5 * (v(h, "carb_PMA") + v(h, "protein_PMA")), "#2f6fd0")]
    for lab, fn, c in series:
        A = _pad([[fn(h) for h in H[s]] for s in H], n)
        ax.plot(ep, np.nanmean(A, 0), color=c, lw=2.0, label=lab)
        ax.fill_between(ep, np.nanmin(A, 0), np.nanmax(A, 0), color=c, alpha=0.18)
    for i, (s, e) in enumerate(sorted(es.items())):
        ax.axvline(e, ls="--", color="#5a6473", lw=1.0, alpha=0.8)
        ax.text(e, ax.get_ylim()[1] * (0.97 - 0.06 * i), f" ES s{s}", color="#5a6473",
                fontsize=9, va="top")
    ax.set_xlabel("epoch"); ax.set_ylabel("val PMA (%)")
    ax.set_title("에폭별 val PMA — 경로별로 쪼갠 것 (ES = 조기종료 선택 에폭)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    for A in axes:
        for sp in ("top", "right"):
            A.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    print(f"wrote {path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True); WEEK.mkdir(parents=True, exist_ok=True)
    lines = []
    def log(s):
        print(s, flush=True); lines.append(s)
    dishes, T, per_seed, ens, meta = collect()
    R = score(T, per_seed, ens, meta, log)
    # 수치를 먼저 저장한다 — 그림에서 죽어도 채점 결과는 남아야 한다(2026-09-28 사고).
    (OUT / f"{SCORE_NAME}.json").write_text(json.dumps(R, indent=2, ensure_ascii=False))
    np.savez(OUT / f"{SCORE_NAME.replace('scores','preds')}.npz", dishes=np.array(dishes), T=T,
             **{f"P_{i}": ens[l] for i, l in enumerate(ens)},
             labels=np.array(list(ens.keys())))
    fig_pma_all(T, ens, WEEK / f"{FIG_PREFIX}_pma_all.png")
    fig_pma_key(T, ens, WEEK / f"{FIG_PREFIX}_pma_key.png")
    fig_loss(WEEK / f"{FIG_PREFIX}_loss_curve.png", LOSS_PAT, LOSS_TITLE)
    (WEEK / f"{SCORE_NAME}.log").write_text("\n".join(lines))
    print(f"\nwrote {OUT/(SCORE_NAME + '.json')}")
    print("\n※ 이 스크립트는 챔피언을 교체하지 않는다. 교체는 사람이 승인한다.")


if __name__ == "__main__":
    main()
