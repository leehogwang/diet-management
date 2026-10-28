"""평가 인프라 — 부트스트랩 신뢰구간 · 검출력 · 층화 test 리포트 (2026-08-21).

## 왜 필요한가
지금까지의 판정 규칙은 "시드 노이즈 ±0.56pp 를 넘는가 + 3시드 부호가 일치하는가"였다.
이 규칙은 **시드 간 변동만** 다루고 **test 표본 자체의 불확실성**은 다루지 않는다. test 는 507접시고
그중 500g+ 는 26장, 900kcal+ 는 8장뿐이라, 구간별 결론은 표본 열 몇 장에 걸려 있다.
실제로 2026-08-21 크롭 통일이 −0.24pp(3/3 일치) → 순정 재학습에서 −0.03pp 로 뒤집혔다.

## 무엇을 재는가
1) **부트스트랩 신뢰구간** — 접시를 복원추출해 지표의 95% CI 를 낸다(전체/구간별).
2) **짝지은 부트스트랩** — arm 과 챔피언의 Δ 를 **같은 접시 재표집**으로 잰다. 두 모델이 같은
   접시를 보므로 짝을 지으면 공통 분산이 상쇄돼 훨씬 예민하다. Δ 의 CI 가 0 을 가르지 않으면 유의.
3) **검출력(최소검출효과, MDE)** — 각 구간에서 80% 검출력으로 잡아낼 수 있는 최소 Δ.
   "이 구간에서는 애초에 몇 pp 미만은 못 가린다"를 미리 알려준다.
4) **층화 리포트** — 중량·칼로리 구간별 n 과 CI 를 한 표로. n 이 규칙 미달인 구간은 명시적으로
   '판단 보류'로 찍는다(숫자를 내놓되 결론을 내지 않는다).

## 판정 규칙 v2 (이 파일이 정의한다)

두 종류의 불확실성이 있고, **둘 다 통과해야** 결론을 확정한다.

  (a) **표본 불확실성** — test 접시가 507장(구간별로는 8~324장)뿐이라서 생기는 흔들림.
      짝지은 부트스트랩 95% CI 가 0 을 배제하면 '이 test 표본에서 유의'.
  (b) **재학습 불확실성** — 같은 설정으로 다시 학습하면 달라지는 폭. 3시드 부호 일치가 1차 증거,
      **체인 전체를 다시 만든 순정 재학습에서 재현되는지**가 최종 증거다(STEP 11-1b).

  확정 = (a) ∧ (b).  하나만 만족하면 '시사점' 수준으로만 쓴다.
  실제로 크롭 통일은 (b) 3시드 부호는 통과했으나 (a) 도 (b) 순정 재현도 실패해 확정되지 않았다.

## n < 30 구간을 다루는 법
- **절대값**은 CI 가 넓으므로 참고값으로만 쓴다(500g+ n=26 의 칼로리 PMA CI 는 ±3.7pp).
- 다만 **짝지은 Δ 는 여전히 예민할 수 있다.** 두 모델이 같은 접시를 보므로 공통 분산이 상쇄되기
  때문이다(예: Max Pooling 의 500g+ Δ 는 n=26 에서도 +2.93 [+1.63,+4.33] 로 유의).
  → 규칙: **절대값은 보류, Δ 는 CI 로 판단**하되 n 을 반드시 함께 적는다.

## 시드 부호 일치를 과신하지 말 것
같은 출발 체크포인트를 공유하는 이어학습 arm 끼리는 시드 산포가 인위적으로 작다
(이어학습 std 0.09 vs 순정 재학습 std 0.46). 부호 일치는 **보조** 증거다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/students/cs/202520906/nutrition5k_pipeline")
sys.path.insert(0, str(ROOT))

NUTR4 = ["calories", "fat", "carb", "protein"]
MASS_BINS = [("0-100g", 0, 100), ("100-300g", 100, 300),
             ("300-500g", 300, 500), ("500g+", 500, np.inf)]
CAL_BINS = [("0-300", 0, 300), ("300-600", 300, 600),
            ("600-900", 600, 900), ("900+", 900, np.inf)]
MIN_N = 30              # 이보다 작은 구간은 '판단 보류'
N_BOOT = 4000
ALPHA = 0.05
RNG_SEED = 20260821


def mean_pma(P, T, idx=None):
    """PMA(%) = 100 * mean|err| / mean(정답). idx=None -> 4영양소 평균."""
    e = np.abs(P - T)
    if idx is not None:
        return 100.0 * e[:, idx].mean() / max(T[:, idx].mean(), 1e-8)
    return float(np.mean([100.0 * e[:, i].mean() / max(T[:, i].mean(), 1e-8) for i in range(4)]))


def cal_rmse(P, T):
    return float(np.sqrt(((P[:, 0] - T[:, 0]) ** 2).mean()))


def boot_idx(n, n_boot=N_BOOT, seed=RNG_SEED):
    """접시 복원추출 인덱스 (모든 arm 이 **같은 재표집**을 쓰도록 시드 고정)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def ci(vals, alpha=ALPHA):
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def stat_ci(P, T, fn, B):
    """지표 하나의 점추정 + 95% CI."""
    point = fn(P, T)
    vals = np.array([fn(P[b], T[b]) for b in B])
    lo, hi = ci(vals)
    return {"point": float(point), "lo": lo, "hi": hi, "se": float(vals.std()), "n": int(len(T))}


def paired_delta(Pa, Pb, T, fn, B):
    """arm − 챔피언 Δ 의 짝지은 부트스트랩. 같은 접시 재표집이라 공통 분산이 상쇄된다."""
    point = fn(Pa, T) - fn(Pb, T)
    vals = np.array([fn(Pa[b], T[b]) - fn(Pb[b], T[b]) for b in B])
    lo, hi = ci(vals)
    return {"delta": float(point), "lo": lo, "hi": hi, "se": float(vals.std()),
            "significant": bool(lo > 0 or hi < 0),
            "direction": ("개선" if point < 0 else "악화") if (lo > 0 or hi < 0) else "구별불가"}


def mde(P, T, fn, B, power=0.80):
    """최소검출효과 — 이 표본에서 80% 검출력으로 잡히는 최소 |Δ|.

    짝지은 Δ 의 표준오차 se 를 쓰면 MDE ≈ (z_{1-a/2} + z_power) * se = 2.80 * se.
    여기서는 '같은 모델을 조금 흔든 것'의 se 를 근사하기 위해 지표 자체의 se 를 쓴다(보수적).
    """
    vals = np.array([fn(P[b], T[b]) for b in B])
    return float(2.80 * vals.std())


def bucket_rows(P, T, values, bins, fn, B, Pref=None):
    rows = []
    for label, lo, hi in bins:
        m = (values >= lo) & (values < hi)
        n = int(m.sum())
        if n == 0:
            continue
        sub = np.where(m)[0]
        # 구간 내부에서만 복원추출한다(구간 크기는 고정, 구간 안 구성만 흔든다)
        rng = np.random.default_rng(RNG_SEED + hash(label) % 1000)
        Bs = rng.choice(sub, size=(N_BOOT // 4, n), replace=True)
        point = fn(P[sub], T[sub])
        vals = np.array([fn(P[b], T[b]) for b in Bs])
        lo_, hi_ = ci(vals)
        r = {"bin": label, "n": n, "point": float(point), "lo": lo_, "hi": hi_,
             "se": float(vals.std()), "mde80": float(2.80 * vals.std()),
             "verdict_ok": n >= MIN_N}
        if Pref is not None:
            d = fn(P[sub], T[sub]) - fn(Pref[sub], T[sub])
            dv = np.array([fn(P[b], T[b]) - fn(Pref[b], T[b]) for b in Bs])
            dlo, dhi = ci(dv)
            r["delta"] = float(d); r["delta_lo"] = dlo; r["delta_hi"] = dhi
            r["delta_significant"] = bool(dlo > 0 or dhi < 0)
        rows.append(r)
    return rows
