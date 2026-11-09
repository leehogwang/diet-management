# 변경사항: 처음 보는 음식을 알아채는 분류기 (Open-set classifier)

- 작성일: 2026-11-02
- 위치: `food_b/experiments/split_txt_3runs/`
- 상태: 4개 런 완료 (2026-11-02). 결과·해석은 [7. 결과](#7-결과)
- **업데이트 (2026-11-02):** 7.5 권장안대로 판정 규칙을 바꾸고 영양 파이프라인에 연결함 (조건 `pipe_osr`). 내용은 [11. 추가 변경: 파이프라인 연결](#11-추가-변경-파이프라인-연결-pipe_osr), G4 재평가 결과는 [12](#12-파이프라인-재평가-결과-g4).
- 기존 파일·기존 결과(`runs/cls_*`, 영양 런, 파이프라인 비교)는 **하나도 수정하지 않음**. 새 파일만 추가.

---

## 1. 왜 바꿨나 (배경)

### 1.1 기존 분류기의 문제

기존 `src/train_classifier.py` (파이프라인 1단계 음식 분류기) 결과:

| split | 클래스 | val top1 | test 음식 | test top1 확률 평균 | test 확률 ≥ 0.7 비율 |
|---|---|---|---|---|---|
| `cls_122_1_forseen` | 78종 | 1.000 | 전부 학습 음식 | 1.000 | 1.000 |
| `cls_122_2_forunseen` | 44종 | 0.828 | **전부 처음 보는 음식** (23종) | 0.678 | **0.503** |

- 파이프라인(`src/texts_ing.py` `pipe_names`)은 `softmax 확률 ≥ 0.7` 이면 분류기 음식명을 쓰고, 미만이면 Gemini 음식명을 쓴다.
- unseen test 는 **모든 사진이 분류기 클래스에 없는 음식**이므로 분류기 답은 전부 오답이다. 그런데 50.3% 가 0.7 을 넘어 **오답 음식명이 그대로 다음 단계로 넘어갔다**.
- 그 결과 unseen 에서 pipe 텍스트의 대분류 정확도가 0.770 (Gemini 0.932) 로 떨어지고, 영양 G4 도 pipe 49.28 vs gemini 45.73 으로 나빠졌다.

### 1.2 원인

1. **softmax 는 상대값이다.** 확률 합이 1 이 되도록 후보끼리 나눠 갖기 때문에, 후보에 없는 음식이 들어와도 "그중 제일 비슷한 것"에 확률이 몰린다. "모르겠다"는 선택지가 구조적으로 없다.
2. **val 에 처음 보는 음식이 없었다.** 기존 val = "분할 val 중 train 음식" (+ unseen 은 train 10% 접시). threshold 0.7 을 처음 보는 음식 없는 조건에서 정했으므로 실제 unseen 상황의 위험이 드러나지 않았다 (val 에서는 threshold 0.7 → coverage 0.82, 정확도 1.0 으로 좋아 보였음).
3. **일반 cross-entropy 학습은 임베딩을 정돈하지 않는다.** 정답 점수만 제일 높으면 되므로 클래스 덩어리가 크고 서로 붙어 있어, 거리로 "아무 클래스와도 가깝지 않음"을 판단하기 어렵다.

### 1.3 대응 (세 가지)

| 번호 | 문제 | 대응 |
|---|---|---|
| 1 | softmax 과신 + val 에 unknown 없음 | 판정에 softmax 를 쓰지 않음. **val 에 처음 보는 음식을 넣고** threshold 를 val 로 정함 |
| 3 | 임베딩이 정돈되지 않음 | **ArcFace (metric learning)** 로 재학습 |
| 2 | 판정 점수 | ArcFace 임베딩에 맞는 **거리 기반 점수** (cos / kNN / Mahalanobis) + energy, msp 비교 |

Outlier Exposure(외부 음식 사진을 "모름"으로 학습)는 이번 변경에 **포함하지 않음** (결과를 보고 필요하면 추가).

---

## 2. 추가된 파일

| 파일 | 내용 |
|---|---|
| `src/train_classifier_osr.py` | open-set 분류기 학습·채점 스크립트 (신규) |
| `run_osr.sh` | 4개 런 일괄 실행 스크립트 (신규) |
| `logs/cls_osr_*.log`, `logs/run_osr.log` | 실행 로그 |
| `runs/cls_osr_<split>/` | ArcFace 결과 |
| `runs/cls_osr_<split>_baseline/` | 기존 CE 분류기를 같은 기준으로 채점한 결과 |
| `CHANGES_OSR_CLASSIFIER.md` | 이 문서 |

`train_classifier_osr.py` 는 기존 `train_classifier.py` 에서 `DS`(데이터셋), `group`(접시 묶음 키)을, `n296/food_classifier.py` 에서 `MODEL_NAME`, `train_transform`, `eval_transform`, `fit_temperature` 를 **import 해서 그대로 재사용**한다 (전처리·증강 동일).

---

## 3. 데이터 구성 변경 (1번)

### 3.1 네 가지 평가 세트

| 세트 | 용도 | 122_2_forunseen | 122_1_forseen |
|---|---|---|---|
| train | 학습 | 9,756 | 31,484 |
| known val | threshold 결정, epoch 선택 | 774 | 3,423 (분할 val) |
| **unknown val** | threshold 결정, epoch 선택 | **2,757** | 0 (처음 보는 음식 없음) |
| known test | 최종 평가 | 542 | 4,387 (분할 test) |
| unknown test | 최종 평가 | 5,884 (분할 test 전체) | 0 |

### 3.2 122_2_forunseen 구성 방법

- **unknown val** = 분할 val(`splits.json` 의 `122_2_forunseen.val`, 12종 2,989장) 중 **train 에 없는 음식** 11종 2,757장.
  - 앞선 누수 점검에서 확인: 분할 val 음식 ∩ test 음식 = **0종** → threshold 를 test 음식으로 정하는 누수 없음.
- **known val** = 분할 val 중 train 음식(1종, 232장) + train 접시 10% 를 떼어낸 것의 절반 (542장) = 774장.
- **known test** = train 접시 10% 중 나머지 절반 = 542장.
  - unseen 분할 test 에는 아는 음식이 없어 AUROC 를 계산할 수 없으므로, 아는 음식 test 를 따로 떼어냄.
- 접시 10% 선택은 **기존 `train_classifier.py` 와 같은 방식·같은 seed** (`group()` 키 = 파일명에서 `side_/top_` 제거, `random.Random(42).shuffle`, 앞 10%). 따라서 **기존 CE 모델도 known val / known test 사진을 학습에 쓰지 않았음** → baseline 비교가 공정.
  - 같은 접시의 다른 시점(side/top)은 같은 묶음이라 train 과 val/test 로 갈라지지 않음.
- train 9,756장은 기존 분류기와 동일.

### 3.3 122_1_forseen 구성 방법

- 처음 보는 음식이 존재하지 않는 분할이므로 unknown 세트 없음.
- known val / known test = 기존과 동일 (분할 val / 분할 test).
- threshold 는 known val 의 95% 통과 기준만 사용 (§5.2). 이 분할에서는 "아는 음식을 Gemini 로 잘못 넘기는 비율"이 늘지 않는지를 확인하는 용도.

---

## 4. 모델 변경 (3번: ArcFace)

### 4.1 구조

```
이미지 (288px)
  → convnext_tiny.fb_in22k_ft_in1k 백본 (num_classes=0, 768차원 특징)   ← 기존과 같은 백본
  → Linear(768→512) + BatchNorm1d                                    ← 신규 projection
  → 임베딩 z (512차원)
  → ArcFace 헤드: cos(z, W_c) (z, W 모두 L2 정규화) × scale 30          ← 기존 Linear 분류층 대체
```

- 학습 시: 정답 클래스의 각도 θ 에 margin m=0.3 을 더해 `cos(θ+m)` 로 벌점 → 같은 음식은 더 촘촘하게, 다른 음식과는 더 멀게.
  - θ+m 이 π 를 넘는 경우는 표준 근사 `cos θ − m·sin m` 사용.
- 추론 시: margin 없이 `30·cos(z, W_c)` 를 logit 으로 사용 (argmax = 예측 음식).
- 헤드 계산은 fp32 (AMP 에서도 `z.float()`).

### 4.2 학습 설정 (기존 대비)

| 항목 | 기존 `train_classifier.py` | 신규 `train_classifier_osr.py` |
|---|---|---|
| 백본 | convnext_tiny in22k→1k | 동일 |
| 입력·증강 | 288px, `train_transform` | 동일 |
| 분류층 | Linear (CE) | **ArcFace (dim 512, s 30, m 0.3)** |
| loss | CE + label smoothing 0.1 | CE on ArcFace logit (smoothing 없음) |
| 옵티마이저 | AdamW lr 1e-4, wd 0.05 | AdamW 백본 lr 1e-4, **projection·헤드 lr 1e-3**, wd 0.05 |
| 스케줄 | warmup 1 epoch + cosine | 동일 |
| drop_path | 0.1 | 동일 |
| AMP | fp16 | 동일 (`torch.amp.GradScaler("cuda")`) |
| 최대 epoch / patience | 30 / 6 | 동일 |
| **체크포인트 선택** | known val top1 최고 | unknown val 있으면 **(known val top1 + unknown val AUROC(proto_cos)) / 2 최고**, 없으면 known val top1 |
| temperature 보정 | val 로 | known val 로 (msp·energy 계산용) |
| seed | 42 | 42 |

---

## 5. 판정 점수와 threshold (2번)

### 5.1 점수 6가지 (모두 "클수록 아는 음식")

학습이 끝난 뒤 **train 사진 전체를 eval 변환으로 다시 통과**시켜 train 임베딩을 만들고, 이를 기준으로 계산한다. 임베딩은 L2 정규화 후 사용.

| 이름 | 계산 | 비고 |
|---|---|---|
| `msp` | softmax(logit / T) 최대값 | 기존 방식 (비교 기준) |
| `energy` | logsumexp(logit / T) | logit 절대 크기 반영. 재학습 불필요 |
| `proto_cos` | max_c cos(z, W_c) | ArcFace 클래스 가중치와의 최대 cos. ArcFace 전용 (baseline 에는 없음) |
| `center_cos` | max_c cos(z, μ_c) | μ_c = 클래스별 train 임베딩 평균 (정규화) |
| `knn` | train 임베딩과의 cos 중 k=10 번째로 큰 값 | 국소 밀도 반영 |
| `maha` | −min_c (z−μ_c)ᵀ Σ⁻¹ (z−μ_c) | 공유 공분산 Σ (+ 1e-3·평균분산 shrinkage) |

- baseline(기존 CE 모델)의 임베딩 = 분류층 직전 특징 (`forward_head(..., pre_logits=True)`, 768차원).
- train 사진이 하나도 없는 클래스는 μ_c 계산에서 제외 (smoke 테스트에서 발견한 NaN 수정).

### 5.2 threshold 두 가지

1. **`at_val_tpr95` (기본)**: known val 의 95% 가 통과하는 값 (`quantile(known val 점수, 0.05)`). OOD 평가 표준 방식. unknown 데이터가 없어도 정할 수 있어 두 분할 모두 적용.
2. **`at_val_balanced`**: unknown val 이 있을 때만. `(known 통과율 + unknown 차단율) / 2` 가 최대인 값.

threshold 는 **항상 val 로만 정하고 test 에 그대로 적용**한다.

### 5.3 라우팅 규칙

- 점수 ≥ threshold → 분류기 음식명 사용 (`route: "classifier"`)
- 점수 < threshold → Gemini 로 넘김 (`route: "gemini"`)
- `test_pred.json` 의 라우팅에 쓰는 점수(`primary`):
  - unknown val 이 있으면 **val AUROC 가 가장 높은 점수** 자동 선택
  - 없으면 `center_cos` (또는 `--primary` 로 지정)

> **업데이트:** 기본 라우팅 규칙이 바뀌었다 (기본 점수 `maha`, unknown val 이 있으면 val 균형 threshold, 없으면 known val 99% 통과). 위 내용은 최초 구현 기준이며, 현재 규칙은 [11.1](#111-분류기-판정-규칙-변경-train_classifier_osrpy) 참고.

---

## 6. 평가 지표와 출력

### 6.1 지표 (`report.json` → `scores.<점수이름>`)

| 키 | 의미 |
|---|---|
| `val_auroc`, `test_auroc` | 아는 음식 vs 처음 보는 음식 구분 능력 (0.5 = 무작위, 1.0 = 완벽) |
| `val_fpr_at95tpr` | 아는 음식 95% 통과 시 처음 보는 음식이 통과(오분류)되는 비율 |
| `test_fpr_at95tpr_valthr` | 위와 같되 val threshold 를 test 에 적용 |
| `known_coverage` | 아는 음식 중 분류기가 처리한 비율 |
| `known_acc_kept` | 분류기가 처리한 아는 음식의 정확도 |
| `known_acc_overall` | 아는 음식 전체 중 분류기가 맞힌 비율 (넘긴 것은 미포함) |
| `unknown_to_gemini` | 처음 보는 음식 중 Gemini 로 넘긴 비율 (**높을수록 좋음**) |
| `unknown_mislabeled` | 처음 보는 음식 중 분류기가 엉뚱한 이름을 붙인 비율 (**기존 msp≥0.7 에서 0.503**) |

그 밖에 `known_val_top1`, `known_test_top1`, `temperature`, `best_epoch`, `primary`, `n`(세트별 장수) 포함.

### 6.2 출력 파일 (`runs/cls_osr_<split>[_baseline]/`)

| 파일 | 내용 |
|---|---|
| `best.pth` | ArcFace 체크포인트 (`state_dict`, `classes`, `img_size`, `dim`, `scale`, `margin`, `epoch`). baseline 폴더에는 없음 |
| `report.json` | §6.1 지표 전체 |
| `test_pred.json` | known test + unknown test 사진별 결과 (아래) |
| `train.log` | epoch 로그 + 요약 |

`test_pred.json` 형식 (기존 `cls_*/test_pred.json` 의 `top`, `prob` 유지 + 필드 추가):

```json
{
  "12345": {
    "top": ["곱창전골", "부대찌개", "..."],
    "prob": [0.81, 0.07, "..."],
    "score": {"msp": 0.81, "energy": 21.3, "proto_cos": 0.42, "center_cos": 0.47, "knn": 0.55, "maha": -310.2},
    "route": "gemini"
  }
}
```

---

## 7. 결과

비교: 같은 split 에서 `cls_osr_<split>_baseline` (기존 CE 모델 + 각 점수, 재학습 없음) vs `cls_osr_<split>` (ArcFace + 각 점수).

### 7.1 122_2_forunseen — 판별력 (AUROC, 아는 음식 vs 처음 보는 음식)

| 점수 | 기존 CE val | 기존 CE test | ArcFace val | ArcFace test |
|---|---|---|---|---|
| msp | 0.757 | 0.989 | 0.771 | 0.981 |
| energy | 0.797 | 0.991 | 0.767 | 0.990 |
| proto_cos | – | – | 0.766 | 0.990 |
| center_cos | 0.755 | 0.996 | **0.838** | 0.997 |
| knn | 0.750 | 0.998 | 0.831 | 0.999 |
| maha | 0.730 | **1.000** | 0.766 | **1.000** |

### 7.2 122_2_forunseen — 실제 라우팅 결과 (test)

처음 보는 음식 5,884장 중 Gemini 로 넘긴 비율 (= 1 − 엉뚱한 이름이 붙은 비율). 아는 음식 test 542장의 통과 정확도는 모든 경우 1.000.

| 방법 | threshold 기준 | 처음 보는 음식 → Gemini | 엉뚱한 이름 | 아는 음식 통과율 |
|---|---|---|---|---|
| **기존 파이프라인** (CE, msp ≥ 0.7) | 고정값 | 0.497 | **0.503** | – |
| 기존 CE + msp | val TPR95 | 0.010 | 0.990 | 1.000 |
| 기존 CE + msp | val 균형 | 0.883 | 0.117 | 0.983 |
| 기존 CE + knn | val 균형 | 0.987 | 0.013 | 0.970 |
| **기존 CE + maha** | **val 균형** | **0.992** | **0.008** | **0.993** |
| ArcFace + center_cos | val TPR95 | 0.199 | 0.801 | 1.000 |
| ArcFace + center_cos | val 균형 | 0.937 | 0.063 | 0.982 |
| ArcFace + knn | val 균형 | 0.954 | 0.046 | 0.994 |
| ArcFace + maha | val 균형 | 0.987 | 0.013 | 0.991 |

val 균형 threshold 에서 val 의 아는 음식 중 통과한 사진의 정확도도 모두 1.000 이었다 (통과율 0.68~0.70, 아래 7.4).

### 7.3 122_1_forseen (아는 음식만)

| 모델 | known test top1 | val TPR95 threshold 에서 아는 음식 통과율 (msp / center_cos / maha) |
|---|---|---|
| 기존 CE | 1.000 | 0.992 / 0.951 / 0.953 |
| ArcFace | 1.000 | 0.949 / 0.955 / 0.955 |

정확도 100% 인 상황에서 TPR95 기준은 정의상 약 5% 를 불필요하게 Gemini 로 넘긴다.

### 7.4 해석

1. **효과는 1번(val 에 처음 보는 음식) + 2번(거리 점수)에서 나왔고, 3번(ArcFace)은 효과가 없었다.**
   - 기존 CE 모델을 재학습 없이 `maha` 점수 + val 균형 threshold 로 쓰면 엉뚱한 이름 0.503 → **0.008**.
   - ArcFace 는 val AUROC(center_cos 0.838 vs 0.755)는 올랐지만 test 라우팅은 기존 CE + maha 보다 좋지 않다.
   - ArcFace best epoch = 1. 이후 epoch 마다 known val top1 이 0.746 → 0.70 으로 떨어져, 학습이 진행될수록 촬영 환경(용기·배경)에 과적합하는 것으로 보인다.
2. **threshold 는 "val TPR95" 가 아니라 "val 균형" 이어야 한다.** TPR95 로는 처음 보는 음식의 1~20% 만 넘긴다. 원인은 다음 항목.
3. **known val 의 오므라이스 232장 = 아는 음식이지만 새 환경.** 분할 val 에서 train 과 겹치는 유일한 음식인 오므라이스는 train 은 `사각배달` 용기, val 은 `접시` 용기다. 두 모델 모두 이 사진들을 거의 다 틀린다 (known val top1 0.71~0.75 ≈ 떼어낸 접시 542장만 맞힘). 점수도 낮아 TPR95 threshold 를 끌어내린다.
   - val 균형 threshold 에서는 이 사진들이 대부분 Gemini 로 넘어가고, 통과한 사진은 100% 정답이다. 즉 **"틀릴 사진을 넘기는" 방향으로는 올바르게 동작**한다.
4. **test AUROC 0.99~1.00 은 낙관적이다.** known test 542장은 train 과 같은 촬영 묶음에서 떼어낸 접시라 train 과 거의 같은 사진이다 (top1 1.000). 실제 운영에서 "아는 음식을 새 환경에서 찍은" 사진은 오므라이스처럼 거부될 가능성이 크다. 진짜 판별력은 val AUROC(0.73~0.84) 쪽에 가깝다고 보는 게 안전하다.
5. **forseen 은 고정 TPR95 대신 별도 기준이 필요하다.** 처음 보는 음식이 없는 분할이라 균형 threshold 를 정할 수 없다. unseen 분할에서 정한 threshold 를 옮겨 쓰거나 TPR99 같은 느슨한 기준을 써야 한다.

### 7.5 권장안

- 파이프라인 1단계: **기존 CE 분류기 (`runs/cls_<split>/best.pth`) + `maha` 점수 + val 균형 threshold**. 재학습이 필요 없다. ArcFace 는 채택하지 않음.
- 판단 근거로는 test AUROC 대신 **val AUROC 와 unseen test 의 Gemini 라우팅 비율**을 쓴다.
- 다음 실험: 음식 단위 K-fold 로 unknown val 을 늘려 threshold 안정성 확인, "아는 음식 + 새 용기" 세트를 따로 만들어 평가.

---

## 8. 실행 방법

```bash
cd /home/leehogwang/food_b/experiments/split_txt_3runs
PY=/home/leehogwang/Food-Nutrition-Estimation/.venv/bin/python

# 일괄 (GPU0: ArcFace unseen / GPU1: baseline 2개 → ArcFace seen)
./run_osr.sh > logs/run_osr.log 2>&1

# 개별
$PY src/train_classifier_osr.py --split 122_2_forunseen --device cuda:0            # ArcFace 학습 + 채점
$PY src/train_classifier_osr.py --split 122_2_forunseen --baseline --device cuda:1 # 기존 CE 분류기 채점만
$PY src/train_classifier_osr.py --split 122_2_forunseen --eval_only                # 저장된 ArcFace best.pth 로 재채점
$PY src/train_classifier_osr.py --split 122_2_forunseen --smoke                    # 소량 동작 확인 (runs_smoke/)
```

주요 인자: `--margin 0.3`, `--scale 30`, `--dim 512`, `--primary <점수>`, `--epochs 30`, `--patience 6`, `--lr 1e-4`, `--img_size 288`, `--batch_size 64`, `--workers 10`.

---

## 9. 바꾸지 않은 것 / 아직 연결되지 않은 것

- **영양 파이프라인은 아직 기존 분류기를 사용한다.** `src/texts_ing.py` 의 `pipe_names()` 는 여전히 `runs/cls_<split>/test_pred.json` 과 `prob[0] ≥ 0.7` 규칙을 쓴다. 새 분류기 결과를 반영하려면:
  1. `pipe_names()` 가 `runs/cls_osr_<split>/test_pred.json` 의 `route` 필드를 보도록 수정
  2. 바뀐 음식명에 대해 `gemini_ing.py --names pipe` (재료·대분류) → `embed_ing.py` 재실행
  3. `eval_quantity_cats.py`, `train.py --eval_only`, `compare_pipeline_cat.py` 재실행
- 단, 새 `test_pred.json` 에는 unseen 분할의 **known test(train 에서 떼어낸 542장)** 도 들어 있다. 이 사진들은 영양 분할상 train 사진이므로, 파이프라인에 연결할 때는 분할 test id 만 사용해야 한다.
- 기존 `runs/cls_*`, 영양 런(`runs/122_*`), `pipeline_compare*.json` 은 수정하지 않음.
- 양 추정 모델, 영양 모델은 변경 없음.

> **업데이트:** 이 장의 "아직 연결되지 않은 것" 1~3단계는 조건 `pipe_osr` 로 연결·실행했다 ([11](#11-추가-변경-파이프라인-연결-pipe_osr)). 기존 `pipe` 조건(softmax ≥ 0.7)은 비교용으로 그대로 남겨 두었다. known test 542장 제외도 `texts_ing.pipe_names(osr=True)` 에서 처리했다.

---

## 10. 한계와 다음 단계

- **unknown val 은 11종뿐**이라 threshold 가 이 음식들에 맞춰질 수 있다. 음식 단위 K-fold(leave-foods-out)로 여러 번 돌려 편차를 봐야 한다.
- epoch 선택에 unknown val 을 쓰므로 **val 지표는 낙관적**이다. 판단은 test 지표로 해야 한다 (test 음식은 val 과 겹치지 않음).
- forunseen 의 known test 는 542장으로 작다.
- forseen 은 처음 보는 음식이 없어 open-set 판별력은 측정할 수 없다 (아는 음식 처리율 유지만 확인).
- 다음 후보:
  1. Outlier Exposure: N5k 등 외부 음식 사진에 균등 분포 정답을 주는 loss 추가 (unseen test 음식은 절대 사용 금지)
  2. margin·scale 조정 (m 0.2~0.5)
  3. 파이프라인 연결 후 영양 G4 재평가 (§9)
  4. 음식 단위 K-fold 평가

---

## 11. 추가 변경: 파이프라인 연결 (`pipe_osr`)

7.5 권장안(기존 CE 분류기 + `maha` + val 균형 threshold, ArcFace 미채택)을 파이프라인 1단계에 적용하고 영양 G4 를 다시 평가했다.
기존 `pipe`(softmax ≥ 0.7)는 지우지 않고, 새 조건 **`pipe_osr`** 를 추가해 **같은 모델·같은 test 에서 나란히 비교**한다.

### 11.1 분류기 판정 규칙 변경 (`train_classifier_osr.py`)

| 항목 | 최초 구현 (§5.3) | 변경 후 |
|---|---|---|
| `--primary` 기본값 | 없음 → unknown val AUROC 최고 자동 선택 (없으면 `center_cos`) | **`maha`** (자동 선택은 `--primary auto`) |
| threshold (unknown val 있음) | known val TPR95 | **val 균형** (`at_val_balanced`) |
| threshold (unknown val 없음, forseen) | known val TPR95 | **known val TPR99** (`--route_tpr 0.99`, 신규 인자) |
| `report.json` | – | **`route` 항목 추가**: `score`, `rule`, `threshold`, `test`(known test / unknown test 운영 지표) |

- TPR99 를 고른 이유: forseen 은 정확도 100% 라 TPR95 면 아는 음식 약 5% 를 불필요하게 Gemini 로 넘긴다 (§7.3). 처음 보는 음식이 없는 분할이라 균형 threshold 는 정할 수 없다.
- 재채점은 학습 없이 `--baseline` 으로만 실행 (기존 CE 모델 `runs/cls_<split>/best.pth`). `runs/cls_osr_<split>_baseline/{report.json, test_pred.json}` 이 새 규칙으로 갱신됨.

재채점 결과 (`logs/cls_osr_*_baseline_route.log`):

| split | 점수 / 규칙 | threshold | 아는 음식 통과율 (known test) | 통과분 정확도 | 처음 보는 음식 → Gemini | 엉뚱한 이름 |
|---|---|---|---|---|---|---|
| 122_2_forunseen | maha / val 균형 | −2323.0 | 0.993 | 1.000 | **0.9915** | **0.0085** |
| 122_1_forseen | maha / known val TPR99 | −2058.8 | 0.993 | 1.000 | – | – |

### 11.2 파이프라인 스크립트 변경

| 파일 | 변경 |
|---|---|
| `src/texts_ing.py` | `pipe_names(split, osr=False)` 에 `osr` 인자 추가. `osr=True` 면 `runs/cls_osr_<split>_baseline/test_pred.json` 의 **`route`** 로 분류기/Gemini 음식명 선택, **분할 test id 만** 사용 (forunseen 의 known test 542장 = 영양 분할상 train 사진 제외). `cond_inputs("pipe_osr", split)` 지원. docstring 에 `pipe_osr` 설명 추가 |
| `src/gemini_ing.py` | `--names pipe_osr` 선택지 추가 (새로 필요한 (id, 음식명) 쌍만 Gemini 재료·대분류 호출) |
| `src/eval_quantity_cats.py` | 대분류 조건에 `pipe_osr` 추가 → `report_pipe.json` 에 `test_cat_pipe_osr` 등, `test_logits_pipe.npz` 에 `logits_pipe_osr`, `cat_pipe_osr` |
| `src/train.py` | `--text ing` 의 test 조건에 `pipe_osr` 추가 → `preds_text_pipe_osr.npz`, `result.json` 의 `test.text_pipe_osr` |
| `src/compare_pipeline_cat.py` | 비교 조건에 `pipe_osr` 추가, `pipe_osr_name_source_classifier`(분류기 음식명을 쓴 비율) 추가 |

`pipe` 와 `pipe_osr` 의 차이는 **1단계 음식명을 분류기에서 가져올지 Gemini 에서 가져올지 정하는 규칙 하나뿐**이다. 2단계 이후(Gemini 재료·대분류, 임베딩, 양 추정, 영양 모델)는 같은 모델·같은 설정이다.

### 11.3 실행 스크립트·백업

- `run_pipe_osr.sh` (신규): ① 분류기 재채점 2개 (GPU 0/1 병렬) → ② `gemini_ing.py --names pipe_osr` → `embed_ing.py` → ③ `eval_quantity_cats.py` 두 분할 → ④ `train.py --eval_only` 영양 4런 → ⑤ `compare_pipeline_cat.py`. 로그 `logs/run_pipe_osr.log`.
- ②에서 **Gemini 신규 호출 0건**: `pipe_osr` 가 만드는 (id, 음식명) 10,024쌍이 모두 기존 `gemini`/`pipe` 조건에서 이미 호출된 쌍이었다 (Gemini 로 넘긴 사진은 Gemini 음식명 그대로라 `gemini` 조건과 같은 쌍). 임베딩 신규도 0건.
- ③~⑤는 기존 조건(`gt`, `gemini`, `pipe`, `masked`)도 같은 모델로 **다시 계산해 덮어쓴다**. 실행 전 원본을 **`backup_before_pipe_osr/`** (167MB) 에 백업:
  - `runs/<split>_quantity/{report_pipe.json, test_logits_pipe.npz}`
  - `runs/<split>_ing_cat/`, `runs/<split>_q4_ing_cat/` 의 `result.json`, `preds_text_*.npz`
  - `runs/cls_osr_<split>_baseline/*` (규칙 변경 전 report·test_pred)
  - `pipeline_compare_ing_cat.json`, `pipeline_compare_rows_ing_cat.csv`, `emb_cache_ing.npz`, `gemini_ing.jsonl`
- 재계산된 기존 조건 값이 백업과 같은지 확인해 재현성을 점검한다 (§12).

### 11.4 실행 방법

```bash
cd /home/leehogwang/food_b/experiments/split_txt_3runs
./run_pipe_osr.sh > logs/run_pipe_osr.log 2>&1
```

---

## 12. 파이프라인 재평가 결과 (G4)

> 재평가 진행 중. 완료 후 추가 예정 (`pipe` vs `pipe_osr`, 두 분할 × actual/q4 × 결합 방식, 기존 조건 재현성 확인).

중간 확인 (양 추정, 122_1_forseen): `pipe_osr` 대분류 정답표 일치 0.986 (`pipe` 0.987), Q acc 0.9954 (동일).

> **업데이트:** 재평가 완료 (`logs/run_pipe_osr.log`, `PIPE_OSR_DONE`). 결과는 아래 12.1~12.6.

### 12.1 재현성 확인

- 기존 조건(`gt`, `gemini`, `pipe`, 양 추정 `masked` 포함)을 같은 모델로 다시 계산한 값과 `backup_before_pipe_osr/` 백업의 **최대 차이 = 0** (영양 예측 `preds_text_*.npz` 8개 런 × 3조건, 양 추정 logits 2분할 × 4조건).
- `pipeline_compare_ing_cat.json` 의 기존 조건 G4 도 전부 동일. → `pipe` vs `pipe_osr` 차이는 **1단계 판정 규칙만의 효과**다.

### 12.2 1단계 음식명 (분할 test)

| split | 조건 | 분류기 음식명 사용률 | 음식명 정확도 | 대분류 정답표 일치 |
|---|---|---|---|---|
| 122_2_forunseen | gemini (Gemini 만) | 0 | 0.3946 | 0.932 |
| 122_2_forunseen | pipe (softmax ≥ 0.7) | 0.5034 | **0.1920** | 0.770 |
| 122_2_forunseen | **pipe_osr** | **0.0085** | **0.3941** | **0.940** |
| 122_1_forseen | gemini | 0 | 0.5535 | 0.911 |
| 122_1_forseen | pipe | 1.0000 | 1.0000 | 0.987 |
| 122_1_forseen | pipe_osr | 0.9932 | 0.9975 | 0.986 |

- unseen: 분류기 음식명은 전부 오답이라, 분류기를 50% 쓰던 `pipe` 는 음식명 정확도가 Gemini 단독의 절반(0.19)이었다. `pipe_osr` 는 분류기를 0.85% 만 써서 Gemini 단독과 거의 같다(0.394).
- seen: `pipe_osr` 는 TPR99 규칙 때문에 0.68% (약 30장)를 Gemini 로 넘기고, 그중 일부가 틀려 정확도 1.000 → 0.9975.

### 12.3 영양 G4 — 영양 모델 직접 예측 (`direct`, `result.json` 의 `text_*`)

| split | 타깃 | gt | gemini | pipe | **pipe_osr** | pipe → pipe_osr |
|---|---|---|---|---|---|---|
| 122_2_forunseen | actual | 45.56 | 45.73 | 49.28 | **45.80** | **−3.48** |
| 122_2_forunseen | q4 | 43.44 | 45.08 | 49.67 | **45.14** | **−4.53** |
| 122_1_forseen | actual | 2.86 | 5.76 | 3.16 | 3.19 | +0.03 |
| 122_1_forseen | q4 | 0.25 | 5.95 | 1.05 | 1.10 | +0.05 |

unseen actual 의 A4: pipe 86.29 → pipe_osr **72.97** (gemini 72.85).

영양소별 G (unseen, actual):

| 조건 | 칼로리 | 지방 | 단백질 | 탄수 | 질량 |
|---|---|---|---|---|---|
| pipe | 43.70 | 57.93 | 54.06 | 41.43 | 38.34 |
| pipe_osr | **39.33** | **55.83** | **50.96** | **37.08** | **36.95** |
| gemini | 39.23 | 55.50 | 51.09 | 37.08 | 37.02 |

### 12.4 영양 G4 — 양 추정 결합 방식별 (`pipeline_compare_ing_cat.json`)

| split / 조건 | direct | pipe_q4 | pipe_gated | pipe_always | pipe_expected_gated | oracle_q |
|---|---|---|---|---|---|---|
| forunseen / pipe | 49.28 | 75.51 | 50.11 | 49.53 | 54.95 | 49.28 |
| **forunseen / pipe_osr** | **45.80** | 70.61 | 45.81 | **45.78** | 50.67 | 45.02 |
| forunseen / gemini | 45.73 | 70.55 | 45.78 | 45.75 | 50.62 | 44.99 |
| forseen / pipe | 3.16 | 46.93 | 1.44 | **1.17** | 1.48 | 1.05 |
| **forseen / pipe_osr** | 3.19 | 46.91 | 1.47 | **1.20** | 1.52 | 1.09 |

### 12.5 해석

1. **unseen 에서 G4 가 3.5~4.5%p 개선됐다** (direct 49.28 → 45.80, q4 49.67 → 45.14, 최선 결합 pipe_always 49.53 → 45.78). 원인은 분명하다: 처음 보는 음식에 엉뚱한 분류기 음식명을 붙이던 50% 가 Gemini 로 넘어가면서 음식명 정확도 0.19 → 0.39, 대분류 일치 0.77 → 0.94 로 회복됐다.
2. **개선의 상한은 "Gemini 단독" 수준이다.** unseen 에서 분류기는 맞힐 수 있는 음식이 없으므로, 최선은 모두 Gemini 로 넘기는 것이고 `pipe_osr` (45.80) 는 그 값(45.73)에 거의 도달했다.
3. **seen 손실은 작다** (+0.03~0.05%p). TPR99 규칙으로 맞는 사진 약 0.7% 를 Gemini 로 넘긴 비용이다. seen 에서만 쓸 거라면 threshold 를 더 느슨하게 할 수 있지만, 실제 운영에서는 사진이 seen 인지 unseen 인지 모르므로 unseen 개선(−3.5%p)이 훨씬 크다.
4. **unseen 의 남은 병목은 1단계가 아니라 영양 모델 자체다.** 정답 텍스트(gt) 를 넣어도 G4 45.56 으로 pipe_osr(45.80) 와 0.24%p 차이뿐이다. 텍스트를 더 정확하게 만들어도 unseen G4 는 거의 줄지 않는다. 남은 개선은 영양 모델의 일반화(음식 단위 K-fold, 이미지 기반 추정, N5k 같은 접시별 실측 데이터)에서 찾아야 한다 (앞선 데이터 누수 점검의 "영양 정답 = (음식, Q) 조회표" 문제와 같은 결론).

### 12.6 권장 (갱신)

- 파이프라인 1단계 판정을 **`pipe_osr` 규칙(기존 CE 분류기 + maha + val 균형 threshold, unknown val 이 없으면 TPR99)으로 교체**한다. 기존 `pipe`(softmax ≥ 0.7) 대비 unseen G4 −3.5~4.5%p, seen +0.05%p 이내.
- 두 분할의 분류기는 서로 다른 모델(78종 / 44종)이라 maha 값의 크기를 그대로 비교할 수 없다 (threshold −2323.0 과 −2058.8 은 서로 옮겨 쓸 수 없는 값). 운영용 분류기 하나에 대해 threshold 를 정하려면, 그 분류기의 학습 음식 일부를 떼어 unknown val 을 만들고(음식 단위 K-fold) 균형 threshold 를 정하는 것이 다음 확인 항목이다.
