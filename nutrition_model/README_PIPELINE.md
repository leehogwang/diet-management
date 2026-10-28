# 영양소 추정 — 현재 가장 성능이 좋은 파이프라인

사진 한 장에서 **칼로리·중량·지방·탄수화물·단백질** 5개를 한 번에 추정합니다.
이 문서는 **그 코드가 어디에 있고, 어떤 데이터를 쓰는지**를 정리한 것입니다.

> **이 폴더(`nutrition_model/`)에 올린 것** — 위 파이프라인을 돌리는 코드 전부(`scripts/`,
> `datasets/`)와 벤치마크 결과 JSON(`outputs/`), 그리고 이 문서입니다.
> **가중치와 데이터는 들어 있지 않습니다**(가중치 174MB×5, 데이터 3GB). 둘 다 연구 서버
> `/home/students/cs/202520851/nutrition_pipeline` 에 있고, 아래 본문의 경로는 그 서버
> 기준입니다. 외부에 바로 돌려볼 수 있는 가중치+테스트데이터 묶음은 9절을 보세요.

---

## 1. 성능

| 타겟 | test 497 | 공식 depth_test 507 |
| :--- | ---: | ---: |
| 칼로리 | 15.87% | 15.91% |
| 중량 | 11.76% | 12.62% |
| 지방 | 23.51% | 23.58% |
| 탄수화물 | 21.29% | 21.77% |
| 단백질 | 19.95% | 21.39% |
| **평균** | **18.48%** | **19.06%** |

5개 모델(5-fold) 예측을 평균한 값입니다. 단일 모델은 497 기준 19.03~19.81%(평균 19.50%)이므로
앙상블이 약 1%p를 가져옵니다.

지표는 **논문식 PMAE = MAE / 정답 평균** 입니다. 샘플별 평균(`mean(|예측−정답|/정답)`)과는
다른 값이 나옵니다. 지방처럼 정답값 자체가 작은 항목에서 둘이 크게 벌어지므로,
논문 수치와 비교할 때는 이 정의만 유효합니다.

비교 기준: DPF 17.8%, Zhao(2024) 17.06% — 둘 다 위에서 내려다본 사진 기준입니다.

---

## 2. 코드 위치

루트: `/home/students/cs/202520851/nutrition_pipeline`

### 실행하는 파일 3개

| 파일 | 역할 |
| :--- | :--- |
| `scripts/run_raw_rgbd_kfold.sh` | 진입점. fold 5개를 순서대로 학습하고 마지막에 앙상블 평가까지 돌립니다 |
| `scripts/train_raw_rgbd_fold.py` | fold 한 개 학습. `--fold 1~5` |
| `scripts/infer_raw_rgbd_kfold_ensemble.py` | 5개 가중치를 불러 497/507 양쪽을 평가하고 JSON으로 저장 |

### 그 3개가 가져다 쓰는 파일

모델 구조와 하이퍼파라미터는 **cutout 계열 스크립트에서 그대로 import**합니다.
입력만 바꾼 실험이었기 때문에 모델 정의를 복사하지 않고 재사용했습니다.
이름에 `cutout`이 들어 있지만 **지금 파이프라인은 세그멘테이션을 쓰지 않습니다.**

| 파일 | 거기서 가져오는 것 |
| :--- | :--- |
| `datasets/nutrition5k_raw_rgbd_dataset.py` | `Nutrition5kRawRGBDDataset` — 입력 4채널을 만드는 곳 |
| `datasets/nutrition5k_rgb_dataset.py` | `load_gt_dict` (정답), `load_split_ids` (공식 split), `TARGETS` |
| `datasets/nutrition5k_cutout_dataset.py` | `load_cutout_split_ids` — 497접시 목록을 정의 |
| `scripts/train_cutout_fold.py` | `build_folds`, `N_FOLDS` — 5분할 |
| `scripts/train_cutout_rgbd.py` | `ResNet101SE_RGBD` — 모델 정의 |
| `scripts/train_cutout_resnet101_se.py` | `SEBlock`, `normalized_mae_loss`, `run_epoch`, `BATCH/LR/LR_DECAY/WORKERS/SEED` |

호출 관계:

```
run_raw_rgbd_kfold.sh
├── train_raw_rgbd_fold.py --fold 1..5
│   ├── datasets/nutrition5k_raw_rgbd_dataset.py   입력 만들기
│   ├── datasets/nutrition5k_rgb_dataset.py        정답 CSV 읽기
│   ├── scripts/train_cutout_fold.py               build_folds()
│   ├── scripts/train_cutout_rgbd.py               ResNet101SE_RGBD
│   └── scripts/train_cutout_resnet101_se.py       손실·에폭 루프·하이퍼파라미터
└── infer_raw_rgbd_kfold_ensemble.py               앙상블 평가
```

### 산출물

| 경로 | 내용 |
| :--- | :--- |
| `runs/raw_rgbd_kfold/fold{1..5}/best.pt` | 최종 가중치. 개당 174MB |
| `runs/raw_rgbd_kfold/fold{1..5}/train_log.csv` | epoch, train_loss, val_loss |
| `runs/raw_rgbd_kfold/fold{K}_run.log` | 학습 로그 |
| `runs/raw_rgbd_kfold/ensemble_run.log` | 앙상블 평가 로그 |
| `outputs/raw_rgbd_kfold_ensemble_results.json` | 위 1절 표의 원본 수치 |

`best.pt`는 `{"epoch", "model_state", "val_loss"}` 딕셔너리입니다.
불러올 때는 `ck["model_state"]`를 꺼내야 합니다.

---

## 3. 사용하는 데이터

### 3-1. 원본 데이터셋 — Nutrition5k (Google)

```
data/nutrition5k  →  /home/students/cs/food_data/nutrition5k_dataset   (심볼릭 링크)
```

팀 공용 경로를 링크로 걸어 둔 것이라 **같은 데이터를 다시 받을 필요가 없습니다.**

실제로 읽는 건 아래 네 군데뿐입니다.

```
data/nutrition5k/
├── imagery/realsense_overhead/<dish_id>/
│   ├── rgb.png              ← 사용. 480×640 RGB, 위에서 내려다본 고정 카메라
│   ├── depth_raw.png        ← 사용. 480×640 uint16, RealSense 실측 depth (mm)
│   └── depth_color.png      ← 안 씀 (사람이 보기 좋게 색을 입힌 버전)
├── metadata/
│   ├── dish_metadata_cafe1.csv   ← 정답
│   └── dish_metadata_cafe2.csv   ← 정답
└── dish_ids/splits/
    ├── rgb_train_ids.txt        4,059접시
    ├── rgb_test_ids.txt           709접시
    └── depth_test_ids.txt         507접시
```

- 접시 폴더 **3,493개**, 용량 **3.0GB**
- `imagery/side_angles/` 는 **199GB**이고 이 파이프라인에서는 쓰지 않습니다
  (옆면 실험용. 8절 참고)

### 3-2. 정답 라벨

`dish_metadata_cafe*.csv`는 헤더가 없고 한 줄에 접시 하나입니다.

```
dish_id, kcal, mass_g, fat_g, carb_g, protein_g, [재료1 id, 이름, 중량, kcal, fat, carb, protein], [재료2 ...]
```

앞 6개 열만 씁니다. 7번째 열부터는 재료별 내역인데 길이가 접시마다 달라서,
`load_gt_dict()`는 쉼표로 쪼갠 뒤 앞 6개만 읽고 나머지는 버립니다.
두 CSV를 합치면 **5,006접시**의 정답이 나옵니다.

### 3-3. 어떤 접시를 쓰는가

세 단계로 걸러집니다.

**1단계 — 공식 train/test split**

코드는 공식 txt 파일을 직접 읽지 않고 `outputs/nutrition5k_depth_features.json` 안의
`split` 필드를 읽습니다(`load_split_ids`). 이 파일은 예전에 depth 특징을 뽑으면서 만든
것이고 split은 공식 `rgb_train/rgb_test`를 그대로 들고 있습니다 — **train 4,059 / test 709**로
공식 파일과 숫자가 같습니다.

**2단계 — 497접시 부분집합**

`load_cutout_split_ids()`가 `data/nutrition5k_seg_labels/segmentation-labels/cutout/`에
파일이 있는 접시만 남깁니다 → **train 2,727 / test 497**.

> **중요** — 세그멘테이션 마스크는 **모델 입력으로 들어가지 않습니다.**
> 접시 목록을 정하는 데만 씁니다. 예전에 cutout(배경 제거) 입력과 원본 입력을
> 비교하던 실험이 있었고, 그때와 같은 접시 집합을 유지해야 수치를 비교할 수 있어서
> 이 필터를 그대로 두었습니다. 원본 입력이 더 좋게 나와서 세그멘테이션은 입력에서
> 빠졌지만, 접시 목록 조건은 비교 기준선 때문에 남아 있는 것입니다.

**3단계 — 5분할**

`build_folds()`가 train 2,727개를 정렬한 뒤 `RandomState(42)`로 섞어 5등분합니다.
fold별로 **학습 2,181~2,182 / 검증 545~546**입니다. seed가 고정이라 어느 프로세스에서
돌려도 같은 분할이 나옵니다.

**평가셋 두 개**

| 이름 | 개수 | 정의 |
| :--- | ---: | :--- |
| test 497 | 497 | 공식 test ∩ cutout 존재. 기존 실험들과 비교하는 선 |
| 공식 depth_test | 507 | `load_split_ids("test")` 중 이미지가 있는 접시. 논문과 맞추는 선 |

### 3-4. 쓰지 않는 데이터 (혼동하기 쉬움)

| 경로 | 용량 | 설명 |
| :--- | ---: | :--- |
| `data/nutrition5k_pred_depth/` | 2.1GB | Depth Anything V2로 예측한 depth. **별도 실험용** |
| `data/nutrition5k_metric_depth/` | — | 미터 단위로 변환한 depth. 별도 실험용 |
| `data/nutrition5k/imagery/side_angles/` | 199GB | 옆면 영상 프레임. 8절 |
| `data/nutrition5k_seg_labels/` | 452MB | 세그멘테이션 라벨(`cutout`/`png`/`json`/`pseudo`). `cutout`의 **파일 이름만** 접시 목록 정의에 쓰고 이미지 내용은 안 씀 |
| `data/nutritionverse_real/`, `data/nv_features/` | — | 다른 데이터셋(NutritionVerse) 실험용 |
| `data/kfood122/`, `data/food_meta*.json` | — | 한식 갈래용 |

예측 depth 쪽은 **센서 없이 사진 한 장만으로 쓰는 경로**로 따로 돌려 둔 것입니다.
`runs/pred_rgbd_kfold/` 에 가중치가 있고 성능은 18.87%(497) / 19.46%(507) —
센서 depth 대비 손실이 0.4%p뿐입니다. 코드는 `--pred` 플래그 하나로 전환됩니다.

---

## 4. 입력 전처리

```
rgb.png (480×640×3)          depth_raw.png (480×640, uint16)
      │                             │
      │                      0인 화소(센서 구멍)를 3754로 대체
      │                             │
      │                      (값 − 3754) / 941        ← 전역 표준화
      │                             │
      └──────────── 4채널로 결합 ────┘
                     │
            resize 256×256  (antialias)
                     │
       학습: RandomCrop 224 + 좌우 반전 50%
       평가: CenterCrop 224
                     │
       RGB 3채널만 ImageNet 통계로 정규화
       (mean .485/.456/.406, std .229/.224/.225)
                     │
            (B, 4, 224, 224)
```

**depth 통계 3754 / 941** 은 학습 접시 300개 표본에서 구한 값입니다(구멍 제외).
`datasets/nutrition5k_raw_rgbd_dataset.py`에 상수로 박혀 있고 **바꾸면 가중치가 안 맞습니다.**

**depth 채널은 정규화를 따로 하지 않습니다.** 위에서 이미 표준화를 했기 때문입니다.
`TF.normalize`가 `x[:3]`에만 걸려 있는 이유입니다.

**순서가 중요합니다.** 먼저 4채널로 합친 다음 resize/crop/flip을 하므로 RGB와 depth에
완전히 같은 기하 변환이 걸립니다. 따로 처리하면 두 채널이 화소 단위로 어긋납니다.

**depth가 깨진 접시** — 파일이 없거나 RGB와 크기가 다르면 전체를 중립값 3754로 채웁니다.
즉 depth 채널이 0(표준화 후)인 상태로 들어가고 학습은 그대로 진행됩니다.

### 캐시 옵션

`Nutrition5kRawRGBDDataset(..., cache=True)` 를 주면 resize까지 끝낸 256×256 4채널을
float16으로 메모리에 들고 있습니다. PNG 디코딩이 병목이라 에폭당 41초 → 25초가 됩니다.
3,224장 전부라도 약 1.7GB이고 DataLoader 워커는 fork로 공유합니다.
crop·flip·정규화는 캐시 뒤에 적용되므로 증강 효과는 달라지지 않습니다
(수치 확인: 캐시 유무 최대 차이 0.001 — float16 반올림).

**k-fold 스크립트는 캐시를 쓰지 않습니다.** 가중치를 만든 당시 코드에 없던 옵션입니다.

---

## 5. 모델

정의: `scripts/train_cutout_rgbd.py` 의 `ResNet101SE_RGBD`

```
입력 (B, 4, 224, 224)
  │
  ResNet101 (ImageNet IMAGENET1K_V2 사전학습, conv1만 3→4채널로 확장)
  │                                    (B, 2048, 7, 7)
  SE Block  2048 → 128 → 2048 → sigmoid, 채널별 곱
  │                                    (B, 2048, 7, 7)
  Global Average Pooling               (B, 2048)
  │
  Linear 2048→1024 → ReLU → Dropout 0.3
  Linear 1024→256  → ReLU → Dropout 0.3
  Linear 256→5
  │
  clamp(min=0)                         음수 영양값은 없으므로
  │
출력 (B, 5) = [kcal, mass_g, fat_g, carb_g, protein_g]
```

### conv1 확장

ImageNet ResNet의 `conv1` 가중치는 RGB 3채널을 전제로 `(64, 3, 7, 7)` 모양입니다.
depth를 붙이려면 `(64, 4, 7, 7)`이 필요합니다.

```python
new.weight[:, :3] = old.weight                        # RGB는 그대로 복사
new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)   # depth는 RGB 세 채널 평균
```

4번째 채널을 무작위로 두면 학습 초반에 사전학습 특징이 흔들립니다. RGB 평균으로 채우면
depth가 "밝기 비슷한 또 하나의 채널"로 시작해서 안정적입니다.
`bias=False`라 채널 수만 늘려도 수식이 깨지지 않습니다.

### 손실 — 정규화 MAE

```python
loss = Σ_t ( Σ_i |예측_it − 정답_it| / Σ_i 정답_it )
```

타겟마다 **그 타겟 정답의 합**으로 나눠 더합니다. 칼로리(수백)와 지방(한 자리)을 그냥
MAE로 더하면 칼로리만 학습됩니다. 이렇게 하면 다섯 타겟이 비슷한 비중을 갖습니다.

이 형태가 논문식 PMAE와 같은 식이라 **`val_loss / 5` 가 곧 논문 지표**입니다.
학습 로그만 보고도 성능을 읽을 수 있습니다.

---

## 6. 학습 설정

| 항목 | 값 | 정의 위치 |
| :--- | :--- | :--- |
| 손실 | 정규화 MAE | `train_cutout_resnet101_se.py: normalized_mae_loss` |
| 최적화 | Adam, lr 1e-4, weight decay 1e-4 | `train_raw_rgbd_fold.py` |
| 스케줄러 | ExponentialLR γ=0.99 (에폭마다) | 같음 |
| batch | 학습 8 / 평가 32 | `BATCH = 8` |
| 최대 에폭 | 300 | `--epochs` |
| early stop | patience 40, 최소 150에폭 | `--patience`, `--min-epochs` |
| seed | 42 | `train_cutout_resnet101_se.py: SEED` |
| DataLoader workers | 4 | `WORKERS` |
| 모델 선택 | val_loss 최소 에폭 | `best.pt` 덮어쓰기 |

실제 소요: fold당 **1시간 40분 ~ 2시간 35분**, 5개 전부 **약 10시간 45분**
(NVIDIA L4 24GB 한 장, 2026-07-15 23:02 → 07-16 09:47).
fold별 종료 에폭은 194~300입니다.

### 실행 환경

- Python 3.11.14, torch 2.10.0+cu128, torchvision 0.25.0+cu128, CUDA 12.8
- 가상환경: `nutrition_pipeline/.venv` — **시스템 python이 아니라 이걸 써야 합니다**
- GPU: L4 24GB × 4장. **팀 공용이므로 한 번에 한 장만** 점유합니다

---

## 7. 재현

```bash
cd /home/students/cs/202520851/nutrition_pipeline
df -h /home                    # 디스크 먼저 확인 (여유가 빡빡합니다)
nvidia-smi                     # 빈 GPU 번호 확인
```

처음부터 다시 학습:

```bash
bash scripts/run_raw_rgbd_kfold.sh
```

스크립트 안에 `CUDA_VISIBLE_DEVICES=1`이 박혀 있으니 다른 GPU를 쓰려면 그 줄을 고칩니다.

이미 있는 가중치로 평가만:

```bash
cd /home/students/cs/202520851/nutrition_pipeline
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD ./.venv/bin/python3 scripts/infer_raw_rgbd_kfold_ensemble.py
```

**18.48%(497) / 19.06%(507)** 이 나오면 정상입니다.
앙상블 평가는 결정적이라 매번 같은 값이 나옵니다.

예측 depth 쪽을 돌리려면 두 스크립트에 `--pred`를 붙입니다(`bash scripts/run_pred_rgbd_kfold.sh`).

> 재학습은 결과가 완전히 같지 않습니다. seed를 고정해도 cuDNN 커널 선택과 DataLoader
> 워커 순서가 비결정적이어서, 단일 모델 기준 **런마다 표준편차 약 0.31%p**가 있습니다.
> 그래서 실험 간 차이는 **0.6%p 넘을 때만** 의미가 있다고 봅니다(타겟별로는 1.4%p).

---

## 8. 알아두어야 할 한계

### 촬영 시점이 바뀌면 크게 무너집니다

같은 데이터셋의 **옆면** 사진으로 평가하면:

| | 위에서 | 옆면 |
| :--- | ---: | ---: |
| 평균 | 18.80% | **68.79%** |
| 중량 | 11.90% | **75.28%** |

원인은 모델이 **음식이 사진에서 차지하는 면적**을 분량의 단서로 쓰고 있기 때문입니다.
카메라가 고정이면 크게 보이는 음식이 실제로도 양이 많아 이 단서가 잘 맞지만,
시점이 바뀌면 그 대응이 깨집니다. 중량이 가장 크게 무너지는 게 그 증거입니다.

### 학습에 옆면을 넣으면 고쳐집니다

접시당 옆면 4장(카메라당 1장)을 학습에 섞어 다시 학습했습니다.

| 평가 | 위만 학습 | 옆면 포함 학습 |
| :--- | ---: | ---: |
| 위 사진 test | **19.55%** | 21.57% |
| 옆면 test | 68.79% | **23.66%** |
| └ 중량만 | 75.28% | **16.36%** |

옆면이 2.9배 좋아지고 중량은 4.6배 좋아집니다. 대신 위 사진에서 2.02%p를 잃습니다.
즉 **시점 붕괴는 과제 자체의 난이도가 아니라 학습 분포의 문제**입니다.
단일 런이라 위 사진 쪽 2.02%p는 시드 반복으로 다시 확인할 필요가 있습니다.

관련 코드: `datasets/nutrition5k_side_dataset.py`, `scripts/train_raw_rgbd.py --side`,
`scripts/eval_side_angle.py`. 가중치는 `runs/raw_rgbd_resnet101_se_side1/`.

### 그 외

- **휴대폰으로 자유롭게 찍는 상황에는 그대로 쓸 수 없습니다.** 18.48%는 고정된 촬영
  조건에서의 값입니다.
- **다른 환경으로 옮기면 배율이 무너집니다.** 예측의 순서(상관)는 살아남는데 배율이
  압축됩니다. 대상 환경의 접시 10~20개를 저울에 재서 배율 하나를 잡아주면 크게 복구됩니다.
- **탄수화물이 가장 약합니다.** DB 잡음이나 조회 실패 때문이 아니고, 사진에서 얻을 수 있는
  신호 자체가 약한 데다(상관 0.181) 정답 분포의 꼬리가 두껍습니다(변동계수 1.51,
  상위 10% 접시가 전체 오차의 56.9%). 진단 코드는 `scripts/diag_carb.py`입니다.

---

## 9. 공유용 패키지

외부에 줄 때는 저장소 전체가 아니라 아래 폴더를 쓰면 됩니다.

```
/home/students/cs/202520851/n5k_model_share/        (압축본: n5k_model_share.tar.gz, 1.1GB)
├── predict.py              모델 정의 + 전처리 + 평가가 전부 한 파일에
├── weights/fold1~5.pt      5-fold 가중치
├── data/test/              497접시 (rgb.png + depth_raw.png)
├── data/labels_test.json   정답
├── figures/                파이프라인 상세도 (PNG/SVG/drawio)
└── README.md
```

이 폴더만으로 돌아갑니다. 저장소나 Nutrition5k 원본을 따로 받을 필요가 없습니다.
파이프라인 그림은 `figures/pipeline_detail.png` 또는 저장소의
`outputs/pipeline_detail.drawio` 에 있습니다.
