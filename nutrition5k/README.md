# Nutrition5k 영양 추정 파이프라인

> **최고 성적 파이프라인 = "side 3체인"** (2026-10-02 최종 챔피언 확정)
> 실행 진입점: `scripts/task_batch/orchestrate_side_full.sh`

---

## A. 최고 성적 파이프라인 — side 3체인

### A-0. 사용 데이터

#### 원본 데이터셋 — Nutrition5k (Google Research, 2021)

- 위치: `/home/students/cs/202520906/nutrition5k` (실제 경로 `/home/students/cs/food_data/nutrition5k_dataset` 의 심볼릭 링크). 공식 설명: https://github.com/google-research-datasets/Nutrition5k
- 구내식당 두 곳(cafe1 4,768접시 / cafe2 238접시, 총 **5,006접시**)에서 접시를 저울에 올린 채 촬영하고, 재료를 하나씩 올릴 때마다 무게를 쟀다. 그래서 **접시 단위 영양 정답과 재료별 그램 정답**이 둘 다 있다.
- 이 파이프라인이 학습·평가에 쓰는 접시는 **전부 cafe1** 이다(train 2,485 / val 269 / test 507 모두 cafe1, 실측). 공식 논문도 cafe1 로만 학습했다.

```
nutrition5k/
├── metadata/
│   ├── dish_metadata_cafe1.csv      # 접시 4,768행 — 영양 정답 + 재료별 그램 (아래 형식)
│   ├── dish_metadata_cafe2.csv      # 접시 238행
│   └── ingredients_metadata.csv     # 재료 554종의 1g당 영양값 (ingr,id,cal/g,fat(g),carb(g),protein(g))
├── dish_ids/
│   ├── dish_ids_{all,cafe1,cafe2}.txt
│   └── splits/
│       ├── rgb_train_ids.txt  (4,059)  rgb_test_ids.txt  (709)   ← 이 저장소의 분할 기준
│       └── depth_train_ids.txt (2,758) depth_test_ids.txt (507)  ← 깊이가 있는 접시만 모은 부분집합
└── imagery/
    ├── realsense_overhead/<dish_id>/       # 3,493접시 — 위에서 내려다본 RealSense RGB-D 한 장
    │   ├── rgb.png          640×480 RGB uint8
    │   ├── depth_raw.png    640×480 16비트 단일채널(I;16), 0 = 측정 실패
    │   └── depth_color.png  깊이를 색으로 칠한 시각화본 (사용 안 함)
    └── side_angles/<dish_id>/              # 5,006접시 — 측면 카메라 4대(A~D)가 회전판을 찍은 영상
        ├── camera_{A,B,C,D}.h264
        └── frames_sampled5/camera_X_frame_NNN.jpeg   # 영상에서 추출한 1920×1080 JPEG
```

**`dish_metadata_cafe*.csv` 한 행의 형식** (헤더 없음, 쉼표 구분):

```
dish_id, 총칼로리(kcal), 총중량(g), 총지방(g), 총탄수화물(g), 총단백질(g),
  [재료ID, 재료명, 그램, 칼로리, 지방, 탄수, 단백] × 재료 수만큼 반복
예) dish_1561662216,300.79,193.0,12.39,28.22,18.63, ingr_0000000508,soy sauce,3.40,1.80,0.02,0.17,0.28, ingr_0000000122,garlic,2.12,...
```

- 앞 6칸 → 영양 정답 4개(칼로리·지방·탄수·단백)와 중량. **중량은 학습 타깃에도 입력에도 쓰지 않는다**(초기 실험에서 빼는 것이 최적으로 정리됨). 중량 구간별 성적표를 만들 때만 쓴다.
- 7칸씩 반복되는 재료 블록 → `train_grams.parse_grams()` 가 `{dish_id: {재료명: 그램}}` 으로 파싱한다(같은 재료가 두 번 나오면 그램을 합침). 이것이 보조헤드의 정답(A-5)이다.
- `ingredients_metadata.csv` 는 이 파이프라인에서 쓰지 않는다.

**깊이 단위 주의**: `depth_raw.png` 값은 **1 = 0.1mm (10,000 = 1m)** 로 보인다. test 30접시의 유효 픽셀 중앙값이 3,366~4,067 이고, 이를 0.1mm 로 읽으면 카메라~접시 거리 약 35cm 로 촬영 장치와 맞는다(mm 로 읽으면 3.5m 가 되어 맞지 않는다). 코드 주석 일부(`train_depth.py`)에는 "mm" 라고 적혀 있지만, 학습 입력은 이미지마다 z-점수화하므로(A-7) 단위가 결과에 영향을 주지 않는다. 단위가 실제로 의미를 갖는 곳은 크롭 검출의 문턱값 30(≈3mm) 하나다.

#### 이 저장소에서 만든 파생 데이터

| 파일 | 만드는 코드 | 내용 |
|------|------|------|
| `outputs/manifests/{train,val,test}.jsonl` | `scripts/prepare_splits.py` | 분할 정의 + 접시별 영양 정답. 한 줄 = side 프레임 한 장(접시당 최대 4장). 챔피언은 이 파일에서 **접시 ID와 정답만** 읽고(`dish_targets`/`dish_targets5`, 접시당 첫 줄), 이미지 경로 칸은 쓰지 않는다 |
| `outputs/task_runs/task4_final/crop_bbox.json` | `precompute_crop_bbox.py` | overhead 3,261접시의 접시 크롭 박스 `[x0,y0,x1,y1]` (A-7) |
| `outputs/task_runs/task_side/side_frames.json` | `index_side_frames.py` | 접시 → `{split, n_frames, frames[], has_overhead}` (단계 0 용) |

**분할 규칙** (`prepare_splits.py`, seed 42):
- test = 공식 `rgb_test_ids.txt` 709접시 그대로(고정).
- 공식 `rgb_train_ids.txt` 를 칼로리·중량·재료 수 각각 5분위로 층을 나눠, 층마다 10% 를 val 로 뽑는다 → train 3,642 / val 405접시. 분할은 **접시 단위**라 한 접시의 사진이 두 split 에 동시에 들어가지 않는다.
- ⚠ 다만 같은 음식을 1분 안에 연달아 찍은 "거의 같은 접시"는 서로 다른 접시 ID 를 가지므로 접시 단위 분할로는 막히지 않는다. 그 결과 val 의 약 90% 가 train 에 근접 중복을 갖는다(test 는 약 5%). val 이 test 보다 훨씬 쉬운 이유다(A-10).

#### 단계별로 실제 쓰는 데이터

| | 단계 0 (side 사전학습) | 단계 1·2 (overhead) | 평가 |
|------|------|------|------|
| 이미지 | `side_angles/.../frames_sampled5/*.jpeg` (RGB만, 깊이 칸은 0) | `realsense_overhead/<dish>/rgb.png` + `depth_raw.png` | 단계 1·2 와 동일 |
| 접시 선택 | train split 중 side 프레임이 있는 3,535접시 → 학습 3,253 + 내부 홀드아웃 282 | rgb·depth 파일이 둘 다 비어 있지 않은 접시 | test 507 |
| 접시 수 | 3,253 (그중 overhead 없던 접시 ≈1,060) | train **2,485** / val **269** | **507** (= 공식 `depth_test_ids.txt` 507접시와 정확히 같은 집합) |
| 정답 | 4영양소 + 재료 그램 | 4영양소 + 재료 그램 | 4영양소 |

- side 인덱스 전체: 4,606접시가 프레임을 보유한다. split 별로 train 3,535(overhead 있음 2,379 / 없음 1,156), val 395, test 676. **val·test 접시의 side 프레임은 학습에 한 장도 쓰지 않는다.** 접시당 프레임은 중앙값 32장(평균 58장), 카메라 A~D 가 고르게 섞여 있다.
- 4영양소 정답 분포 (overhead train 2,485접시): 평균 칼로리 254.4 kcal · 지방 12.7 g · 탄수 19.3 g · 단백 18.1 g · 중량 217.5 g, 중앙값 207 kcal. 최대값은 칼로리 3,943 kcal · 중량 3,051 g 으로 꼬리가 길다. test 평균은 255.5 kcal · 12.8 / 19.7 / 17.5 g 으로 train 과 거의 같다.
- 재료 어휘: train 접시에 나오는 재료는 196종이고, 그중 등장 빈도 상위 80종만 보조헤드 타깃으로 쓴다.

### A-1. 성적 (test 507접시, 공식 test split)

| 지표 | 3시드 앙상블 | 3시드 평균±SD | 시드 42 / 1 / 2 |
|------|------|------|------|
| 평균 PMA (4영양소) | **20.75%** | 21.27±0.27% | 21.54 / 21.28 / 21.00 |
| 칼로리 PMA | **15.24%** | 15.47±0.28% | 15.77 / 15.42 / 15.23 |
| 지방 PMA | 22.41% | 22.94±0.45% | 22.84 / 23.43 / 22.55 |
| 탄수화물 PMA | 22.62% | 23.20±0.47% | 23.74 / 22.84 / 23.03 |
| 단백질 PMA | 22.74% | 23.47±0.32% | 23.81 / 23.42 / 23.18 |
| 칼로리 RMSE | 64.5 kcal | 65.4±2.2 | 67.6 / 65.2 / 63.3 |

- PMA = `100 · mean|예측−정답| / mean(정답)` (영양소별로 계산, 평균 PMA 는 4영양소의 산술평균). 낮을수록 좋다.
- 앙상블 = 세 시드 모델의 test 예측값을 단순 평균한 것.
- 원본 표: `results/final/FINAL_performance_table.md` (중량 구간별 표 포함), 예측값: `results/final/FINAL_champion_preds.npz`.

### A-2. 한눈에 보는 구조

```
                 ┌──────────── 단계 0 ────────────┐   ┌──────── 단계 1 ────────┐   ┌──────── 단계 2 ────────┐
ImageNet ConvNeXt-S ─▶ side 사진 사전학습 (깊이=0) ─▶ overhead RGB-D 크롭 적응 ─▶ 저학습률 미세조정(nRMSE 기준)
                 │ train split side 3,253접시       │   │ overhead 2,485접시      │   │ overhead 2,485접시      │
                 │ lr 1e-4 · 상한200 · pat12        │   │ lr 1e-4 · 상한200 · pat12│  │ lr 2e-5 · 상한150 · pat10│
                 │ 조기종료 = 홀드아웃 mean_PMA     │   │ 조기종료 = val mean_PMA │   │ 조기종료 = val mean_nRMSE│
                 └──────────────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
                       train_side_pretrain.py                 train_strong.py                  train_b8.py
```

- 이 체인을 **시드 42 / 1 / 2 로 단계 0 부터 따로** 세 번 돌린다(= 3체인). 시드를 단계 0 에서부터 가르는 이유: 체크포인트를 공유하면 시드 간 차이가 작아 보여 "개선이 일관됐다"는 착시가 생긴다(experiment_log 의 함정 7번).
- 세 단계 모두 **모델 구조는 동일**하다(ConvNeXt-Small 4채널 + 영양소 헤드 + 재료 그램 보조헤드). 단계마다 바뀌는 것은 *데이터·학습률·조기종료 기준* 뿐이다.
- 이전 챔피언(mx_rmse, 평균 PMA 22.56%)과의 차이는 **단계 0 을 끼워 넣은 것 하나**다. overhead RGB-D 가 없어 학습에 못 쓰던 접시 약 1,060장을 side 사진으로 끌어온다.

### A-3. 실행 방법

```bash
cd /home/students/cs/202520906/nutrition5k_pipeline

# (사전 준비, 한 번만) — 이미 만들어져 있으면 생략
python -m scripts.prepare_splits --dataset-dir /home/students/cs/202520906/nutrition5k \
    --output-dir outputs --val-ratio 0.1 --max-side-frames 4          # outputs/manifests/{train,val,test}.jsonl
python scripts/task_batch/precompute_crop_bbox.py                     # outputs/task_runs/task4_final/crop_bbox.json
python scripts/task_batch/index_side_frames.py                        # outputs/task_runs/task_side/side_frames.json

# 학습: 시드 42/1/2 를 GPU 0/1/2 에 하나씩 병렬로 (체인당 GPU 1장)
bash scripts/task_batch/orchestrate_side_full.sh

# 중단 (pkill -f 는 쓰지 말 것 — 자기 자신과 매칭돼 셸이 죽는다)
kill $(cat outputs/task_runs/task_side/logs/sidefull_pids.txt)

# 최종 성능표: 3개 체크포인트로 test 를 다시 추론해 시드별·앙상블·중량구간 수치 산출
CUDA_VISIBLE_DEVICES=3 python -m scripts.task_batch.final_paper_table
```

⚠ 공용 서버 주의: 현재 csgpu 에서는 **GPU 3 한 장만** 쓸 수 있다. 오케스트레이터는 GPU 0/1/2 를 하드코딩하고 있으므로, 지금 다시 돌리려면 `chain "$s" "$i"` 의 GPU 번호를 3 으로 바꾸고 세 체인을 순차로(`&` 제거) 돌려야 한다.

오케스트레이터는 각 단계의 결과 파일(`pretrain_metrics.json` / `test_metrics.json`)이 이미 있으면 그 단계를 **SKIP** 한다. 중간에 죽어도 다시 실행하면 끝난 단계는 건너뛰고 이어서 진행한다. 반대로 처음부터 다시 돌리려면 해당 런 폴더를 지워야 한다.

### A-4. 파일 지도

| 파일 | 역할 |
|------|------|
| `scripts/task_batch/orchestrate_side_full.sh` | **진입점.** 시드별로 단계 0→1→2 를 순서대로 호출 |
| `scripts/task_batch/train_side_pretrain.py` | 단계 0. side 프레임 데이터셋 `SideDS` + 사전학습 루프 |
| `scripts/task_batch/train_strong.py` | 단계 1. 모델 `StrongNet`·백본 수술 `build_backbone`·nRMSE 계산 `_nrmse` 정의 |
| `scripts/task_batch/train_b8.py` | 단계 2. 모델 `B8Net`(기본 옵션이면 `StrongNet` 과 동일 구조)·체크포인트 이식 `init_from_champion` |
| `scripts/task_batch/train_grams_crop.py` | 접시 크롭 데이터셋 `CropDS`, `crop_rgb`/`crop_depth` (단계 1·2 의 입력 전처리) |
| `scripts/task_batch/train_grams.py` | `parse_grams`(재료별 그램 라벨 파싱), `usable`(접시 목록), `VOCAB_K=80` |
| `scripts/task_batch/train_depth.py` | 경로 상수 `OV`, `dish_targets`, `pma`, `MANIFEST_DIR` |
| `scripts/task_batch/task2_common.py` | `dish_targets5`(4영양소+중량), `metrics`(PMA/RMSE/nRMSE 동시 계산) |
| `scripts/task_batch/preprocess_plate_roi.py` · `precompute_crop_bbox.py` | 접시 검출 → 크롭 박스 캐시 생성 |
| `scripts/task_batch/index_side_frames.py` | side 프레임 인덱스 생성 (접시 → 프레임 목록, split, overhead 보유 여부) |
| `scripts/task_batch/score_hybrid.py` · `final_paper_table.py` | 체크포인트 재추론·3시드 앙상블·최종 성능표 |
| `nutrition5k_cbam/utils.py` | `seed_everything`, `save_json` |

### A-5. 모델 — ConvNeXt-Small 4채널 (ResNet50 대비 무엇이 다른가)

세 단계가 공유하는 모델이다. 단계 0·1 은 `train_strong.StrongNet`, 단계 2 는 `train_b8.B8Net`(기본값 `pool=avg, spatial=none, decode=head`)으로 만드는데, 두 클래스는 이 설정에서 **파라미터 이름·모양이 완전히 같다**(단계 2 로그: `copied=350 expanded=0 skipped=0`).

**입력** `(B, 4, 384, 384)` = RGB 3채널(ImageNet 평균/표준편차로 정규화) + 깊이 1채널.

**백본 텐서 흐름** (배치 1, 실측):

| 블록 (`backbone.features[i]`) | 출력 크기 | 파라미터 | ResNet50@224 에서 대응하는 것 |
|------|------|------|------|
| `[0]` stem: Conv2d(**4**→96, k=4, s=4) + LayerNorm | (96, 96, 96) | 6,432 | conv1 7×7 s2 + maxpool → (64, 56, 56) |
| `[1]` stage1 (블록 3개) | (96, 96, 96) | 237,888 | layer1 → (256, 56, 56) |
| `[2]` 다운샘플 | (192, 48, 48) | 74,112 | — |
| `[3]` stage2 (블록 3개) | (192, 48, 48) | 918,144 | layer2 → (512, 28, 28) |
| `[4]` 다운샘플 | (384, 24, 24) | 295,680 | — |
| `[5]` stage3 (블록 27개) | (384, 24, 24) | 32,451,840 | layer3 → (1024, 14, 14) |
| `[6]` 다운샘플 | (768, 12, 12) | 1,181,184 | — |
| `[7]` stage4 (블록 3개) | (768, **12, 12**) | 14,289,408 | layer4 → (2048, 7, 7) |
| 전역평균풀링 + LayerNorm2d + Flatten | **768** | 1,536 | avgpool → 2048 |

- 마지막 특징맵이 12×12 = **144칸**이다(384 ÷ 32). 전역평균풀링이 이 144칸을 균등 평균해 768차원 벡터 하나로 만든다. 배터리 8 에서 Max/어텐션 풀링으로 바꿔 봤지만 개선이 없었고, 어텐션 가중치도 균등에서 거의 움직이지 않아 평균풀링을 유지했다.
- ResNet50 은 stem 이 7×7 stride 2 합성곱 + maxpool(해상도 1/4)인데, ConvNeXt 는 4×4 stride 4 합성곱 하나로 바로 1/4 로 줄인다(패치화). 블록 내부도 ResNet 의 1×1→3×3→1×1 병목 대신 7×7 depthwise 합성곱 → LayerNorm → 1×1 확장(4배) → GELU → 1×1 축소 구조다. BatchNorm 대신 LayerNorm 을 쓰므로 배치 통계(running stat)가 없다.

**4채널 stem 수술** (`train_strong.build_backbone`, `train_b8.build_convnext4ch`):
ImageNet 가중치의 stem 은 Conv2d(3→96). 이를 Conv2d(4→96) 으로 교체하면서
`new.weight[:, :3] = 기존 RGB 가중치`, `new.weight[:, 3] = RGB 가중치의 채널 평균`, bias 그대로 복사한다. ResNet50 시절 conv1 을 4채널로 넓힌 방식과 같다.
분류기 `classifier[2]`(Linear 768→1000)는 `Identity` 로 바꿔 LayerNorm2d + Flatten 만 남긴다.

**헤드 두 개** (백본 768차원 특징 `f` 를 공유):

```
nutr_head : Dropout(0.2) → Linear(768,512) → ReLU → Dropout(0.2) → Linear(512,4) → Softplus   # [칼로리, 지방, 탄수, 단백] 원단위
aux_head  : Dropout(0.2) → Linear(768,512) → ReLU → Linear(512,80)                            # 재료 80종의 표준화 log1p(그램)
```

- `Softplus` 로 영양소 예측이 항상 0 이상이 되게 한다. 타깃은 로그 변환 없이 원단위(kcal, g) 그대로다.
- `aux_head` 는 **학습 때만 쓰는 보조 과제**다. 추론·평가에서는 `nutr_head` 출력만 쓴다. 재료 그램은 입력이 아니라 정답 라벨로만 쓰이므로 test 시점에 정보가 새지 않는다.
- 파라미터: 백본 49,456,224 + 영양소 헤드 395,780 + 보조헤드 434,768 = **총 50,286,772, 전부 학습(동결 0%)**.

**손실** (세 단계 공통):

```
loss = SmoothL1(nutr_pred, nutr_true) + 30 × SmoothL1(aux_pred, z_grams)
z_grams = (log1p(재료별 그램) − μ) / σ        # μ, σ 는 해당 단계의 학습 접시로만 계산
```

- 재료 어휘 = 학습 접시에 등장한 빈도 상위 **80종**(`VOCAB_K`). 없는 재료는 0g → log1p(0)=0 에서 표준화.
- 보조 손실 가중치 λ=30. 보조 손실 값 자체가 0.1 안팎으로 작아 영양소 손실(수 단위)과 크기를 맞추기 위한 값이다.
- 옵티마이저 AdamW(weight_decay 1e-4), AMP(autocast + GradScaler), 배치 16, 학습률 스케줄러 없음(상수).

### A-6. 단계 0 — side 사진 사전학습 (`train_side_pretrain.py`)

**목적**: Nutrition5k 에는 영양 라벨은 있지만 overhead RGB-D 가 없어서 단계 1·2 에서는 못 쓰는 접시가 있다. 이 접시들에도 측면 카메라 영상(side_angles)은 있으므로, 측면 사진으로 영양 회귀를 먼저 학습해 백본을 "음식 사진 → 영양" 과제에 미리 적응시킨다.

**데이터 구성** (`main()` 앞부분):
1. `side_frames.json`(인덱스)에서 **split 이 train 인 접시만** 고른다 → 3,535접시. val/test 접시는 인덱스에 있어도 제외하고, `assert` 로 다시 확인한다(누수 차단).
2. 시드로 섞은 뒤 8%(282접시)를 **내부 홀드아웃**으로 떼어낸다. 진짜 val/test 는 이 단계에서 전혀 보지 않는다.
3. 나머지 **3,253접시**로 학습. 그중 overhead 가 없어 원래 못 쓰던 접시가 1,060(시드 42) / 1,074(시드 1) / 1,067(시드 2)장이다(홀드아웃이 시드마다 달라 조금씩 다르다). 프레임은 총 187,854장, 접시당 평균 58장.

**`SideDS` (접시 1개 = 샘플 1개)**:
- 학습 시: 매 호출마다 그 접시의 프레임 중 **한 장을 무작위**로 뽑는다. 카메라 4대 × 회전 프레임이라 자연스러운 시점 증강이 된다. 50% 확률로 좌우 반전.
- 평가 시: 항상 **가운데 프레임** 하나(결정적).
- 이미지를 384×384 로 리사이즈(크롭 없음) → ImageNet 정규화 → 4번째 채널은 **전부 0**(측면 사진에는 깊이가 없다). 모델의 4채널 stem 형태를 맞추기 위한 것이며, 단계 1 에서 이 채널에 실제 깊이가 들어가며 학습된다.
- 이미지를 못 읽으면 0 이미지로 대체(학습이 죽지 않게).

**학습 설정**: lr 1e-4, 배치 16, 에폭 상한 200, patience 12, 조기종료·체크포인트 선택 기준 = **홀드아웃 mean_PMA**. 상한 60 이던 8/22 파일럿은 58에폭에서 상한에 잘렸기 때문에 200 으로 늘렸다. 세 시드 모두 상한 전에 조기종료했다.

| 시드 | best epoch / 실행 에폭 | 홀드아웃 mean_PMA |
|------|------|------|
| 42 | 121 / 133 | 22.25% |
| 1 | 49 / 61 | 27.69% |
| 2 | 56 / 68 | 25.03% |

단계 0 성적은 최종 성적을 예측하지 못한다(시드 1 이 단계 0 에서 가장 나빴지만 최종은 중간). 단계 0 은 출발 가중치를 만드는 과정으로만 본다.

**출력**: `outputs/task_runs/task_side/runs/side_pre200_seed{S}/` — `checkpoints/best.pt`, `training_history.json`, `pretrain_metrics.json`(best_epoch, hit_epoch_cap 등).

### A-7. 단계 1 — overhead RGB-D 접시 크롭 적응 (`train_strong.py`)

호출 인자: `--backbone convnext_small --resolution 384 --crop 1 --lam 30 --batch-size 16 --lr 1e-4 --epochs 200 --patience 12 --init-from <단계0 best.pt>`

**데이터** (`usable(split)`): manifest 의 접시 중 `realsense_overhead/<dish>/rgb.png` 와 `depth_raw.png` 가 둘 다 비어 있지 않은 것만 → **train 2,485 / val 269 / test 507**. 분할은 `outputs/manifests/` (공식 train/test 목록 고정, val 은 공식 train 에서 접시 단위 10%).

**`CropDS` 전처리** (`train_grams_crop.py`):
1. **크롭 박스**: `crop_bbox.json` 에서 접시별 박스를 읽는다(`mixed` 모드). 박스는 `preprocess_plate_roi.process` 가 만든다 — HSV 로 흰 접시(S<70, V>110)를 찾고, 깊이에서 배경(유효 깊이의 85 백분위)보다 깊이값이 30 이상(≈3mm, 단위는 A-0 참조) 작은 = 카메라 쪽으로 솟은 영역을 합친 뒤, 모폴로지 열기/닫기 → 가장 큰 연결 성분 → 구멍 채우기 → 12px 여백 → 정사각형으로 확장. 박스 면적이 프레임의 15% 미만이거나 97% 초과면 **고정 박스 (110,15,575,470)** 를 쓴다. 실측으로 3,261접시 중 2,870장(88%)이 고정 박스, 391장이 검출 박스다.
2. **RGB**: 크롭 → 384×384 bilinear 리사이즈 → [0,1] → ImageNet 정규화.
3. **깊이**: `depth_raw.png`(uint16, mm 단위, 0 = 측정 실패)를 같은 박스로 크롭 → 384×384 **nearest** 리사이즈(깊이값을 섞지 않기 위해) → 유효 픽셀(>0)만으로 **이미지마다 z-점수화**, 측정 실패 픽셀은 0. 절대 거리 대신 "접시 안에서 어디가 높고 낮은가"라는 상대 높이 지도가 된다. 유효 픽셀이 10개 이하면 전부 0.
4. 학습 시 50% 확률로 RGB·깊이를 **같이** 좌우 반전. 그 외 증강은 없다.
5. 반환: `x (4,384,384)`, `y (4,)` 영양소, `gy (80,)` 표준화 그램, 접시 ID.

**학습**: `--init-from` 으로 단계 0 의 `model_state` 를 통째로 불러온 뒤 학습한다(구조가 같아 그대로 로드). 조기종료 기준은 지정하지 않았으므로 기본값 **val mean_PMA**. 학습이 끝나면 best 체크포인트로 test 를 한 번 평가해 기록한다(이 test 값으로 무언가를 선택하지는 않는다).

| 시드 | best epoch / 실행 | 단계 1 test 평균 PMA | 칼로리 PMA |
|------|------|------|------|
| 42 | 27 / 39 | 21.83% | 16.15% |
| 1 | 45 / 57 | 21.75% | 15.89% |
| 2 | 40 / 52 | 21.51% | 15.60% |

**출력**: `outputs/task_runs/task4_final/runs/cs_sidefull_seed{S}/` — `checkpoints/best.pt`, `training_history.json`, `test_metrics.json`.

### A-8. 단계 2 — nRMSE 기준 저학습률 미세조정 (`train_b8.py`)

호출 인자: `--resolution 384 --epochs 150 --patience 10 --lr 2e-5 --batch-size 16 --monitor rmse --init-from <단계1 best.pt>` (나머지 실험 옵션은 전부 기본값 = 현행 구조)

- 데이터·전처리는 단계 1 과 **완전히 같다**(`SpatialCropDS` 는 공간벡터를 안 쓰면 `CropDS` 와 같은 텐서를 내고 더미 `torch.zeros(1)` 만 덧붙인다).
- 달라지는 것은 두 가지: 학습률 **1e-4 → 2e-5**(1/5), 조기종료 기준 **val mean_PMA → val mean_nRMSE**.
- nRMSE = 영양소별 `100 · RMSE / mean(정답)` 의 4영양소 평균(`train_strong._nrmse`). 오차를 제곱해 벌하므로 PMA 보다 **큰 오차(주로 대용량 접시)에 민감**하다. 7/25 TASK5 에서 이 기준으로 바꾸면 같은 학습량 대비 칼로리 PMA −0.31%p, 칼로리 RMSE −1.9 kcal 개선(3시드 3/3)이 확인돼 채택했다.
- `init_from_champion` 이 단계 1 가중치를 이식한다(모양이 같은 텐서는 그대로 복사 — 이 설정에서는 350개 전부 복사).
- 단계 2 는 매우 빨리 끝난다: best epoch 21 / 12 / 7, 실행 31 / 22 / 17 에폭(시드 42/1/2). 상한 도달 없음.

**출력**: `outputs/task_runs/task_b8/runs/b8_sidefull_s{S}/` — `checkpoints/best.pt`(모델 + `cfg` 딕셔너리), `training_history.json`, `test_metrics.json`. **이 세 체크포인트가 최종 챔피언이다.**

### A-9. 평가·앙상블 (`final_paper_table.py` → `score_hybrid.collect`)

1. test 507접시를 접시 ID 순으로 정렬하고, 단계 2 와 같은 전처리(`train=False` → 반전 없음)로 로더를 만든다.
2. 각 시드의 `b8_sidefull_s{S}/checkpoints/best.pt` 를 읽어 `cfg` 로 `B8Net` 을 다시 만들고 예측 `(507, 4)` 를 얻는다.
3. 시드별 지표를 다시 계산해 학습 때 저장한 `test_metrics.json` 과 0.01 이내로 일치하는지 대조한다(불일치 시 중단).
4. 세 시드 예측을 산술평균 → **앙상블 예측**. 이것으로 전체 지표·중량 구간(0~100 / 100~300 / 300~500 / 500g+)별 칼로리 PMA·RMSE 를 계산한다.
5. 결과: `results/final/FINAL_performance_table.{md,json,png}`, `FINAL_champion_preds.npz`.

### A-10. 재현성·주의사항

- **완전 결정적**: 2026-10-02 순정 재학습에서 같은 시드로 다시 돌린 체인이 기존 런을 float 끝자리까지 재현했다. 따라서 *같은 시드 재실행은 독립 표본이 아니다.* 효과를 확인하려면 새 시드를 써야 한다.
- **누수 차단 지점**: 단계 0 은 train split 접시만(assert), 그램 정규화 통계·재료 어휘는 각 단계의 학습 접시로만 계산, 체크포인트 선택은 val(단계 0 은 내부 홀드아웃)만 본다. test 는 학습이 끝난 뒤 한 번 평가할 뿐이다.
- **보조헤드 어휘가 단계 0 과 단계 1·2 에서 다르다**: 단계 0 은 side 학습 접시 3,253장, 단계 1·2 는 overhead 2,485장에서 상위 80종을 뽑는다. 두 어휘는 75종이 겹치지만 같은 순서(같은 출력 인덱스)인 것은 80개 중 11개뿐이다(시드 42 확인). 그래서 단계 0 의 `aux_head` 가중치는 단계 1 에서 사실상 다른 재료 배치에 맞춰 다시 학습된다. 백본과 영양소 헤드의 의미는 그대로 이어진다. 최종 성적은 이 상태에서 측정한 값이며, 어휘를 맞췄을 때의 효과는 따로 실험하지 않았다.
- **에폭 상한 확인**: 각 단계의 metrics json 에 `hit_epoch_cap` 이 있다. `true` 면 아직 학습 중에 잘린 것이므로 결과를 믿으면 안 된다(과거 cs384_crop_seed2 사례). 이 챔피언은 세 단계·세 시드 모두 `false`.
- **val 의 한계**: 기존 val 은 90% 가 train 과 1분 안에 촬영된 거의 같은 접시라 test 보다 훨씬 쉽다(val 평균 PMA ≈ 14.5% vs test ≈ 21%). val 수치를 성능으로 보고하지 말고 체크포인트 선택용으로만 쓴다.
- 이 체인 위에 시도한 변형(단계 1 리허설, 재료 그램 경유 디코딩, 지방 전용 헤드, 학습률·patience·손실·스케줄 변경, ⑦ 순정 재학습)은 판정규칙 v2 를 통과하지 못했거나 순정 재학습에서 재현되지 않아 채택하지 않았다. 기록: `experiment_log.md`, `final_report/`.

