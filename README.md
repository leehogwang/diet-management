# n296_stage3_3 — 음식 사진 → 음식명·재료·대분류 → 영양성분 × 양 보정

`n296_stage3_2` 를 복사해 **음식 대분류**(Gemini 예측)를 양추정 모델 입력과 보정 조건에 추가한 버전이다.
다른 서버(`/home/player0070/2025-2/n296_stage3_3`)에서 작업한 뒤 2026-10-28 에 csgpu2 로 옮겼다
(같은 폴더 압축본: `../n296_stage3_3.tar.gz`, 4.0GB).

관련 문서
- 원본 구조 복원·디스크 문제·AIHub 122 학습 상세: [`../n296_stage3/README.md`](../n296_stage3/README.md)
- 음식 분류기·영양 모델 재학습 보고서: [`FOOD_CLASSIFIER_REPORT.md`](FOOD_CLASSIFIER_REPORT.md)
- 양 보정 이전 파이프라인 평가 보고서: [`PIPELINE_EVAL_REPORT.md`](PIPELINE_EVAL_REPORT.md)
- 구조도: `../n296_stage3_3_pipeline.drawio`, 버전 비교: `../n296_파이프라인_비교_보고서.docx`
- **2026-11 개선 파이프라인 (open-set 분류기 + 모르는 음식은 Gemini 1인분)과 수정사항 전체: [8절](#8-2026-11-개선-파이프라인-실험-experimentssplit_txt_3runs)** — 아직 `pipeline_infer.py` 에는 미반영

---

## 1. 이전 버전과 달라진 점

| 버전 | 내용 |
|---|---|
| n296_stage3 | 원본 체크포인트 구조 복원, 음식 분류기(109종) + Gemini 폴백, 영양 모델 재학습(`nutrition_cls109`) |
| n296_stage3_2 | 양추정 모델(Q1~Q5, 이미지만) 추가, 영양값 × 양 비율 |
| **n296_stage3_3** | 아래 4가지 |

1. **Gemini 재료 단계에서 음식 대분류도 받는다.** 18개 대분류 중 하나를 고른다(`n296/food_category.py`, 목록에 없으면 `기타`).
2. **양추정 모델 입력이 사진 + 대분류다.** 64차원 대분류 임베딩을 ConvNeXt 특징에 붙인다(`QuantityNet`).
   체크포인트는 `runs/quantity_122_cat/best.pth` 이다.
3. **양 보정을 거는 조건을 바꿨다.** stage3_2 는 `known_food`(양추정 학습 음식 78종일 때만)였고,
   stage3_3 은 `known_category`(대분류가 `기타` 가 아닐 때)다. 처음 보는 음식도 대분류가 있으면 보정한다.
4. **평가를 더했다.** `eval_pipeline.py` 에 `category_acc`(Gemini 대분류 정확도)를 추가했다.
   양추정 모델은 처음 보는 음식 실험(`--holdout`)과 대분류를 가린 비교(`*_category_masked`)를 함께 보고한다.
   평가 데이터는 원래 AIHub 122·296 이었고(6.4), 지금은 Nutrition5k test(`runs/eval_n5k/`)로 바꿔 테스트하고 있다(6.5, 진행 중).

음식 분류기(`runs/food_cls_122/best.pth`)와 영양 모델(`runs/nutrition_cls109/model_stage3_296_final.pth`)은 stage3 에서 바꾸지 않았다.

---

## 2. 파이프라인 (`n296/pipeline_infer.py`)

```
이미지
 ├─ 1.   음식명   음식 분류기(ConvNeXt-Tiny, 109종) top1
 │   └─ 1-1.     보정된 확신도 < 0.7 이면 Gemini(gemini-3.5-flash-lite)가 사진만 보고 음식명 예측
 ├─ 2.   재료·대분류  Gemini: 사진 + 음식명 → 재료 목록 + 대분류 (18개 중 하나)
 ├─ 3.   텍스트   "{음식명} (재료: {재료1}, {재료2}, …)" → Gemini 임베딩(gemini-embedding-2, 3072차원)
 ├─ 4-1. 영양     영양 모델(텍스트 임베딩 + ConvNeXt-Base 이미지 + FFT 주파수 특징)
 │               → 칼로리·중량·지방·탄수화물·단백질, 1인분(Q4) 기준          결과 열 pred_<k>_q4
 ├─ 4-2. 양       양추정 모델(ConvNeXt-Tiny + 대분류 임베딩) → Q1~Q5
 └─ 5.   보정     pred_<k> = pred_<k>_q4 × 비율     Q1 0.25 · Q2 0.5 · Q3 0.75 · Q4 1.0 · Q5 1.25
                 대분류가 `기타` 면 보정하지 않음 (비율 1.0)
```

| 단계 | 코드 | 모델 / 외부 API | 사용하는 파일 |
|---|---|---|---|
| 1 음식명 | `n296/food_classifier.py` | timm `convnext_tiny.fb_in22k_ft_in1k`, 288px, T=0.82 | `runs/food_cls_122/best.pth` |
| 1-1 폴백 | `n296/predict_food_gemini.py` (`gemini_json`) | `gemini-3.5-flash-lite` | `.gemini_key` |
| 2 재료·대분류 | `n296/pipeline_infer.py` (`ING_PROMPT`), `n296/food_category.py` | `gemini-3.5-flash-lite` (JSON schema, 대분류는 enum) | `.gemini_key` |
| 3 임베딩 | `n296/embeddings.py` | `gemini-embedding-2` (3072차원) | config 의 `paths.emb_cache` (`data/emb_cache.json`) |
| 4-1 영양 | `n296/model.py` (`Stage3NutritionModel`) | ConvNeXt-Base + VGG16 주파수 분기 + Mixer | `configs/cls109_nutrition.yaml`, `runs/nutrition_cls109/model_stage3_296_final.pth` (`target_stats` 포함) |
| 4-2 양 | `n296/quantity_estimator.py` (`QuantityEstimator`) | timm `convnext_tiny.fb_in22k_ft_in1k_384`, 384px, T=0.53 | `runs/quantity_122_cat/best.pth` |
| 5 보정 | `n296/pipeline_infer.py` | — | — |

주요 옵션 (`pipeline_infer.py`, `eval_pipeline.py` 동일)

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--threshold` | 0.7 | 분류기 확신도가 이보다 낮으면 Gemini 가 음식명을 예측한다 (0.8 로 올려 봤지만 정확도가 떨어졌다) |
| `--quantity-ckpt` | `runs/quantity_122_cat/best.pth` | 양추정 체크포인트. `categories` 가 비어 있으면 이미지만 쓰는 모델로 동작 |
| `--quantity-gate` | `known_category` | `known_category` / `known_food`(stage3_2 방식) / `always` |
| `--quantity-ratio` | `argmax` | `expected` 면 Q 확률로 가중 평균한 비율 |
| `--quantity-min-prob` | 0.0 | 양 확신도가 이 값보다 낮으면 보정하지 않는다 |
| `--skip-quantity` | — | 양 보정 없이 1인분 값 출력 (stage3 동작) |
| `--skip-nutrition` | — | 음식명·재료·대분류까지만 |
| `--gemini-model` / `--rpm` | `gemini-3.5-flash-lite` / 14 | 무료 등급 분당 한도에 맞춘 값 |

출력 (`--out-dir`)
- `pipeline_results.json`, `.csv` 의 사진별 열: 분류기 top5(`cls_name`, `cls_prob`), 음식명 출처 `name_source`(classifier/gemini),
  `food_name`, `ingredients`, `category`, 임베딩 `text`, `q_level`·`q_prob`·`q_probs`·`q_ratio`·`q_ratio_expected`·`q_category`,
  보정 여부 `q_applied`, `scale`, `pred_<k>`(보정 후), `pred_<k>_q4`(보정 전)
- `pipeline_raw.jsonl`: Gemini 결과를 한 장씩 저장한 캐시로, 다시 실행하면 재사용한다. 다음 두 경우에는 그 사진만 다시 처리한다.
  - 저장된 음식명 경로가 지금 기준과 다를 때
  - `category` 가 없는 예전(stage3_2) 결과일 때

  일일 한도에 걸리면 지금까지 결과를 저장하고 멈춘다.

### 음식 대분류 (`n296/food_category.py`)

| 대분류 | 대분류 | 대분류 |
|---|---|---|
| 밥 | 국·탕 | 볶음 |
| 비빔밥·덮밥 | 찌개·전골 | 찜·수육 |
| 볶음밥 | 국물면 | 조림 |
| 김밥·주먹밥 | 비빔·볶음면 | 튀김 |
| 죽·스프 | 파스타 | 전·부침 |
| 반찬 | 구이 | **기타** (만두·회·샐러드 등, 보정 안 함) |

- `CATEGORIES`: 대분류마다 Gemini 프롬프트에 함께 보여 줄 예시를 담는다.
- `FOOD_CATEGORY`: 사람이 정한 음식 → 대분류 정답표다. 양추정 78종과 분류기 109종을 모두 덮는다.
  양추정 학습 라벨과 평가 때 `category_acc` 계산에 쓴다.
- 모든 대분류(기타 제외)에는 양추정 학습 음식이 하나 이상 있다. 학습 음식이 없는 대분류의 임베딩은 학습되지 않기 때문이다.

---

## 3. 데이터와 파일 경로

### 3.1 원본 데이터 (csgpu2)

| 데이터 | 경로 | 쓰는 곳 |
|---|---|---|
| AIHub 122 음식분류 Validation 109종 (21,539장) | `/home/students/cs/202321024/datasets/aihub_122/val_images_109cls/images/<코드>/` | 음식 분류기, 영양 모델 |
| AIHub 122 라벨 zip·영양DB | `/home/students/cs/202321024/datasets/aihub_122/122.음식_분류를_위한_…/01.데이터/` | manifest·정답 |
| AIHub 122 양추정 이미지 (TRAIN 43GB, VAL 5GB, filekey 44898·44879) | **csgpu2 에 없음**, 다른 서버에서 받아 512px 사본만 `data/quantity_122/img/` 에 둠 | 양추정 모델 |
| AIHub 296 로컬 테스트셋 (19종, 2,664장) | `/home/students/cs/202321024/datasets/test_dataset_296/` | test296 평가 |
| 원본 체크포인트 | `/home/students/cs/202321024/datasets/model_stage3_296_final.pth` | 구조 확인용 (`target_stats` 없음) |
| Nutrition5k test (709접시) | `../n296_stage3_4/data/n5k/` (`img/`, `manifest_test.csv`, `manifest.csv`) | Nutrition5k 평가 |

### 3.2 이 폴더의 `data/`

| 파일 | 내용 |
|---|---|
| `food_cls_122/split.csv`, `img/` | 분류기 분할 (train 17,221 / val 2,166 / test 2,152)과 384px 사본 |
| `quantity_122/split.csv`, `img/` | 양추정 분할 (train 31,484 / val 3,423 / test 4,387, 78종)과 512px 사본 39,294장 |
| `manifest_122.csv`, `manifest_122_db.csv` | AIHub 122 양추정 manifest (정답 = 영양DB × Q 비율)와 영양DB 텍스트 전용 manifest |
| `manifest_cls109.csv` | 영양 모델 학습 manifest (16,768행, `category` 열 포함) |
| `manifest_cls109_test_seen.csv`, `_test_unseen.csv` | 영양 모델 테스트 (학습한 음식 / 처음 보는 음식) |
| `manifest_cls109_db.csv` | 영양DB 텍스트 전용 샘플 |
| `manifest_test296.csv` | test_dataset_296 manifest |
| `food_ingredients_122.json` | 영양DB 400종의 재료 목록 (Gemini 가 음식명만 보고 생성). 영양 모델 학습 텍스트와 oracle 평가에 사용 |
| `emb_cache.json` | 텍스트 → Gemini 임베딩 캐시 |
| `emb_cache_n5k.json` | Nutrition5k 평가용 임베딩 캐시 (`configs/cls109_nutrition_n5k.yaml`) |
| `emb_cache_FAKE.json` | 스모크 테스트용 가짜 임베딩 (**실제 학습 금지**) |
| `image_cache/` | 영양 모델 이미지 384px 캐시 |

### 3.3 `runs/` (체크포인트·결과)

| 폴더 | 내용 |
|---|---|
| `food_cls_122/` | 분류기 `best.pth`, `report.json`, `test_per_class.csv`, `ood_test296.csv`, `figures/` |
| `nutrition_cls109/` | 영양 모델 `model_stage3_296_final.pth`, 단계별 best 3개, `split.json`, `history.csv`, `eval_test_{seen,unseen}_{full,masked}.csv`, `figures/` |
| `quantity_122/` | stage3_2 양추정 모델 (이미지만) |
| **`quantity_122_cat/`** | **stage3_3 양추정 모델 (이미지 + 대분류), 파이프라인 기본값** |
| `quantity_holdout_cat/`, `quantity_holdout_img/` | 처음 보는 음식 실험: 대분류마다 1종(16종)을 학습에서 뺐다. `unseen_per_food.csv` 포함 |
| `pipeline_eval/` | 양 보정 없음 (stage3). `PIPELINE_EVAL_REPORT.md` 의 결과 |
| `pipeline_eval_q/` | stage3_2 양 보정 평가 (분류기 기준 0.8, 이미지만 쓰는 양추정) |
| **`pipeline_eval_q2/`** | **stage3_3 평가** (기준 0.7, `quantity_122_cat`, `known_category`). test296 은 실패 (`run_fail_test296.log`) |
| `eval_n5k/` | Nutrition5k test 평가 (진행 중, 6.5절) |
| `eval_original/` | 원본 가중치 `--raw-output` 예측 |
| `smoke/`, `pipeline_smoke*/`, `eval_n5k_smoke/` | 동작 확인용 |
| `quantity_cat_train.log`, `quantity_stream_train.log` | 양추정 학습 로그 |

그 밖의 폴더
- `gemini_food_pred*/`: Gemini 만으로 음식명을 예측한 실험 (분류기 도입 전)
- `notebooks/`: 시각화 노트북. `food_classifier_results`, `nutrition_cls109_results`, `pipeline_eval_results`
- `.pydeps/`: 이전 서버에서 쓴 pip 패키지

### 3.4 코드 (`n296/`)

| 파일 | 내용 |
|---|---|
| `pipeline_infer.py` | 추론 파이프라인 (1~5단계) |
| `eval_pipeline.py` | 파이프라인 전체 평가 (aihub122 · test296 · quantity122) |
| `food_category.py` | **[stage3_3]** 대분류 목록, 음식 → 대분류 정답표, `normalize_category` |
| `quantity_estimator.py` | 양추정 데이터 준비(`prepare`)·학습(`train`)·추론(`QuantityEstimator`). `QuantityNet` 포함 |
| `food_classifier.py` | 음식 분류기 데이터 준비·학습·추론 |
| `predict_food_gemini.py` | Gemini 음식명·재료 예측, 공통 호출 함수 `gemini_json` |
| `model.py` | 영양 모델 (원본 state_dict 에서 복원) |
| `embeddings.py` | Gemini 임베딩과 캐시 (`--provider fake` 는 테스트용) |
| `train.py`, `evaluate.py`, `dataset.py`, `metrics.py`, `common.py` | 영양 모델 학습·평가·데이터·PMAE·설정 |
| `build_manifest*.py`, `build_text_only.py` | manifest 생성 (296 / 122 / cls109 / 영양DB) |
| `check_compat.py` | 원본 가중치 strict 로드 확인 |

### 3.5 config (`configs/`)

| 파일 | 용도 | `storage_root` |
|---|---|---|
| `cls109_nutrition.yaml` | 파이프라인 영양 모델 | `/home/player0070/…` (이전 서버) |
| `cls109_nutrition_n5k.yaml` | 위와 같은 설정의 csgpu2 용 복사본. 임베딩 캐시만 `emb_cache_n5k.json` | **이 폴더** |
| `aihub122.yaml` | 분류기·양추정·AIHub 122 학습 | `/home/player0070/…` |
| `stage3_296.yaml`, `smoke_test296_fake.yaml` | 원본 296 학습, 스모크 테스트 | `/home/player0070/…` |

> ⚠ **csgpu2 에서 실행할 때.** config 의 상대 경로는 모두 `storage_root` 기준으로 해석된다(`n296/common.py` 의 `resolve`).
> 그런데 `cls109_nutrition_n5k.yaml` 을 뺀 config 는 `storage_root` 가 이전 서버 경로라서, 이 서버에서는 경로를 찾지 못한다.
> - **추론**은 `cls109_nutrition_n5k.yaml` 을 쓰거나, 다른 config 의 `storage_root` 를 이 폴더로 바꾼다.
> - **재학습**은 `data/*/split.csv`, `manifest_cls109*.csv` 의 이미지 절대 경로도 `/home/player0070/2025-2/n296_stage3_3/…` 로 되어 있어 고쳐야 한다.
>   예: `sed 's#/home/player0070/2025-2/n296_stage3_3#'"$PWD"'#'`
>
> mergerfs 250GB 기준선 문제가 있으므로 큰 파일을 쓰기 전에 `df -h /mnt/nvme1 /mnt/nvme2 /mnt/nvme3 /mnt/root-extra` 로 여유 공간을 확인한다.
> 원인은 `../n296_stage3/README.md` 의 "디스크" 절을 참고.

---

## 4. 실행 (csgpu2, conda env `diet`)

```bash
cd /home/students/cs/202321024/2025_2/2025_2-2/2025-2/n296_stage3_3
PY=~/.conda/envs/diet/bin/python      # torch 2.2, google-genai 설치됨. Gemini 키는 .gemini_key

# 추론
CUDA_VISIBLE_DEVICES=0 $PY -m n296.pipeline_infer --config configs/cls109_nutrition_n5k.yaml \
    --nutrition-ckpt runs/nutrition_cls109/model_stage3_296_final.pth \
    --images <이미지 또는 폴더> --out-dir runs/<이름>
#   stage3_2 방식과 비교:  --quantity-gate known_food --quantity-ckpt runs/quantity_122/best.pth
#   양 보정 없이:         --skip-quantity

# 파이프라인 평가 (aihub122 109장 · test296 57장 · quantity122 115장 = 281장, 사진마다 Gemini 1~2회)
CUDA_VISIBLE_DEVICES=0 $PY -m n296.eval_pipeline --config configs/cls109_nutrition_n5k.yaml \
    --nutrition-ckpt runs/nutrition_cls109/model_stage3_296_final.pth --out-dir runs/pipeline_eval_q3

# 양추정 모델 재학습 (split.csv 경로 수정 후)
CUDA_VISIBLE_DEVICES=0 $PY -m n296.quantity_estimator train --config configs/aihub122.yaml            # 이미지 + 대분류
CUDA_VISIBLE_DEVICES=0 $PY -m n296.quantity_estimator train --config configs/aihub122.yaml --holdout  # 처음 보는 음식 실험
#   --no-category: 이미지만 / --cat-dropout 0.2: 학습 때 대분류를 기타로 바꾸는 확률

# Nutrition5k 평가 (Gemini 일일 한도에 걸리면 30분마다 다시 시도)
nohup bash runs/eval_n5k/run_eval_n5k.sh &
```

`eval_pipeline.py` 는 세트마다 `pipeline`(보정 후)과 `pipeline_q4`(보정 전)를 함께 낸다.
quantity122 에는 `q_acc`, `q_acc_within1`, 정답 Q 등급별 PMAE, `oracle_q`(정답 음식명·재료 + 정답 Q)가 추가된다.

---

## 5. 지표

- **PMAE** = `100 × MAE ÷ 정답 평균` (%). **mean PMAE** = 칼로리·지방·단백질 PMAE 의 평균 (원본 체크포인트 `mean_pmae` 와 같은 정의).
  표에서 탄수화물은 carb 헤드 출력이다.
- **oracle**: 파이프라인 인식 결과 대신 정답 음식명과 `food_ingredients_122.json` 재료로 텍스트를 만든 결과.
  같은 사진, 같은 영양 모델을 쓴다. 파이프라인 − oracle = 인식 단계에서 생긴 오차.
- **pipeline_q4**: 양 보정 전 (모든 사진을 1인분으로 출력).

## 6. 성능 정리

### 6.1 음식 분류기 (`runs/food_cls_122/report.json`, AIHub 122 109종)

| | top1 | top5 | ECE |
|---|---|---|---|
| val (2,166장) | 89.7% | 98.3% | 0.022 |
| test (2,152장) | 88.8% | 98.2% | 0.026 |

| 확신도 기준 (test) | 분류기 결과 사용 | 그중 정확도 | Gemini 로 |
|---|---|---|---|
| 0.5 | 95.2% | 91.5% | 4.8% |
| **0.7 (기본)** | 89.4% | 94.1% | 10.6% |
| 0.8 | 85.1% | 95.1% | 14.9% |
| 0.9 | 78.1% | 96.9% | 21.9% |

- 약한 음식: 라면 53%, 비빔국수 55%, 코다리조림·잔치국수·돼지갈비찜 67%.
- 분류기가 모르는 음식에도 확신도가 높게 나온다. test_dataset_296 19종은 모두 109종 밖인데, 그중 32% 가 확신도 0.7 이상이다(`ood_test296.csv`).

### 6.2 영양 모델 (`runs/nutrition_cls109/`, 정답 = 영양DB 1인분)

| test | 입력 | 칼로리 | 지방 | 단백질 | 중량 | 탄수화물 | mean PMAE |
|---|---|---|---|---|---|---|---|
| 학습한 음식 83종 (1,641장) | 평균값 예측 | 44.9 | 67.4 | 57.3 | 67.8 | 63.1 | 56.6 |
| | 텍스트만 | 2.2 | 3.4 | 2.2 | 2.7 | 3.1 | 2.6 |
| | 텍스트 + 이미지 | 2.9 | 4.5 | 4.0 | 4.0 | 3.9 | **3.8** |
| 처음 보는 음식 16종 (3,130장) | 평균값 예측 | 42.3 | 63.2 | 39.1 | 61.0 | 79.9 | 48.2 |
| | 텍스트만 | 42.1 | 60.4 | 40.8 | 19.6 | 46.1 | 47.8 |
| | 텍스트 + 이미지 | 39.4 | 53.9 | 37.9 | 18.9 | 47.1 | **43.8** |

학습한 음식에서는 "음식명 → 1인분 값"을 외운 결과다. 처음 보는 음식에서는 평균값 예측보다 조금 나은 수준이다.

### 6.3 양추정 모델 (AIHub 122 양추정, Q1~Q5)

| 체크포인트 | 입력 | 학습 음식 | test 정확도 | ±1 등급 | 비율 MAE | best epoch |
|---|---|---|---|---|---|---|
| `quantity_122` (stage3_2) | 이미지 | 78종 | 99.2% | 100% | 0.0019 | 25 |
| **`quantity_122_cat` (stage3_3)** | 이미지 + 대분류 | 78종 | **99.5%** | 99.98% | 0.0014 | 26 |
| ↳ 같은 사진, 대분류를 가림 | 이미지 | | 99.5% | 99.98% | 0.0014 | |

처음 보는 음식 실험: 대분류마다 1종씩, 16종 7,945장을 학습에서 뺐다.

| 체크포인트 | 입력 | 학습한 음식 test (3,500장) | **처음 보는 음식 (7,945장)** | ±1 등급 | 비율 MAE |
|---|---|---|---|---|---|
| `quantity_holdout_cat` | 이미지 + 대분류 | 98.8% | **46.5%** | 93.2% | 0.151 |
| ↳ 대분류를 가림 | 이미지 | 98.9% | 46.5% | 93.2% | 0.151 |
| `quantity_holdout_img` | 이미지 | 99.0% | **48.5%** | 93.4% | 0.147 |

- **대분류는 양추정 정확도에 거의 영향이 없다.** 대분류를 가려도 결과가 같고, 이미지만 쓴 모델이 처음 보는 음식에서 오히려 2%p 높다.
  학습 때 모델이 대분류 임베딩을 거의 쓰지 않은 것으로 보인다.
- 학습한 음식의 99% 는 같은 음식·같은 촬영 환경 안에서 잰 값이다. 처음 보는 음식에서는 46~49% 로 떨어진다(5등급이라 무작위는 20%).
  다만 ±1 등급 안에는 93% 가 들어간다.
- 처음 보는 음식별 정확도(`quantity_holdout_cat/unseen_per_food.csv`): 콩밥·돼지갈비 81% 로 높고,
  일반김밥 24%, 채소튀김 24%, 시금치나물 26%(±1 등급 69%), 자장면 29% 로 낮다.

### 6.4 이전 평가: AIHub 122 · 296 (`eval_pipeline.py`)

원래 평가는 AIHub 122(학습한 한식, 양추정 사진)와 AIHub 296(`test_dataset_296`, 분류기가 모르는 한식 19종)으로 했다.
지금은 Nutrition5k 로 바꿔 테스트하고 있다(6.5). 이전 결과는 다음과 같다.

**요약: 버전별 최종 출력 (PMAE %)**

| 세트 | 버전 | 칼로리 | 중량 | 지방 | 탄수화물 | 단백질 | mean PMAE |
|---|---|---|---|---|---|---|---|
| aihub122 (109장, 1인분 정답) | stage3 (양 보정 없음) | 15.8 | 12.8 | 25.3 | 15.9 | 20.8 | **20.6** |
| | stage3_3 (보정 후) | 30.3 | 27.2 | 35.7 | 30.8 | 33.4 | 33.1 |
| quantity122 (115장, 양 반영 정답) | stage3_3 (보정 전) | 57.6 | 54.6 | 66.1 | 57.3 | 62.8 | 62.2 |
| | stage3_3 (보정 후) | 22.5 | 14.8 | 35.4 | 17.7 | 35.9 | **31.3** |
| test296 (57장, 처음 보는 음식) | stage3 (양 보정 없음) | 44.7 | 59.9 | 54.6 | 58.7 | 53.8 | **51.0** |

- test296 에서 음식명 출처는 Gemini 71.9%, 분류기 28.1% 였다. 이름이 정답과 일치한 경우는 5.3% 다.
  분류기가 확신한 16장(50.4%)과 Gemini 41장(51.7%)의 mean PMAE 차이는 거의 없다.
- stage3_3 의 test296 은 이전 서버에 이미지가 없어 돌리지 못했다. 위 test296 값은 stage3(양 보정 없음) 결과다.
- 아래는 버전별 비교와 stage3_3 상세 결과다.

| 세트 | 사진 | 정답 | stage3 (`pipeline_eval`) | stage3_2 (`pipeline_eval_q`) | **stage3_3 (`pipeline_eval_q2`)** |
|---|---|---|---|---|---|
| aihub122 | 109 (109종 × 1) | 영양DB 1인분 | 20.6 | 35.3 (보정 전 21.6) | **33.1** (보정 전 20.9) |
| quantity122 | 115 (23종 × Q1~Q5) | 영양DB × Q 비율 | — | 30.4 (보정 전 63.0) | **31.3** (보정 전 62.2) |
| test296 | 57 (19종 × 3) | 296 라벨 1인분 | 51.0 | — | 실패 (이미지 경로 없음) |

stage3_3 상세 (`runs/pipeline_eval_q2/summary.json`)

| 항목 | aihub122 | quantity122 |
|---|---|---|
| Gemini 로 넘긴 비율 | 14.7% | 26.1% |
| 음식명 정확도 | 78.0% | 66.1% |
| **대분류 정확도 (Gemini)** | **94.5%** | **94.8%** |
| 양 보정 적용 | 97.2% | 98.3% |
| 양 등급 정확도 / ±1 | — | 99.1% / 100% |
| 예측 Q 분포 | Q1 6 · Q2 5 · Q3 23 · Q4 43 · Q5 32 | 정답과 거의 같음 |

| 세트 · 그룹 | 사진 | 입력 | 칼로리 | 중량 | 지방 | 탄수화물 | 단백질 | mean PMAE |
|---|---|---|---|---|---|---|---|---|
| quantity122 전체 | 115 | 파이프라인 (보정 후) | 22.5 | 14.8 | 35.4 | 17.7 | 35.9 | **31.3** |
| | | 보정 전 (1인분) | 57.6 | 54.6 | 66.1 | 57.3 | 62.8 | 62.2 |
| | | oracle (정답 음식명 + 예측 Q) | 13.8 | 9.9 | 19.0 | 13.6 | 17.3 | 16.7 |
| | | oracle_q (정답 음식명 + 정답 Q) | 12.0 | 8.4 | 16.9 | 12.0 | 15.7 | 14.8 |
| ↳ 음식명 정답 | 76 | 파이프라인 | 18.3 | 12.8 | 24.8 | 15.5 | 26.4 | 23.2 |
| ↳ 음식명 오답 | 39 | 파이프라인 | 29.4 | 17.8 | 59.5 | 21.1 | 49.6 | 46.2 |
| ↳ 영양 모델 학습 음식 | 85 | 파이프라인 / oracle | 16.4 / 6.0 | 11.8 / 6.2 | 29.8 / 9.1 | 11.4 / 5.6 | 29.5 / 8.0 | 25.2 / 7.7 |
| ↳ 영양 모델 처음 보는 음식 | 30 | 파이프라인 / oracle | 46.9 / 44.5 | 28.4 / 26.8 | 56.2 / 55.8 | 39.6 / 41.8 | 70.9 / 69.2 | 58.0 / 56.5 |
| aihub122 전체 | 109 | 파이프라인 (보정 후) | 30.3 | 27.2 | 35.7 | 30.8 | 33.4 | 33.1 |
| | | 보정 전 (1인분) | 16.6 | 13.3 | 25.3 | 16.5 | 20.8 | 20.9 |
| | | oracle | 25.3 | 22.2 | 27.0 | 27.9 | 24.8 | 25.7 |

quantity122 의 정답 Q 등급별 mean PMAE (보정 후 / 보정 전)

| Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|
| 53.2 / 302.6 | 32.3 / 99.1 | 26.8 / 40.8 | 31.7 / 31.3 | 28.7 / 36.8 |

해석
- **양이 정답에 반영된 세트(quantity122)에서는 보정 효과가 크다.** 62.2 → 31.3% 로 절반이 됐고, 특히 Q1·Q2 사진에서 크다.
- **aihub122 는 보정하면 나빠진다(20.9 → 33.1%).** 이 세트의 정답은 모든 사진을 1인분으로 가정한 값이다.
  게다가 음식분류 사진은 양추정 사진과 촬영 환경이 달라, 61% 가 Q4 가 아닌 등급으로 예측됐다(Q5 32장, Q3 23장 등).
  이 세트의 보정 후 수치는 성능 저하로 해석하면 안 된다.
- stage3_2 와 stage3_3 의 차이는 표본 수준이다. quantity122 는 30.4 → 31.3%, aihub122 는 35.3 → 33.1% 다.
  두 버전은 분류기 기준(0.8 / 0.7)도 달라 대분류 효과만 따로 볼 수 없다.
  6.3 결과와 합치면 **대분류를 넣어도 양 보정 자체는 거의 달라지지 않는다**. 바뀐 것은 보정을 거는 범위다(`known_category`).
- 남은 오차의 대부분은 음식명이다. 음식명이 틀리면 46.2% 로, 맞을 때(23.2%)의 2배다.
  정답 음식명을 주면(oracle_q) 14.8% 까지 내려간다.

### 6.5 Nutrition5k test (`runs/eval_n5k/`, **진행 중 — 중간 결과**)

- 대상: Nutrition5k test 709접시, 접시당 사진 1장(`selected_images.csv`, camera_A 또는 overhead).
  정답은 접시 전체의 실제 값이다(`../n296_stage3_4/data/n5k/manifest_test.csv`).
- 2026-10-28 20:27 기준 **320/709장**을 처리했다. Gemini 일일 한도에 걸려 `run_eval_n5k.sh` 가 30분마다 다시 시도한다.
  끝나면 `eval_n5k_stage3_3.py` 가 `summary.json`, `dish_results.csv`, `detail_metrics.csv` 를 만든다.
- 아래 수치는 처리된 320장 중 0 < 칼로리 ≤ 3000 인 317접시로 같은 코드를 돌려 계산했다. 결과 파일은 만들지 않았다.

| 모델 | 칼로리 | 중량 | 지방 | 탄수화물 | 단백질 | mean PMAE |
|---|---|---|---|---|---|---|
| stage3_3 (보정 후) | 103.3 | 63.6 | 116.3 | 117.9 | 89.8 | **103.1** |
| stage3_3 (보정 전, 1인분) | 133.8 | 84.2 | 135.4 | 163.2 | 105.9 | 125.0 |
| 평균값 예측 (N5K train 평균) | 102.3 | 90.3 | 128.8 | 87.6 | 142.7 | 124.6 |
| stage3_4 (N5K 로 학습, 같은 사진) | 19.4 | 29.5 | 25.4 | 18.6 | 23.4 | **22.7** |

- 정답 평균은 칼로리 154 kcal, 중량 115 g 이다. 한식 1인분(aihub122 평균 459 kcal, 424 g)보다 훨씬 작다.
  그래서 1인분 기준 출력은 크게 과대 예측하고, 양 보정(대부분 Q1~Q3 으로 예측)으로 일부만 줄어든다.
- 음식명 출처는 Gemini 163장, 분류기 157장이다. 분류기 157장은 109종 밖 음식에 확신한 경우다.
  대분류는 `기타` 가 160장이고, 이 160장은 양 보정을 하지 않는다.
- **stage3_3 은 Nutrition5k 에서 평균값 예측 수준이다.** 한식 109종 밖 음식에는 일반화하지 못한다.

**이전 평가와 비교 (stage3_3 계열 mean PMAE %)**

| 평가 세트 | 음식 | mean PMAE |
|---|---|---|
| aihub122 | 학습한 한식 109종 (1인분 정답) | 20.6 (stage3) / 33.1 (stage3_3 보정 후) |
| quantity122 | 학습한 한식 23종 (양 반영 정답) | 31.3 |
| test296 | 처음 보는 한식 19종 | 51.0 (stage3) |
| Nutrition5k (중간) | 서양식 접시 | 103.1 |

---

## 7. 한계와 다음 단계

1. **대분류 입력의 효과가 없다** (6.3). 대분류가 정말 도움이 되는지 보려면 둘 중 하나가 필요하다.
   - cat-dropout 을 낮추거나 대분류별 Q 기준(그릇 크기)을 명시적으로 주기
   - 더 다양한 촬영 환경의 데이터
2. **양추정 모델의 일반화.** 학습한 음식·환경 밖에서는 정확도가 절반 아래다(46%).
   그런데 `known_category` 는 처음 보는 음식에도 보정을 건다. aihub122(다른 촬영 환경)에서는 보정이 오히려 손해였다.
   보정을 거는 범위와 `--quantity-min-prob` 을 함께 조정할 필요가 있다.
3. **음식명 오류가 가장 큰 오차 원인이다.** 분류기는 모르는 음식에도 높은 확신도를 낸다(OOD 검출 없음).
   Gemini 폴백은 영양DB 에 없는 이름을 낸다. 개선 방법은 두 가지다.
   - Gemini 에게 분류기 top-5 나 영양DB 목록 중에서 고르게 하기
   - 분포 밖(OOD) 검출 추가
4. **영양 모델은 처음 보는 음식에 약하다**(43.8%, Nutrition5k 103%). 학습 음식 확대나 영양DB 최근접 검색 방식과 비교해야 한다.
5. **test296 미평가.** stage3_3 의 test296 평가는 이전 서버에 이미지가 없어 실패했다.
   csgpu2 에는 `datasets/test_dataset_296` 이 있으므로 4절 명령으로 다시 돌릴 수 있다.
6. 표본이 작다. aihub122 109장, quantity122 115장이라 그룹별 수치(20~40장)는 변동이 크다.
   Gemini 출력도 temperature 0 이지만 호출마다 조금 다를 수 있다.

---

## 8. 2026-11 개선 파이프라인 (실험: `experiments/split_txt_3runs/`)

- 작성 2026-11-02. AIHub 122 를 학습한 음식 / 처음 보는 음식 분할(`food_split_txt`)로 나눠 단계별로 바꿔 보고, test G4 가 가장 낮은 구성을 정리했다.
- **상태: 실험 스크립트로만 평가했다.** `n296/pipeline_infer.py` 에는 아직 반영하지 않았다. 2절은 지금 `pipeline_infer.py` 의 동작 그대로다.
- 원본 코드에서 고친 파일은 `n296/model.py` 하나다 (대분류 입력 옵션 `model.cat_dim`, 기본 0 이면 원래 구조와 같다). 나머지는 모두 `experiments/split_txt_3runs/` 의 새 파일이다.
- 상세: [`REPORT.md`](experiments/split_txt_3runs/REPORT.md)(분할 학습), [`CHANGES_OSR_CLASSIFIER.md`](experiments/split_txt_3runs/CHANGES_OSR_CLASSIFIER.md)(open-set 분류기·파이프라인 재평가). 8.5 이후 내용은 이 절에만 있다.

### 8.1 현재 최고 성능 파이프라인

```
이미지
 ├─ 1.   음식 분류기 top1 + open-set 판정 (Mahalanobis 거리 점수, 처음 보는 음식을 넣은 val 로 정한 threshold)
 │
 ├─ [아는 음식]                                   ├─ [모르는 음식]
 │   1-a. 음식명 = 분류기 top1                     │   1-b. 음식명 = Gemini (사진)
 │   2.   재료·대분류 = Gemini (사진 + 음식명)       │   2.   재료·대분류 = Gemini (사진 + 음식명)
 │   3.   "음식명 (재료: …)" → Gemini 임베딩         │   3.   1인분 영양값 = Gemini (음식명 + 재료, 텍스트만)
 │   4-1. 1인분 영양값 = 영양 모델                   │        (임베딩·영양 모델 단계 없음)
 │        (텍스트 임베딩 + 이미지 + 대분류 입력)       │
 │
 ├─ 4-2. 양       양추정 모델 (사진 + 대분류) → Q1~Q5, argmax
 └─ 5.   보정     최종 = 1인분 영양값 × Q 비율 (0.25 / 0.5 / 0.75 / 1.0 / 1.25), 항상 보정
```

| 단계 | 2절(stage3_3) 과 다른 점 | 실험 코드·파일 |
|---|---|---|
| 1 판정 | softmax 확신도 ≥ 0.7 → **Mahalanobis 거리 + val 균형 threshold** (처음 보는 음식 val 이 없는 분할은 known val TPR99). 분류기는 재학습 없이 그대로 | `src/train_classifier_osr.py --baseline`, `runs/cls_osr_<split>_baseline/` |
| 1-b, 2 Gemini | `gemini-3.5-flash-lite` → `gemini-3.8-flash` (temperature 0, thinking 0) | `src/gemini.py`, `src/gemini_ing.py` |
| 4-1 영양 모델 | 대분류 임베딩(64차원)을 헤드 입력에 추가 (`--cat`, 학습 때 20% 확률로 기타로 가림) | `src/train.py --text ing --cat`, `runs/<split>_q4_ing_cat/` |
| 3 (모르는 음식) | **신규.** 영양 모델 대신 Gemini 가 음식명 + 재료로 1인분 영양값(칼로리·지방·단백질·탄수·중량) 추정 | `src/gemini_serving.py --prompt ing`, `gemini_serving_ing.jsonl` |
| 5 보정 | `known_category`(대분류 기타면 보정 안 함) → `always` | `src/compare_pipeline_cat.py`, `src/eval_gemini_serving.py` |

실험에서는 분류기·영양 모델·양추정 모델을 분할마다 처음부터 따로 학습했다 (2절의 109종 모델이 아님).

### 8.2 성능 (영양 G4 %, 낮을수록 좋음)

test: `122_2_forunseen` = 처음 보는 음식 23종 5,884장, `122_1_forseen` = 학습한 음식 78종 4,387장.

| 단계별 누적 변경 | unseen G4 | unseen A4 | seen G4 |
|---|---|---|---|
| 2절 규칙 (softmax 0.7, 보정 `known_category`) | 50.11 | 94.8 | 1.44 |
| + 보정 항상 적용 | 49.53 | 94.1 | 1.17 |
| + open-set 분류기 판정 | 45.78 | 82.0 | 1.20 |
| + 모르는 음식은 Gemini 1인분 (음식명만) | 34.59 | 36.4 | 1.40 |
| **+ Gemini 1인분에 재료 추가 (현재 최고)** | **33.67** | **35.0** | **1.40** |

현재 최고 구성의 영양소별 G: unseen 칼로리 29.8 · 지방 47.5 · 단백질 31.5 · 탄수 25.9 · 중량 21.1 / seen 1.01 · 1.64 · 1.27 · 1.69 · 1.04.
분류기 음식명 사용률은 unseen 0.85%, seen 99.3% 다. seen 이 1.17 → 1.40 으로 오른 것은 아는 음식 약 30장(0.7%)이 Gemini 로 넘어갔기 때문이다.

### 8.3 변경 사항 (시간순)

| # | 날짜 | 변경 | 결과 |
|---|---|---|---|
| 1 | 11/1 | 영양 모델을 3개 분할(122_1_forseen, 122_2_forunseen, n5k)에 따로 학습 | 정답 텍스트 G4: seen 2.82 / unseen 50.77 / n5k 31.39 |
| 2 | 11/1 | 분할별 1인분(Q4) 영양 모델·양추정 모델 학습, 직접 예측 vs 1인분 × 양 비교 | 두 방식 차이 거의 없음 (unseen 45.80 vs 45.78) |
| 3 | 11/2 | 텍스트를 `음식명 (재료: …)` 형식으로 학습 (`_ing`) | 중단, 4로 대체 |
| 4 | 11/2 | 영양 모델에 대분류 입력 추가 (`_ing_cat`), `n296/model.py` 수정 | 현재 사용하는 영양 모델 |
| 5 | 11/2 | 분할별 음식 분류기 학습 + 2절 규칙 평가 | 처음 보는 음식의 50.3% 가 확신도 0.7 을 넘어 오답 음식명으로 넘어감 |
| 6 | 11/2 | open-set 분류기: ArcFace 재학습(효과 없음, 미채택) vs 기존 CE + 거리 점수 | 기존 CE + Mahalanobis 로 오답 음식명 통과 0.503 → 0.008 |
| 7 | 11/2 | 파이프라인 재평가(`pipe_osr`), 기존 조건 재계산값이 백업과 같음을 확인 | unseen 49.53 → 45.78 |
| 8 | 11/2 | 오차 분해 (8.4) | 병목은 처음 보는 음식의 1인분 영양값 |
| 9 | 11/2 | 모르는 음식의 1인분 값을 Gemini 가 추정 (음식명만) | unseen 45.78 → 34.59 |
| 10 | 11/2 | Gemini 1인분 추정에 재료 추가 | unseen 34.59 → 33.67 (지방 50.5 → 47.5) |

### 8.4 병목 분석 (unseen, 7번 시점 G4 45.78 기준)

| 정답으로 바꾼 단계 | G4 |
|---|---|
| 음식명 | 44.86 |
| 양(Q) | 45.02 |
| **1인분 영양값** | **15.76** |

- 122 정답은 `영양DB 1인분값 × Q 비율` 이라 같은 음식 안에서 1인분 정답이 일정하다. 영양 모델은 학습 음식 44종의 값을 외우고, 처음 보는 음식은 train 평균 쪽으로 수축한다 (음식별 정답 → 예측 기울기 칼로리 0.29, 지방 0.17).
- 정답 음식명·재료를 줘도 거의 나아지지 않아, 텍스트에서 영양값으로 일반화하지 못한다. 그래서 외부 지식(Gemini)으로 1인분 값을 공급하는 9번을 시도했다.
- 7절 3번(OOD 검출 없음)은 6번으로, 7절 4번(처음 보는 음식에 약함)은 9·10번으로 일부 해결했다.

### 8.5 시도했지만 채택하지 않은 것

| 시도 | unseen G4 | 이유 |
|---|---|---|
| ArcFace 로 분류기 재학습 | 라우팅이 기존 CE + Mahalanobis 보다 나쁨 | best epoch 1, 학습이 진행될수록 촬영 환경에 과적합 |
| Gemini 프롬프트에 "식약처 DB 1인분 기준" 명시 | 37.18 | 영양값이 흔들림. Gemini 1인분 중량은 원래도 DB 와 중앙값 1.00 으로 맞음 |
| Gemini g당 밀도 × 사진 기반 중량 예측 | 42.5 ~ 45.3 | 사진 중량 추정 G 34~37 (Gemini 1인분 중량 × Q 비율은 20.3). 정답 중량이면 25~28 |
| 대분류별 평균·중앙값 1인분값 (Gemini 없이) | 48 ~ 50 | 같은 대분류 안에서도 값 차이가 큼 (국·탕 38~964 kcal). 정답 대분류여도 49.7 |
| 양 비율 Q 확률 가중 평균 (`expected`) | 50.67 (영양 모델 기준) | argmax 보다 나쁨, 특히 Q1·Q5 |

### 8.6 남은 문제

1. **`pipeline_infer.py` 반영.** open-set 판정(점수·threshold 저장)과 모르는 음식의 Gemini 1인분 단계를 옵션으로 넣어야 한다. 실험은 분할별 모델·threshold 라 운영용 분류기 하나와 threshold 하나를 정해야 한다.
2. **양 추정.** Gemini 1인분값에 정답 Q 를 쓰면 27.93 (현재 33.67). unseen 양 정확도 0.557.
3. **지방.** G 47.5 로 가장 크다. 재료 목록에는 양이 없고, DB 값 자체가 일반 조리법과 다른 음식도 있다 (김치볶음밥 500 g 에 지방 5.1 g).
4. **1인분 값.** 정답 1인분값 × 예측 Q 는 15.76 이라 아직 여지가 크다. 후보: 같은 음식명끼리 Gemini 값 중앙값으로 안정화, Gemini 에 사진도 함께 주기, 로컬 영양 DB 최근접 검색.
5. **Gemini 의존.** 모르는 음식 1장당 Gemini 3회 호출 (음식명, 재료·대분류, 1인분).
6. **검증 범위.** test 한 번의 결과라 작은 차이(10번의 0.9%p)는 우연일 수 있다.

### 8.7 파일 (`experiments/split_txt_3runs/`)

| 구분 | 파일 |
|---|---|
| 데이터 | `src/prep.py`, `records.json`, `splits.json` |
| 영양·양 모델 | `src/train.py`, `src/train_quantity.py`, `src/eval_quantity_cats.py`, `runs/<split>[_q4]_ing_cat/`, `runs/<split>_quantity/` |
| 분류기 | `src/train_classifier.py`, `src/train_classifier_osr.py`, `runs/cls_<split>/`, `runs/cls_osr_<split>[_baseline]/` |
| Gemini·텍스트 | `src/gemini.py`, `src/gemini_ing.py`, `src/gemini_serving.py` (`--prompt default/mfds/ing`), `src/texts_ing.py`, `src/embed.py`, `src/embed_ing.py` |
| 평가 | `src/compare_pipeline.py`, `src/compare_pipeline_cat.py`, `src/eval_gemini_serving.py` (현재 최고 = 조건 `r_ing`), `src/eval_category_serving.py` |
| 결과 | `pipeline_compare_ing_cat.json`, `gemini_serving_eval.json`, `logs/eval_gemini_serving.log`, `logs/eval_category_serving.log` |
| 실행 | `run_all.sh`, `run_pipeline.sh`, `run_ing.sh`, `run_ing_cat.sh`, `run_after_cat.sh`, `run_osr.sh`, `run_pipe_osr.sh` |
| 백업 | `backup_before_pipe_osr/` (7번 실행 전 결과) |

현재 최고 구성 재현 (이미 만든 Gemini 캐시가 있으면 새 호출 없음):

```bash
cd /home/leehogwang/food_b/experiments/split_txt_3runs
PY=/home/leehogwang/Food-Nutrition-Estimation/.venv/bin/python
./run_pipe_osr.sh > logs/run_pipe_osr.log 2>&1          # open-set 판정 → 재료·임베딩 → 양추정·영양 재평가
$PY src/gemini_serving.py --prompt ing                    # 모르는 음식 1인분 (음식명 + 재료)
$PY src/eval_gemini_serving.py                            # 조건 r_ing
```
