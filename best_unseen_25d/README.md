# 실험 25-d: 처음 보는 음식 영양 추정 최고 모델

> 코드 원본: [leehogwang/Food-Nutrition-Estimation `feat/stage3-kfold-ensemble`](https://github.com/leehogwang/Food-Nutrition-Estimation/tree/feat/stage3-kfold-ensemble) — 아래 경로는 모두 그 저장소 기준이다.

AIHub 122 에서 **학습에 한 번도 나오지 않은 음식 23종(5,884장)** 을 대상으로 한 최고 단일 모델이다.
예측을 **1인분 × 배율**로 나누고, 1인분은 LLM(Gemini)이 음식명만 보고 추정한 값에서 출발해 작은 보정만 학습한다.

## 성능 (122 처음 보는 음식 23종, 5-fold 로그 평균 앙상블, 4대 영양소 평균 %)

| 영양성분 | ⑧ 기준 모델 A | **25-d A** | ⑧ G | **25-d G** |
|---|---|---|---|---|
| 칼로리 | 41.58 | **31.37** | 38.75 | **30.66** |
| 지방 | 48.87 | **44.95** | 57.44 | **48.26** |
| 단백질 | 33.71 | **28.48** | 36.20 | **30.78** |
| 탄수화물 | 42.98 | **32.03** | 33.07 | **26.43** |
| **4대 평균** | 41.79 | **34.21** | 41.37 | **34.03** |

- A = mean(|p−t| / (|t|+1)), G = Σ|p−t| / Σt (논문 표준 PMAE)
- fold 별 단일 모델 A 36.44 ± 1.75 (`experiments/25d_decomp_delta_l2/ensemble_results.json`), 영양소별 값은 fold 별 `test_preds.npz` 를 로그 평균해 계산
- 같은 계열의 25-f 는 G 33.51 로 G 최고지만 A 36.36 이라, A·G 가 고르게 좋은 25-d 를 대표로 둔다.

## 코드 위치

이 폴더는 정리용이고, 실제 코드는 아래 경로에 있다 (경로는 이 저장소 기준).

| 단계 | 파일 | 하는 일 |
|---|---|---|
| 모델 본체 | `depth/model_depth.py` | ConvNeXt-B + 주파수 분기(FFT → VGG16) + UniDepth 깊이 분기(ResNet18), 사진별 게이트, 텍스트 64D |
| Stage 1·2 학습 | `depth/train_depth.py` | Stage 1 OmniFood8K → Stage 2 122 분류 369종 (⑧ 과 동일) |
| 깊이 사전계산 | `depth/precompute_depth.py`, `depth/depth_data.py` | UniDepth V2 깊이 → 112×112 캐시 |
| LLM 1인분 사전값 | `depth/llm_prior/llm_serving38.py` | Gemini 3.8-flash 에 음식명만 주고 1인분 [kcal, 지방, 단백질, 탄수화물, 중량] 추정 |
| DB 음식 목록 | `depth/llm_prior/fewshot_and_db.py` | 052 영양성분 DB 에서 test 음식과 이름이 겹치는 항목을 뺀 1,266종 |
| DB 음식 사전값·임베딩 | `depth/decomp/prep_db.py` | DB 음식의 LLM 1인분 + 음식명 Gemini 임베딩 |
| **25-d 학습 (Stage 3)** | **`depth/decomp/train_decomp_d.py`** | 분해형 헤드 학습, fold 1개 |
| fold 작업자 | `depth/decomp/worker_d.sh` | fold 0~4 학습 후 앙상블 평가 |
| 앙상블 평가 | `depth/ensemble_eval.py` | 5-fold 로그 평균, A/B/G |
| 실행 순서 | `best_unseen_25d/run_25d.sh` | 위 단계를 순서대로 |

## 모델 구조 (Stage 3)

```
사진 ─ ConvNeXt-B / 주파수 / 깊이 → 게이트 → 특징 1536 ─→ 배율 헤드 → s (log 배율, ±3 제한)
                                                  └→ 분량 등급 보조 CE (Q1~Q5, 가중치 0.5)
음식명 ─ Gemini 임베딩 3072 → 텍스트 64 ─┐
LLM 1인분 5개 → log1p·표준화 → MLP 64 ─┴→ δ 헤드 → log1p(1인분) = log1p(LLM 값) + δ
                                          (LLM 값이 없거나 30% 드롭되면 자유 1인분 헤드)
최종 = 1인분 × exp(s)
```

손실
- 영양값 5개 SmoothL1 (log1p)
- 배율 직접 지도: MSE(s, log 정답 배율), 122 분량 사진은 log(0.25~1.25), 122 분류 369종 사진은 0 (가중치 1.0)
- δ 크기 벌점 1.0·mean(δ²) — 이것이 25-c 와의 유일한 차이
- 052 DB 음식(사진 없음) 128개/스텝으로 1인분 헤드만 추가 학습 (가중치 1.0)
- 분량 등급 보조 CE 0.5

## 학습 데이터

| 단계 | 데이터 | 규모 | 비고 |
|---|---|---|---|
| Stage 0 | kFood 분류 140종 | 70,000장 | 백본 분류 사전학습만 (영양 라벨 없음). test 23종과 이름 겹침 없음 |
| Stage 1 | OmniFood8K (중식) | train 5,101장 | 영양 회귀 |
| Stage 2 | AIHub 122 음식분류 369종 (1인분 사진) | train 70,110장 | test·검증 31종 제외 |
| Stage 3 (25-d) | AIHub 122 양추정 사진, 학습 음식 50종 중 fold 의 검증 음식 제외 | fold 당 10,840~11,356장 | 정답 = DB 1인분 × 등급 배율 (Q1~Q5 = 0.25~1.25) |
| Stage 3 리허설 | 122 분류 369종 (비중 0.3) + OmniFood train 앞 500장 (비중 0.1) | – | 잊지 않도록 섞어 학습 |
| Stage 3 보조 | 052 통합영양성분 DB 음식 (사진 없음) | fold 당 1,234~1,250종 | test 음식과 이름이 겹치는 항목, 그 fold 검증 음식 제외 |
| 외부 지식 | Gemini 3.8-flash 1인분 추정 (음식명만 입력) | 122 음식 + DB 음식 | 정답 라벨을 프롬프트에 넣지 않음 |
| 외부 지식 | Gemini 임베딩 (음식명 + 재료) | `text_baseline_the_best/emb_cache_with_ingredients.json` | |
| 깊이 | UniDepth V2 단안 깊이 | `/mnt/data/food_leehogwang/depth_cache_unidepthv2` | |

평가 분할
- test: `foodlist_test_122.txt` (25종 중 사진이 있는 23종, 5,884장) — 모든 학습 단계에서 제외
- 검증: `foodlist_folds_122.json` 의 음식 단위 층화 5-fold (fold 당 11~12종)

데이터 위치
- AIHub 122: `/mnt/data/food_leehogwang/122.음식_분류를_위한_음식종류_및_양에_따른_칼로리_데이터셋(재료,_양념,_완제품_등)`
- OmniFood8K: `/mnt/data/food_leehogwang/omnifood8k`
- 052 영양성분 DB: `052.건강관리를_위한_음식_이미지_데이터/통합_영양성분DB.xlsx`
- kFood: `data/kfood_pretrain_manifest.parquet`

## 산출물 위치 (git 밖)

| 항목 | 경로 |
|---|---|
| Stage 2 시작 체크포인트 (⑧) | `experiments/10_depth/with_depth/checkpoint_stage2_122_std.pth` |
| LLM 1인분 사전값 | `experiments/24b_llm_prior_38flash/llm_serving.json` |
| DB 음식 목록·사전값·임베딩 | `experiments/25_llm_fewshot_prior/db052_foods.json`, `experiments/25_decomp_db_prior/db_prior.json`, `db_emb.json` |
| fold 별 모델·예측 | `experiments/25d_decomp_delta_l2/fold{0..4}/model_stage3_final.pth`, `test_preds.npz` |
| 앙상블 결과 | `experiments/25d_decomp_delta_l2/ensemble_results.json` |

(`experiments/` 는 저장소 루트 `/home/leehogwang/Food-Nutrition-Estimation/experiments/`)

## 해석할 때 주의

1. **음식명을 정답으로 입력받는다.** 사진에서 음식명을 예측해야 하는 실사용에서는 인식 오차가 더해진다(아직 측정하지 않음).
2. **LLM 이 표준 DB 를 외웠을 수 있다.** 정답 라벨은 쓰지 않았으므로 누수는 아니지만, 공개 지식에 기댄 결과다.
3. LLM 1인분 사전값은 test 음식 이름에 대해서도 만든다(추론 입력이 음식명뿐이므로 필요). 숫자 라벨은 쓰지 않는다.
4. 지방은 여전히 가장 나쁘다(G 48.3).
5. Gemini 를 부르는 스크립트(`llm_serving38.py`, `prep_db.py`)는 `.env` 를 첫 `=` 뒤 전체로 읽는다. `.env` 에 키가 여러 줄이면 키가 깨지므로 `GEMINI_KEY` 줄만 읽도록 고쳐 써야 한다. 학습 자체는 캐시 JSON 만 읽고 Gemini 를 부르지 않는다.
