#!/bin/bash
# 실험 25-d 재현 순서 (이 폴더 안에서 완결). 산출물이 이미 있는 단계는 건너뛴다.
# 실행: setsid nohup bash run_25d.sh 0 > run_25d.log 2>&1 &   (인자 = GPU 번호)
# 환경변수: PY(학습용 python), UPY(UniDepth 전용 python), STAGE0(kFood 백본 체크포인트), DEPTH_DIR(깊이 캐시)
set -e
GPU=${1:-0}
B=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}; UPY=${UPY:-python}
E=$B/experiments/10_depth/with_depth
STAGE0=${STAGE0:-$B/checkpoints_curriculum_4stages/checkpoint_stage0_backbone.pth}
cd $B; mkdir -p $E

# 1) UniDepth V2 깊이 캐시 (precompute_depth.py 의 OUT 경로, 기본 /mnt/data/food_leehogwang/depth_cache_unidepthv2)
[ -f depth/depth_manifest.jsonl ] || $PY depth/build_depth_manifest.py
[ -f ${DEPTH_DIR:-/mnt/data/food_leehogwang/depth_cache_unidepthv2}/depth_all.npy ] || {
  $UPY depth/precompute_depth.py --device cuda:$GPU --shard 0 --nshards 1
  $UPY depth/precompute_depth.py --merge; }

# 2) Stage 1 (OmniFood8K) → Stage 2 (122 분류 369종). Stage 0 kFood 백본은 STAGE0 체크포인트
[ -f $E/checkpoint_stage1_omnifood.pth ] || $PY -u depth/train_depth.py --stage 1 --out_dir $E --portion_aux 0.5 --backbone_path $STAGE0 --device cuda:$GPU
[ -f $E/checkpoint_stage2_122_std.pth ] || $PY -u depth/train_depth.py --stage 2 --out_dir $E --portion_aux 0.5 --resume $E/checkpoint_stage1_omnifood.pth --device cuda:$GPU

# 3) LLM 1인분 사전값 · DB 음식 (Gemini, 환경변수 GEMINI_KEY 또는 .env 의 GEMINI_KEY 줄). 작은 결과 JSON 은 experiments/ 에 이미 포함
[ -f experiments/24b_llm_prior_38flash/llm_serving.json ] || $PY -u depth/llm_prior/llm_serving38.py
[ -f experiments/25_llm_fewshot_prior/db052_foods.json ] || $PY -u depth/llm_prior/fewshot_and_db.py
[ -f experiments/25_decomp_db_prior/db_emb.json ] || $PY -u depth/decomp/prep_db.py   # db_emb.json(53MB)은 저장소에 없어 생성 필요

# 4) Stage 3: 25-d 5-fold 학습 → 앙상블 평가 (experiments/25d_decomp_delta_l2/)
PY=$PY bash depth/decomp/worker_d.sh $GPU
