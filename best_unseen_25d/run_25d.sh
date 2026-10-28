#!/bin/bash
# 실험 25-d 재현 순서. 각 단계 산출물이 이미 있으면 그 줄은 건너뛰어도 된다 (README "산출물 위치").
# 실행: setsid nohup bash best_unseen_25d/run_25d.sh 0 > run_25d.log 2>&1 &   (인자 = GPU 번호)
set -e
GPU=${1:-0}
R=/home/leehogwang/Food-Nutrition-Estimation
W=$R/.claude/worktrees/kfold
PY=$R/.venv/bin/python
UPY=/mnt/data/food_leehogwang/.venv_unidepth/bin/python
E=$R/experiments/10_depth/with_depth
cd $W

# 1) UniDepth V2 깊이 캐시 (depth/depth_manifest.jsonl 기준)
[ -f /mnt/data/food_leehogwang/depth_cache_unidepthv2/depth_all.npy ] || {
  $UPY depth/precompute_depth.py --device cuda:$GPU --shard 0 --nshards 1
  $UPY depth/precompute_depth.py --merge; }

# 2) Stage 1 (OmniFood8K) → Stage 2 (122 분류 369종). Stage 0 kFood 백본은 기존 체크포인트 사용
[ -f $E/checkpoint_stage1_omnifood.pth ] || $PY -u depth/train_depth.py --stage 1 --out_dir $E --portion_aux 0.5 \
  --backbone_path $R/checkpoints_curriculum_4stages/checkpoint_stage0_backbone.pth --device cuda:$GPU
[ -f $E/checkpoint_stage2_122_std.pth ] || $PY -u depth/train_depth.py --stage 2 --out_dir $E --portion_aux 0.5 \
  --resume $E/checkpoint_stage1_omnifood.pth --device cuda:$GPU

# 3) LLM 1인분 사전값 · DB 음식 (Gemini 호출, .env 의 GEMINI_KEY 필요 — README 주의 5)
[ -f $R/experiments/24b_llm_prior_38flash/llm_serving.json ] || $PY -u depth/llm_prior/llm_serving38.py
[ -f $R/experiments/25_llm_fewshot_prior/db052_foods.json ] || $PY -u depth/llm_prior/fewshot_and_db.py
[ -f $R/experiments/25_decomp_db_prior/db_prior.json ] || $PY -u depth/decomp/prep_db.py

# 4) Stage 3: 25-d 5-fold 학습 → 앙상블 평가 (experiments/25d_decomp_delta_l2/)
bash depth/decomp/worker_d.sh $GPU
