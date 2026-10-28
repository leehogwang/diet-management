#!/usr/bin/env bash
# 예측 depth RGBD 5-fold 순차 학습 (GPU 1 단독) -> 앙상블 추론 (497+507)
set -e
cd /home/students/cs/202520851/nutrition_pipeline
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/students/cs/202520851/nutrition_pipeline
RUN=runs/pred_rgbd_kfold
mkdir -p "$RUN"

echo "=== pred-depth RGBD 5-fold 시작 $(date) ==="
for k in 1 2 3 4 5; do
  echo "--- fold $k 시작 $(date) ---"
  python scripts/train_raw_rgbd_fold.py --fold $k --pred > "$RUN/fold${k}_run.log" 2>&1
  echo "--- fold $k 완료 $(date) ---"
done

echo "=== 앙상블 추론 $(date) ==="
python scripts/infer_raw_rgbd_kfold_ensemble.py --pred > "$RUN/ensemble_run.log" 2>&1
echo "=== 전체 완료 $(date) ==="
tail -30 "$RUN/ensemble_run.log"
