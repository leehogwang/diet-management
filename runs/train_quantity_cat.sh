#!/bin/bash
# stage3_3 양추정 모델 학습: 최종(전체 78종 + 대분류) → 처음 보는 음식 실험(대분류 / 이미지만)
cd /home/player0070/2025-2/n296_stage3_3
PY="env PYTHONPATH=.pydeps PYTHONUNBUFFERED=1 $HOME/miniconda3/envs/robodiff_cu12/bin/python -m n296.quantity_estimator train --config configs/aihub122.yaml --workers 10"
L=runs/quantity_cat_train.log; : > $L
$PY --out-dir runs/quantity_122_cat >> $L 2>&1; echo "=== final exit=$?" >> $L
$PY --out-dir runs/quantity_holdout_cat --holdout --epochs 12 >> $L 2>&1; echo "=== holdout_cat exit=$?" >> $L
$PY --out-dir runs/quantity_holdout_img --holdout --epochs 12 --no-category >> $L 2>&1; echo "=== holdout_img exit=$?" >> $L
echo "=== ALL_DONE" >> $L
