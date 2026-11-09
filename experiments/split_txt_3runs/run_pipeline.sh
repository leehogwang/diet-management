#!/bin/bash
# 파이프라인 비교용 4런: 영양 모델 1인분(Q4) 정답 ×2, 양추정 모델 ×2 (122 두 분할).
cd /home/leehogwang/food_b/experiments/split_txt_3runs
PY=/home/leehogwang/Food-Nutrition-Estimation/.venv/bin/python; L=logs
$PY -u src/train.py --split 122_1_forseen --target q4 --device cuda:0 --budget_min 180 > $L/122_1_forseen_q4.log 2>&1 &
$PY -u src/train_quantity.py --split 122_2_forunseen --device cuda:0 > $L/122_2_forunseen_quantity.log 2>&1 &
$PY -u src/train_quantity.py --split 122_1_forseen --device cuda:1 > $L/122_1_forseen_quantity.log 2>&1 &
$PY -u src/train.py --split 122_2_forunseen --target q4 --device cuda:1 --budget_min 150 > $L/122_2_forunseen_q4.log 2>&1 &
wait; echo done > $L/PIPELINE_DONE
