#!/bin/bash
# 122 텍스트 = 파이프라인 형식 "음식명 (재료: …)" 로 영양 모델 4런 재학습 (직접 예측 ×2, 1인분 Q4 ×2). 출력 runs/<split>[_q4]_ing.
cd /home/leehogwang/food_b/experiments/split_txt_3runs
PY=/home/leehogwang/Food-Nutrition-Estimation/.venv/bin/python; L=logs
$PY -u src/train.py --split 122_1_forseen --text ing --device cuda:0 --budget_min 180 > $L/122_1_forseen_ing.log 2>&1 &
$PY -u src/train.py --split 122_2_forunseen --text ing --target q4 --device cuda:0 --budget_min 150 > $L/122_2_forunseen_q4_ing.log 2>&1 &
$PY -u src/train.py --split 122_1_forseen --text ing --target q4 --device cuda:1 --budget_min 180 > $L/122_1_forseen_q4_ing.log 2>&1 &
$PY -u src/train.py --split 122_2_forunseen --text ing --device cuda:1 --budget_min 150 > $L/122_2_forunseen_ing.log 2>&1 &
wait; echo done > $L/ING_DONE
