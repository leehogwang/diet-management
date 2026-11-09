#!/bin/bash
# 영양 모델 4런: 텍스트 "음식명 (재료: …)" + 대분류 입력 (--text ing --cat). 출력 runs/<split>[_q4]_ing_cat.
cd /home/leehogwang/food_b/experiments/split_txt_3runs
PY=/home/leehogwang/Food-Nutrition-Estimation/.venv/bin/python; L=logs
$PY -u src/train.py --split 122_1_forseen --text ing --cat --device cuda:0 --budget_min 180 > $L/122_1_forseen_ing_cat.log 2>&1 &
$PY -u src/train.py --split 122_2_forunseen --text ing --cat --target q4 --device cuda:0 --budget_min 150 > $L/122_2_forunseen_q4_ing_cat.log 2>&1 &
$PY -u src/train.py --split 122_1_forseen --text ing --cat --target q4 --device cuda:1 --budget_min 180 > $L/122_1_forseen_q4_ing_cat.log 2>&1 &
$PY -u src/train.py --split 122_2_forunseen --text ing --cat --device cuda:1 --budget_min 150 > $L/122_2_forunseen_ing_cat.log 2>&1 &
wait; echo done > $L/ING_CAT_DONE
