#!/bin/bash
# 3런 + Gemini 추정. GPU0: 122_1_forseen, GPU1: 122_2_forunseen·n5k 동시. Gemini 추정·임베딩은 학습과 병렬(평가 직전에 다시 읽음).
cd /home/leehogwang/food_b/experiments/split_txt_3runs
PY=/home/leehogwang/Food-Nutrition-Estimation/.venv/bin/python; L=logs; mkdir -p $L
( $PY -u src/gemini.py --workers 16 > $L/gemini.log 2>&1; $PY -u src/embed.py > $L/embed_pred.log 2>&1; echo done > $L/GEMINI_DONE ) &
$PY -u src/train.py --split 122_1_forseen  --device cuda:0 --budget_min 180 > $L/122_1_forseen.log 2>&1 &
$PY -u src/train.py --split 122_2_forunseen --device cuda:1 --budget_min 150 > $L/122_2_forunseen.log 2>&1 &
$PY -u src/train.py --split n5k            --device cuda:1 --budget_min 150 > $L/n5k.log 2>&1 &
wait; echo ALL_DONE > $L/ALL_DONE
