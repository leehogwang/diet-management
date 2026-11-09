#!/bin/bash
# open-set 분류기: ArcFace 학습 + 기존 CE 분류기 같은 기준으로 채점
cd /home/leehogwang/food_b/experiments/split_txt_3runs
PY=/home/leehogwang/Food-Nutrition-Estimation/.venv/bin/python; L=logs
( $PY src/train_classifier_osr.py --split 122_2_forunseen --device cuda:0 > $L/cls_osr_122_2_forunseen.log 2>&1 ) &
( $PY src/train_classifier_osr.py --split 122_2_forunseen --baseline --device cuda:1 > $L/cls_osr_122_2_forunseen_baseline.log 2>&1
  $PY src/train_classifier_osr.py --split 122_1_forseen --baseline --device cuda:1 > $L/cls_osr_122_1_forseen_baseline.log 2>&1
  $PY src/train_classifier_osr.py --split 122_1_forseen --device cuda:1 > $L/cls_osr_122_1_forseen.log 2>&1 ) &
wait; echo OSR_ALL_DONE
