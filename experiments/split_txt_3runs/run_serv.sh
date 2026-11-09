#!/bin/bash
# 양추정 + Gemini 1인분 중량 (122_2_forunseen): in (cuda:0), reg (cuda:1)
cd "$(dirname "$0")"; PY=/home/leehogwang/Food-Nutrition-Estimation/.venv/bin/python
$PY -u src/train_quantity_serving.py --mode in --device cuda:0 > logs/122_2_forunseen_quantity_serv_in.log 2>&1 &
$PY -u src/train_quantity_serving.py --mode reg --device cuda:1 > logs/122_2_forunseen_quantity_serv_reg.log 2>&1 &
wait; echo SERV_DONE
