#!/bin/bash
# 양추정 soft label 비교 (122_2_forunseen): σ=0.5 (cuda:0), σ=1.0 (cuda:1) → 파이프라인 대분류로 test 재예측
cd "$(dirname "$0")"; PY=/home/leehogwang/Food-Nutrition-Estimation/.venv/bin/python; L=logs; S=122_2_forunseen
run() { $PY -u src/train_quantity.py --split $S --device $1 --soft_sigma $2 --tag $3 > $L/${S}_quantity$3.log 2>&1 \
        && $PY src/eval_quantity_cats.py --split $S --device $1 --tag $3 >> $L/${S}_quantity$3.log 2>&1; }
run cuda:0 0.5 _soft05 & run cuda:1 1.0 _soft10 & wait; echo SOFT_DONE
