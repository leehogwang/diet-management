#!/bin/bash
# 양추정 처음 보는 음식 일반화 (122_2_forunseen): GRL(음식 정체성 제거, λ=1) + 강한 증강 (cuda:0), 강한 증강만 (cuda:1)
# → 파이프라인 대분류로 test 재예측 → compare_quantity_soft.py 로 G4 비교
cd "$(dirname "$0")"; PY=/home/leehogwang/Food-Nutrition-Estimation/.venv/bin/python; L=logs; S=122_2_forunseen
run() { $PY -u src/train_quantity.py --split $S --device $1 $2 --tag $3 > $L/${S}_quantity$3.log 2>&1 \
        && $PY src/eval_quantity_cats.py --split $S --device $1 --tag $3 >> $L/${S}_quantity$3.log 2>&1; }
run cuda:0 "--grl 1.0 --aug strong" _grlaug & run cuda:1 "--aug strong" _aug & wait
$PY src/compare_quantity_soft.py --split $S > $L/compare_quantity_grl.log 2>&1; echo GRL_DONE
