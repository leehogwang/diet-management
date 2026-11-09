#!/bin/bash
# 양추정 GRL 후속 (122_2_forunseen): λ 0.5 / 2.0 (+강한 증강, cuda:0 두 런), GRL λ1.0 만 (기존 증강, cuda:1)
# → 파이프라인 대분류로 test 재예측 → compare_quantity_soft.py
cd "$(dirname "$0")"; PY=/home/leehogwang/Food-Nutrition-Estimation/.venv/bin/python; L=logs; S=122_2_forunseen
run() { $PY -u src/train_quantity.py --split $S --device $1 --workers 8 $2 --tag $3 > $L/${S}_quantity$3.log 2>&1 \
        && $PY src/eval_quantity_cats.py --split $S --device $1 --tag $3 >> $L/${S}_quantity$3.log 2>&1; }
run cuda:0 "--grl 0.5 --aug strong" _grl05aug & run cuda:0 "--grl 2.0 --aug strong" _grl20aug & run cuda:1 "--grl 1.0" _grl & wait
$PY src/compare_quantity_soft.py --split $S > $L/compare_quantity_grl2.log 2>&1; echo GRL2_DONE
