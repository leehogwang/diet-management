#!/bin/bash
# open-set 판정(cls_osr_*_baseline route)으로 파이프라인 재평가: pipe_osr 조건 추가 (기존 pipe 와 비교)
set -e
cd /home/leehogwang/food_b/experiments/split_txt_3runs
PY=/home/leehogwang/Food-Nutrition-Estimation/.venv/bin/python; L=logs
$PY src/train_classifier_osr.py --split 122_2_forunseen --baseline --device cuda:0 > $L/cls_osr_122_2_forunseen_baseline_route.log 2>&1 &
$PY src/train_classifier_osr.py --split 122_1_forseen --baseline --device cuda:1 > $L/cls_osr_122_1_forseen_baseline_route.log 2>&1 &
wait; grep -h "^route" $L/cls_osr_*_baseline_route.log; echo "[1] 분류기 판정 완료"
$PY src/gemini_ing.py --names pipe_osr --workers 16 2>&1 | grep -v AFC > $L/gemini_ing_pipe_osr.log; tail -n 2 $L/gemini_ing_pipe_osr.log
$PY src/embed_ing.py > $L/embed_ing_pipe_osr.log 2>&1; tail -n 1 $L/embed_ing_pipe_osr.log; echo "[2] 재료·대분류·임베딩 완료"
for s in 122_1_forseen 122_2_forunseen; do $PY src/eval_quantity_cats.py --split $s --device cuda:1 2>&1 | grep -E "대분류|Error|Traceback"; done; echo "[3] 양추정 재평가 완료"
for r in "122_1_forseen actual" "122_1_forseen q4" "122_2_forunseen actual" "122_2_forunseen q4"; do set -- $r
  $PY -u src/train.py --split $1 --target $2 --text ing --cat --eval_only --device cuda:0 2>&1 | grep -E "EVAL_ONLY|Error|Traceback" | sed "s/^/$1 $2 /"; done
echo "[4] 영양 재평가 완료"
$PY src/compare_pipeline_cat.py 2>&1 | tail -n 4; echo PIPE_OSR_DONE
