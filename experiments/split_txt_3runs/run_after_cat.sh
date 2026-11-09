#!/bin/bash
# 대분류 실험 후처리: 분류기 → 파이프라인 음식명 Gemini 재료·대분류 → 임베딩 → 양추정 대분류 재평가 → 영양 test 재평가 → 비교.
cd /home/leehogwang/food_b/experiments/split_txt_3runs
PY=/home/leehogwang/Food-Nutrition-Estimation/.venv/bin/python; L=logs
until grep -q "^DONE" $L/embed_ing.log 2>/dev/null; do sleep 30; done; echo "[1] Gemini 음식명 재료·임베딩 완료"
until grep -q "^DONE" $L/cls_122_1_forseen.log && grep -q "^DONE" $L/cls_122_2_forunseen.log; do sleep 30; done; echo "[2] 분류기 2개 완료"
$PY src/gemini_ing.py --names pipe --workers 16 2>&1 | grep -v AFC > $L/gemini_ing_pipe.log; tail -n 2 $L/gemini_ing_pipe.log
$PY src/embed_ing.py > $L/embed_ing_pipe.log 2>&1; tail -n 1 $L/embed_ing_pipe.log; echo "[3] 파이프라인 음식명 재료·임베딩 완료"
for s in 122_1_forseen 122_2_forunseen; do $PY src/eval_quantity_cats.py --split $s --device cuda:1 2>&1 | grep -E "대분류|Error|Traceback"; done; echo "[4] 양추정 재평가 완료"
until [ -f $L/ING_CAT_DONE ]; do sleep 60; done; echo "[5] 영양 4런 완료"
for r in "122_1_forseen actual" "122_1_forseen q4" "122_2_forunseen actual" "122_2_forunseen q4"; do set -- $r
  $PY -u src/train.py --split $1 --target $2 --text ing --cat --eval_only --device cuda:0 2>&1 | grep -E "EVAL_ONLY|Error|Traceback" | sed "s/^/$1 $2 /"; done
$PY src/compare_pipeline_cat.py 2>&1 | tail -n 4; echo ALL_CAT_DONE
