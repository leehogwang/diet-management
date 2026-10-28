#!/bin/bash
# stage3_3 파이프라인(음식 분류기 → Gemini 음식명·재료 → Gemini 임베딩 → 영양 모델 cls109 × 양 비율)으로
# Nutrition5k test 709 접시(접시당 사진 1장, selected_images.csv)를 처리하고 평가한다.
# Gemini 무료 일일 한도에 걸리면 pipeline_raw.jsonl 에 저장된 데까지 두고 30분마다 다시 시도한다 (한도는 16:00 KST 초기화).
cd /home/students/cs/202321024/2025_2/2025_2-2/2025-2/n296_stage3_3
PY=$HOME/.conda/envs/diet/bin/python
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU:-0}
O=runs/eval_n5k
L=$O/run.log
IMGS=$(tail -n +2 $O/selected_images.csv | cut -d, -f3)
N=$(echo "$IMGS" | wc -l)
for i in $(seq 1 200); do
  echo "=== 시도 $i $(date '+%F %T')" >> $L
  $PY -m n296.pipeline_infer --config configs/cls109_nutrition_n5k.yaml \
      --nutrition-ckpt runs/nutrition_cls109/model_stage3_296_final.pth \
      --images $IMGS --out-dir $O >> $L 2>&1
  echo "=== exit=$?" >> $L
  if grep -q "($N/$N장)" <(tail -3 $L); then
    $PY $O/eval_n5k_stage3_3.py > $O/eval.log 2>&1
    echo "=== eval exit=$? $(date '+%F %T')" >> $L
    echo "=== ALL_DONE" >> $L
    exit 0
  fi
  sleep 1800
done
