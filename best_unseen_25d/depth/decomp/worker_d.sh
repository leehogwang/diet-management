#!/bin/bash
# 실험 25-d fold 작업자: fold 를 하나씩 가져가 학습 → 5개 끝나면 앙상블 평가. 인자: GPU 번호.
# 같은 OUT 으로 여러 GPU 에서 동시에 실행하면 fold 를 나눠 가진다 (claim 폴더로 원자적 선점).
GPU=$1
B=$(cd "$(dirname "$0")/../.." && pwd)          # best_unseen_25d
PY=${PY:-python}
OUT=${OUT:-$B/experiments/25d_decomp_delta_l2}
cd $B; mkdir -p $OUT
for k in 0 1 2 3 4; do
  mkdir $OUT/claim_fold$k 2>/dev/null || continue
  mkdir -p $OUT/fold$k
  $PY -u depth/decomp/train_decomp_d.py --fold $k --out_dir $OUT/fold$k --device cuda:$GPU > $OUT/fold$k/train.log 2>&1 || echo "fold $k FAILED gpu$GPU" >> $OUT/launch.log
done
if [ $(ls $OUT/fold*/test_result.json 2>/dev/null | wc -l) -eq 5 ] && mkdir $OUT/claim_eval 2>/dev/null; then
  $PY -u depth/ensemble_eval.py --root $OUT > $OUT/ensemble_eval.log 2>&1; echo DONE > $OUT/DONE
fi
