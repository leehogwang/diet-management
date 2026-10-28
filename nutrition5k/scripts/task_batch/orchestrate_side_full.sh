#!/usr/bin/env bash
# side 3체인 정식 승격 (2026-09-10) — 시드당 GPU 1장, 3시드 병렬
#
# 8/22 파일럿과의 차이 두 가지뿐:
#   ① 단계0 에폭 상한 60 → 200 (파일럿은 ep58 best/60 상한에 잘렸다 = 아직 내려가는 중이었다)
#   ② 단계0 부터 시드를 갈라 3체인 (체크포인트 공유 시 짝지은 비교 착시 — 함정 7번)
# 나머지 하이퍼파라미터는 파일럿과 100% 동일하다.
#
# 사용법:  bash scripts/task_batch/orchestrate_side_full.sh          (시드 42/1/2 → GPU 0/1/2)
# 중단:    kill $(cat outputs/task_runs/task_side/logs/sidefull_pids.txt)   ← pkill -f 금지(자기매칭)
set -u
cd /home/students/cs/202520906/nutrition5k_pipeline
SIDE=outputs/task_runs/task_side/runs
S1=outputs/task_runs/task4_final/runs
S2=outputs/task_runs/task_b8/runs
LS=outputs/task_runs/task_side/logs
L1=outputs/task_runs/task4_final/logs
L2=outputs/task_runs/task_b8/logs
mkdir -p $LS $L1 $L2
: > $LS/sidefull_pids.txt

chain () {          # $1 = seed, $2 = gpu
  local s=$1 g=$2
  local n0="side_pre200_seed${s}" n1="cs_sidefull_seed${s}" n2="b8_sidefull_s${s}"

  if [ -f "$SIDE/$n0/pretrain_metrics.json" ]; then echo "SKIP  $n0"; else
    echo "START $n0 (단계0 side 사전학습 상한200) gpu$g  $(date +%m-%d_%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$g python -u scripts/task_batch/train_side_pretrain.py \
        --run-name "$n0" --backbone convnext_small --resolution 384 \
        --epochs 200 --patience 12 --batch-size 16 --lr 1e-4 \
        --num-workers 6 --seed "$s" > "$LS/$n0.log" 2>&1
    echo "DONE  $n0  $(date +%m-%d_%H:%M:%S) -> $(grep -h '^DONE' $LS/$n0.log | tail -1)"
  fi

  [ -f "$SIDE/$n0/checkpoints/best.pt" ] || { echo "ABORT seed$s (단계0 없음)"; return 1; }
  if [ -f "$S1/$n1/test_metrics.json" ]; then echo "SKIP  $n1"; else
    echo "START $n1 (단계1) gpu$g  $(date +%m-%d_%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$g python -u scripts/task_batch/train_strong.py \
        --backbone convnext_small --resolution 384 --crop 1 \
        --lam 30 --batch-size 16 --lr 1e-4 --epochs 200 --patience 12 \
        --num-workers 6 --seed "$s" --run-name "$n1" \
        --init-from "$SIDE/$n0/checkpoints/best.pt" > "$L1/$n1.log" 2>&1
    echo "DONE  $n1  $(date +%m-%d_%H:%M:%S) -> $(grep -h '^DONE' $L1/$n1.log | tail -1)"
  fi

  [ -f "$S1/$n1/checkpoints/best.pt" ] || { echo "ABORT seed$s (단계1 없음)"; return 1; }
  if [ -f "$S2/$n2/test_metrics.json" ]; then echo "SKIP  $n2"; return 0; fi
  echo "START $n2 (단계2 RMSE 기준) gpu$g  $(date +%m-%d_%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$g python -m scripts.task_batch.train_b8 \
      --run-name "$n2" --resolution 384 --epochs 150 --patience 10 --lr 2e-5 \
      --batch-size 16 --num-workers 6 --monitor rmse --seed "$s" \
      --init-from "$S1/$n1/checkpoints/best.pt" > $L2/$n2.log 2>&1
  echo "DONE  $n2  $(date +%m-%d_%H:%M:%S) -> $(grep -h '^DONE' $L2/$n2.log | tail -1)"
}

i=0
for s in 42 1 2; do
  chain "$s" "$i" >> "$LS/sidefull_seed${s}.orch.log" 2>&1 &
  echo $! >> $LS/sidefull_pids.txt
  i=$((i+1))
done
wait
echo "ALL_DONE_SIDEFULL  $(date +%m-%d_%H:%M:%S)"
