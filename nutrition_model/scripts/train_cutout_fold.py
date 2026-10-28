#!/usr/bin/env python3
"""cutout ResNet101+SE 5-fold 중 단일 fold 학습 (GPU 병렬용).

사용: CUDA_VISIBLE_DEVICES=1 python scripts/train_cutout_fold.py --fold 1 --epochs 300
산출: runs/cutout_resnet101_se_kfold/fold{K}/best.pt, train_log.csv
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from datasets.nutrition5k_rgb_dataset import load_gt_dict, TARGETS
from datasets.nutrition5k_cutout_dataset import Nutrition5kCutoutDataset, load_cutout_split_ids
from scripts.train_cutout_resnet101_se import (
    ResNet101SE, normalized_mae_loss, compute_metrics, run_epoch, evaluate,
    BATCH, LR, LR_DECAY, WORKERS, SEED,
)

N_FOLDS = 5


def build_folds():
    """official train split ∩ cutout 을 고정 seed로 5분할 (모든 프로세스 동일 결과)."""
    train_ids, _ = load_cutout_split_ids()
    ids = sorted(train_ids)                     # 결정적 입력 순서
    rng = np.random.RandomState(SEED)
    rng.shuffle(ids)
    return np.array_split(np.array(ids), N_FOLDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)      # 1..5
    ap.add_argument("--epochs", type=int, default=300)      # 상한
    ap.add_argument("--patience", type=int, default=40)     # early stop
    ap.add_argument("--min-epochs", type=int, default=150)
    args = ap.parse_args()
    k = args.fold - 1

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gt = load_gt_dict()
    folds = build_folds()
    val_ids   = folds[k].tolist()
    train_ids = np.concatenate([folds[i] for i in range(N_FOLDS) if i != k]).tolist()
    print(f"[fold {args.fold}] device={device} train={len(train_ids)} val={len(val_ids)} epochs={args.epochs}", flush=True)

    train_ds = Nutrition5kCutoutDataset(train_ids, gt, augment=True)
    val_ds   = Nutrition5kCutoutDataset(val_ids,   gt, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH,   shuffle=True,  num_workers=WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH*4, shuffle=False, num_workers=WORKERS, pin_memory=True)

    model = ResNet101SE(n_targets=len(TARGETS)).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=LR_DECAY)

    fold_dir = ROOT / "runs" / "cutout_resnet101_se_kfold" / f"fold{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    log_path = fold_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])
    best = float("inf"); best_ep = 0; no_improve = 0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        trl = run_epoch(model, train_loader, opt, device, train=True)
        vll = run_epoch(model, val_loader,   opt, device, train=False)
        sched.step()
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, round(trl, 5), round(vll, 5)])
        if vll < best:
            best = vll; best_ep = ep; no_improve = 0
            torch.save({"epoch": ep, "model_state": model.state_dict(), "val_loss": vll}, fold_dir / "best.pt")
        else:
            no_improve += 1
        if ep % 20 == 0 or ep == args.epochs:
            print(f"[fold {args.fold}] ep{ep:03d} train={trl:.4f} val={vll:.4f} best={best:.4f}@{best_ep} noimp={no_improve} {time.time()-t0:.0f}s", flush=True)
        if ep >= args.min_epochs and no_improve >= args.patience:
            print(f"[fold {args.fold}] EARLY STOP at ep{ep} (no improve {no_improve})", flush=True)
            break

    print(f"[fold {args.fold}] DONE best_val={best:.4f}@ep{best_ep}", flush=True)


if __name__ == "__main__":
    main()
