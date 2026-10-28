#!/usr/bin/env python3
"""raw RGBD 5-fold 중 단일 fold 학습 (최종 앙상블용).

fold 분할은 build_folds() 재사용(RGB/cutout k-fold와 동일 seed) → 계열 간 비교 가능.
사용: CUDA_VISIBLE_DEVICES=1 python scripts/train_raw_rgbd_fold.py --fold 1
산출: runs/raw_rgbd_kfold/fold{K}/best.pt, train_log.csv
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from datasets.nutrition5k_rgb_dataset import load_gt_dict, TARGETS
from datasets.nutrition5k_raw_rgbd_dataset import Nutrition5kRawRGBDDataset
from datasets.nutrition5k_pred_rgbd_dataset import Nutrition5kPredRGBDDataset
from scripts.train_cutout_resnet101_se import run_epoch, BATCH, LR, LR_DECAY, WORKERS, SEED
from scripts.train_cutout_fold import build_folds, N_FOLDS
from scripts.train_cutout_rgbd import ResNet101SE_RGBD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)      # 1..5
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--min-epochs", type=int, default=150)
    ap.add_argument("--pred", action="store_true", help="예측 depth(Depth Anything) 사용")
    args = ap.parse_args()
    k = args.fold - 1
    DS = Nutrition5kPredRGBDDataset if args.pred else Nutrition5kRawRGBDDataset
    run_name = "pred_rgbd_kfold" if args.pred else "raw_rgbd_kfold"

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gt = load_gt_dict()
    folds = build_folds()
    val_ids   = folds[k].tolist()
    train_ids = np.concatenate([folds[i] for i in range(N_FOLDS) if i != k]).tolist()
    print(f"[raw fold {args.fold}] device={device} train={len(train_ids)} val={len(val_ids)}", flush=True)

    train_ds = DS(train_ids, gt, augment=True)
    val_ds   = DS(val_ids,   gt, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH,   shuffle=True,  num_workers=WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH*4, shuffle=False, num_workers=WORKERS, pin_memory=True)

    model = ResNet101SE_RGBD(len(TARGETS)).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=LR_DECAY)

    fold_dir = ROOT / "runs" / run_name / f"fold{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    log_path = fold_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])
    best, best_ep, noimp = float("inf"), 0, 0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        trl = run_epoch(model, train_loader, opt, device, train=True)
        vll = run_epoch(model, val_loader,   opt, device, train=False)
        sched.step()
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, round(trl, 5), round(vll, 5)])
        if vll < best:
            best, best_ep, noimp = vll, ep, 0
            torch.save({"epoch": ep, "model_state": model.state_dict(), "val_loss": vll}, fold_dir / "best.pt")
        else:
            noimp += 1
        if ep % 20 == 0 or ep == args.epochs:
            print(f"[raw fold {args.fold}] ep{ep:03d} train={trl:.4f} val={vll:.4f} best={best:.4f}@{best_ep} noimp={noimp} {time.time()-t0:.0f}s", flush=True)
        if ep >= args.min_epochs and noimp >= args.patience:
            print(f"[raw fold {args.fold}] EARLY STOP at ep{ep}", flush=True)
            break

    print(f"[raw fold {args.fold}] DONE best_val={best:.4f}@ep{best_ep}", flush=True)


if __name__ == "__main__":
    main()
