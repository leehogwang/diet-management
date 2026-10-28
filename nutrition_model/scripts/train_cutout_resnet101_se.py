#!/usr/bin/env python3
"""
ResNet101 + SE 모듈, cutout 이미지 기반 영양소 회귀.

논문 설정 재현:
  - 입력: U-Net cutout (배경=0) 이미지
  - 백본: ResNet101 + Squeeze-and-Excitation
  - 출력: kcal, mass_g, fat_g, carb_g, protein_g (5타겟)
  - 손실: normalized MAE (논문 수식)
  - 옵티마이저: Adam, LR exp decay (0.99)
  - 배치: 8, 에포크: 300

산출물:
  runs/cutout_resnet101_se/best.pt
  outputs/cutout_resnet101_se_results.json
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from datasets.nutrition5k_rgb_dataset import load_gt_dict, TARGETS
from datasets.nutrition5k_cutout_dataset import Nutrition5kCutoutDataset, load_cutout_split_ids

EPOCHS   = 300
BATCH    = 8
LR       = 1e-4
LR_DECAY = 0.99
WORKERS  = 4
SEED     = 42
VAL_RATIO = 0.1
RUN_DIR  = ROOT / "runs" / "cutout_resnet101_se"
OUT_JSON = ROOT / "outputs" / "cutout_resnet101_se_results.json"


# ── SE 모듈 ──────────────────────────────────────────────────────

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=[2, 3])           # (B, C)
        scale = self.fc(scale).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * scale


# ── 모델 ─────────────────────────────────────────────────────────

class ResNet101SE(nn.Module):
    def __init__(self, n_targets: int = 5):
        super().__init__()
        base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.se      = SEBlock(2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        f = self.se(f)
        f = self.avgpool(f).flatten(1)
        return self.head(f)


# ── 손실 함수 ────────────────────────────────────────────────────

def normalized_mae_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """논문 정규화 MAE: 각 타겟별 |pred-gt|의 합 / gt의 합."""
    loss = torch.zeros(1, device=pred.device)
    for j in range(pred.size(1)):
        loss = loss + (pred[:, j] - gt[:, j]).abs().sum() / (gt[:, j].sum().clamp(min=1e-8))
    return loss


# ── 지표 ─────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    result = {}
    for j, t in enumerate(TARGETS):
        p, g = pred[:, j], gt[:, j]
        mask = g > 0
        pm, gm = p[mask], g[mask]
        if len(gm) == 0:
            result[t] = {}
            continue
        pmae    = float(np.mean(np.abs(pm - gm) / gm * 100))
        mae     = float(np.mean(np.abs(pm - gm)))
        smape   = float(np.mean(2 * np.abs(pm - gm) / (np.abs(pm) + np.abs(gm) + 1e-8) * 100))
        log_err = float(np.mean(np.abs(np.log1p(pm.clip(0)) - np.log1p(gm))))
        result[t] = {
            "pmae": round(pmae, 2), "mae": round(mae, 3),
            "smape": round(smape, 2), "log_err": round(log_err, 4),
            "n": int(mask.sum()),
        }
    pmaes    = [result[t]["pmae"]  for t in TARGETS if result[t]]
    smapes   = [result[t]["smape"] for t in TARGETS if result[t]]
    log_errs = [result[t]["log_err"] for t in TARGETS if result[t]]
    result["mean"] = {
        "pmae":    round(float(np.mean(pmaes)),    2),
        "smape":   round(float(np.mean(smapes)),   2),
        "log_err": round(float(np.mean(log_errs)), 4),
    }
    return result


# ── 학습/평가 루프 ────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).clamp(min=0)
            loss = normalized_mae_loss(pred, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * x.size(0)
            n     += x.size(0)
    return total / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_gt = [], []
    for x, y in loader:
        pred = model(x.to(device)).clamp(min=0).cpu()
        all_pred.append(pred); all_gt.append(y)
    return torch.cat(all_pred).numpy(), torch.cat(all_gt).numpy()


# ── 메인 ─────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  epochs={EPOCHS}  |  batch={BATCH}  |  ResNet101+SE  |  cutout input")

    gt_dict = load_gt_dict()
    train_ids, test_ids = load_cutout_split_ids()

    # train → train/val 분리
    random.shuffle(train_ids)
    n_val     = int(len(train_ids) * VAL_RATIO)
    val_ids   = train_ids[:n_val]
    train_ids = train_ids[n_val:]

    print(f"train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    train_ds = Nutrition5kCutoutDataset(train_ids, gt_dict, augment=True)
    val_ds   = Nutrition5kCutoutDataset(val_ids,   gt_dict, augment=False)
    test_ds  = Nutrition5kCutoutDataset(test_ids,  gt_dict, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH,   shuffle=True,  num_workers=WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH*4, shuffle=False, num_workers=WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH*4, shuffle=False, num_workers=WORKERS, pin_memory=True)

    model     = ResNet101SE(n_targets=len(TARGETS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_DIR / "train_log.csv"
    best_val_loss = float("inf")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_pmae", "val_smape"])

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step()

        if epoch % 10 == 0 or epoch == EPOCHS:
            pred_np, gt_np = evaluate(model, val_loader, device)
            m = compute_metrics(pred_np, gt_np)
            val_pmae  = m["mean"]["pmae"]
            val_smape = m["mean"]["smape"]
            print(f"  [{epoch:03d}/{EPOCHS}] train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"PMAE={val_pmae:.1f}%  SMAPE={val_smape:.1f}%  {time.time()-t0:.0f}s")
        else:
            val_pmae, val_smape = 0.0, 0.0
            print(f"  [{epoch:03d}/{EPOCHS}] train={train_loss:.4f}  val={val_loss:.4f}  {time.time()-t0:.0f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(train_loss,5), round(val_loss,5),
                                    round(val_pmae,2), round(val_smape,2)])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": val_loss}, RUN_DIR / "best.pt")

    # 최종 test 평가
    print(f"\n{'='*60}")
    print("최종 Test 평가 (best checkpoint)")
    print(f"{'='*60}")
    ckpt = torch.load(RUN_DIR / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    pred_np, gt_np = evaluate(model, test_loader, device)
    metrics = compute_metrics(pred_np, gt_np)

    for t in TARGETS:
        m = metrics[t]
        print(f"  {t:<10}  PMAE={m['pmae']:5.1f}%  SMAPE={m['smape']:5.1f}%  MAE={m['mae']:.2f}")
    print(f"  {'mean':<10}  PMAE={metrics['mean']['pmae']:5.1f}%  SMAPE={metrics['mean']['smape']:5.1f}%")

    OUT_JSON.parent.mkdir(exist_ok=True)
    result = {
        "model": "ResNet101+SE",
        "input": "cutout",
        "epochs": EPOCHS,
        "best_epoch": ckpt["epoch"],
        "test_metrics": metrics,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n결과 저장: {OUT_JSON}")


if __name__ == "__main__":
    main()
