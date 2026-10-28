#!/usr/bin/env python3
"""cutout RGBD (RGB + raw depth 4채널) ResNet101+SE 회귀.

Step 5(RGB-only)와 동일 설정 + depth 4번째 채널. depth가 도움되는지 확인용.
사용: CUDA_VISIBLE_DEVICES=2 python scripts/train_cutout_rgbd.py
산출: runs/cutout_rgbd_resnet101_se/best.pt, outputs/cutout_rgbd_resnet101_se_results.json
"""
from __future__ import annotations
import csv, json, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from datasets.nutrition5k_rgb_dataset import load_gt_dict, TARGETS
from datasets.nutrition5k_cutout_dataset import load_cutout_split_ids
from datasets.nutrition5k_cutout_rgbd_dataset import Nutrition5kCutoutRGBDDataset
from scripts.train_cutout_resnet101_se import (
    SEBlock, normalized_mae_loss, compute_metrics, run_epoch, evaluate,
    BATCH, LR, LR_DECAY, WORKERS, SEED, VAL_RATIO,
)

EPOCHS, PATIENCE, MIN_EPOCHS = 300, 40, 150
RUN_DIR = ROOT / "runs" / "cutout_rgbd_resnet101_se"
OUT_JSON = ROOT / "outputs" / "cutout_rgbd_resnet101_se_results.json"


class ResNet101SE_RGBD(nn.Module):
    """conv1을 4채널로 확장 (RGB 가중치 복사 + depth=RGB평균으로 init)."""
    def __init__(self, n_targets: int = 5):
        super().__init__()
        base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        old = base.conv1
        new = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
        base.conv1 = new
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.se = SEBlock(2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(1024, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, n_targets),
        )

    def forward(self, x):
        f = self.backbone(x); f = self.se(f)
        return self.head(self.avgpool(f).flatten(1))


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | RGBD 4ch | epochs<= {EPOCHS} early-stop(p{PATIENCE})", flush=True)

    gt = load_gt_dict()
    train_ids, test_ids = load_cutout_split_ids()
    random.shuffle(train_ids)                        # Step 5와 동일 seed/순서 → 동일 val
    n_val = int(len(train_ids) * VAL_RATIO)
    val_ids, train_ids = train_ids[:n_val], train_ids[n_val:]
    print(f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}", flush=True)

    tr = Nutrition5kCutoutRGBDDataset(train_ids, gt, augment=True)
    va = Nutrition5kCutoutRGBDDataset(val_ids, gt, augment=False)
    te = Nutrition5kCutoutRGBDDataset(test_ids, gt, augment=False)
    trl = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=WORKERS, pin_memory=True)
    val = DataLoader(va, batch_size=BATCH*4, shuffle=False, num_workers=WORKERS, pin_memory=True)
    tel = DataLoader(te, batch_size=BATCH*4, shuffle=False, num_workers=WORKERS, pin_memory=True)

    model = ResNet101SE_RGBD(len(TARGETS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=LR_DECAY)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log = RUN_DIR / "train_log.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])
    best, best_ep, noimp = float("inf"), 0, 0

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tl = run_epoch(model, trl, opt, device, train=True)
        vl = run_epoch(model, val, opt, device, train=False)
        sched.step()
        with open(log, "a", newline="") as f:
            csv.writer(f).writerow([ep, round(tl, 5), round(vl, 5)])
        if vl < best:
            best, best_ep, noimp = vl, ep, 0
            torch.save({"epoch": ep, "model_state": model.state_dict(), "val_loss": vl}, RUN_DIR / "best.pt")
        else:
            noimp += 1
        if ep % 10 == 0 or ep == EPOCHS:
            print(f"[rgbd] ep{ep:03d} train={tl:.4f} val={vl:.4f} best={best:.4f}@{best_ep} noimp={noimp} {time.time()-t0:.0f}s", flush=True)
        if ep >= MIN_EPOCHS and noimp >= PATIENCE:
            print(f"[rgbd] EARLY STOP ep{ep}", flush=True); break

    ckpt = torch.load(RUN_DIR / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    P, G = evaluate(model, tel, device)
    m = compute_metrics(P, G)
    # 논문식 PMAE 추가
    paper = {}
    for j, t in enumerate(TARGETS):
        g = G[:, j]
        paper[t] = round(float(np.abs(P[:, j]-g).mean() / g.mean() * 100), 2)
    paper["mean"] = round(float(np.mean([paper[t] for t in TARGETS])), 2)
    print("\n=== RGBD test (best ep%d) ===" % ckpt["epoch"])
    for t in TARGETS:
        print(f"  {t:<10} paperPMAE={paper[t]:5.1f}%  perSmpPMAE={m[t]['pmae']:5.1f}%  SMAPE={m[t]['smape']:5.1f}%")
    print(f"  {'mean':<10} paperPMAE={paper['mean']:5.2f}%  perSmpPMAE={m['mean']['pmae']:5.2f}%")
    print("  참고: RGB-only(Step5) test paperPMAE 22.68% / 논문 17.06%")

    OUT_JSON.write_text(json.dumps(
        {"model": "ResNet101+SE RGBD(4ch)", "best_epoch": ckpt["epoch"],
         "test_paper_pmae": paper, "test_metrics": m}, ensure_ascii=False, indent=2))
    print(f"\n저장: {OUT_JSON}")


if __name__ == "__main__":
    main()
