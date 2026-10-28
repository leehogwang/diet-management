"""Depth (RGB-D) experiment: does adding the overhead depth channel help?

Controlled ablation — SAME overhead dishes, SAME proven baseline recipe (raw
targets + Softplus + SmoothL1, 224px), the ONLY difference is the depth channel:
  --use-depth 0 : 3-channel RGB-overhead  (control)
  --use-depth 1 : 4-channel RGB + depth   (treatment)

Depth (depth_raw.png, uint16 mm, 0=hole) is per-image z-scored into a relative
height map (the volume cue). ResNet50 conv1 is widened to 4 channels with the
pretrained RGB weights copied and the 4th channel = mean of the RGB filters.

Mass excluded (settled optimal in Task 3/4). 1 overhead frame per dish, so test
eval is naturally per-dish. Reports PMA on the real scale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from PIL import Image

ROOT = Path("/home/students/cs/202520906/nutrition5k_pipeline")
OV = Path("/home/students/cs/202520906/nutrition5k/imagery/realsense_overhead")
sys.path.insert(0, str(ROOT))
from nutrition5k_cbam.data import load_manifest  # noqa: E402
from nutrition5k_cbam.utils import save_json, seed_everything  # noqa: E402

NUTR = ["calories", "fat", "carb", "protein"]
IMN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMN_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# 2026-08-21: 분할 교체용. N5K_MANIFEST_DIR=outputs/manifests_sess 로 세션 분할을 쓴다.
# 기본값은 기존 분할이라 아무것도 지정하지 않으면 동작이 완전히 같다.
import os as _os
MANIFEST_DIR = _os.environ.get("N5K_MANIFEST_DIR", "outputs/manifests")


def dish_targets(split):
    """dish_id -> 4-nutrient vector (1 value per dish)."""
    out = {}
    for it in load_manifest(ROOT / f"{MANIFEST_DIR}/{split}.jsonl"):
        out.setdefault(it["dish_id"], [it["targets"][c] for c in NUTR])
    return out


def _ok(p):
    try:
        return p.exists() and p.stat().st_size > 0
    except OSError:
        return False


def usable_dishes(split):
    """Dishes with non-empty rgb AND depth — same set for both arms (fair)."""
    tg = dish_targets(split)
    return [d for d in tg if _ok(OV / d / "rgb.png") and _ok(OV / d / "depth_raw.png")], tg


class DepthDS(Dataset):
    def __init__(self, dishes, targets, use_depth, res=224, train=False):
        self.dishes = dishes
        self.targets = targets
        self.use_depth = use_depth
        self.res = res
        self.train = train

    def __len__(self):
        return len(self.dishes)

    def __getitem__(self, i):
        d = self.dishes[i]
        rgb = Image.open(OV / d / "rgb.png").convert("RGB").resize((self.res, self.res), Image.BILINEAR)
        rgb = np.asarray(rgb, dtype=np.float32) / 255.0
        flip = self.train and (i * 2654435761 % 2 == 0)  # deterministic-ish per-sample flip
        if self.train:
            # cheap synced augmentation: random horizontal flip
            flip = bool(torch.rand(1).item() < 0.5)
        if flip:
            rgb = rgb[:, ::-1, :].copy()
        rgb = (rgb - IMN_MEAN) / IMN_STD
        chans = [rgb.transpose(2, 0, 1)]  # (3,H,W)

        if self.use_depth:
            try:
                dep = Image.open(OV / d / "depth_raw.png").resize((self.res, self.res), Image.NEAREST)
                dep = np.asarray(dep, dtype=np.float32)  # mm, 0 = hole
            except Exception:
                dep = np.zeros((self.res, self.res), dtype=np.float32)
            mask = dep > 0
            if mask.sum() > 10:
                m, s = dep[mask].mean(), dep[mask].std() + 1e-6
                depn = np.where(mask, (dep - m) / s, 0.0).astype(np.float32)
            else:
                depn = np.zeros_like(dep, dtype=np.float32)
            if flip:
                depn = depn[:, ::-1].copy()
            chans.append(depn[None, :, :])  # (1,H,W)

        x = np.concatenate(chans, axis=0)
        y = torch.tensor(self.targets[d], dtype=torch.float32)
        return torch.from_numpy(x), y, d


class ResNet50RGBD(nn.Module):
    def __init__(self, in_ch=3, pretrained=True, dropout=0.2):
        super().__init__()
        bb = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        if in_ch != 3:
            old = bb.conv1.weight.data  # (64,3,7,7)
            new = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                new.weight[:, :3] = old
                new.weight[:, 3:] = old.mean(dim=1, keepdim=True).repeat(1, in_ch - 3, 1, 1)
            bb.conv1 = new
        nf = bb.fc.in_features
        bb.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(nf, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(512, 4), nn.Softplus(),
        )
        self.bb = bb

    def forward(self, x):
        return self.bb(x)


def pma(pred, tgt):
    abs_err = np.mean(np.abs(pred - tgt), axis=0)
    denom = np.maximum(np.mean(tgt, axis=0), 1e-8)
    vals = 100.0 * abs_err / denom
    out = {f"{n}_PMA": float(vals[i]) for i, n in enumerate(NUTR)}
    out["mean_PMA"] = float(np.mean(vals))
    out.update({f"{n}_MAE": float(abs_err[i]) for i, n in enumerate(NUTR)})
    return out


def evaluate(model, loader, device):
    model.eval(); P, T = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            with torch.autocast("cuda"):
                out = model(x.to(device))
            P.append(out.float().cpu().numpy()); T.append(y.numpy())
    return pma(np.concatenate(P), np.concatenate(T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-depth", type=int, choices=[0, 1], required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda")
    in_ch = 4 if args.use_depth else 3
    print(f"depth run={args.run_name} use_depth={args.use_depth} in_ch={in_ch}", flush=True)

    tr_d, tr_t = usable_dishes("train")
    va_d, va_t = usable_dishes("val")
    te_d, te_t = usable_dishes("test")
    print(f"dishes train {len(tr_d)} val {len(va_d)} test {len(te_d)}", flush=True)

    tr = DataLoader(DepthDS(tr_d, tr_t, args.use_depth, train=True), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va = DataLoader(DepthDS(va_d, va_t, args.use_depth), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers, pin_memory=True)
    te = DataLoader(DepthDS(te_d, te_t, args.use_depth), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = ResNet50RGBD(in_ch=in_ch, pretrained=True).to(device)
    crit = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    run_dir = ROOT / "outputs/task_runs/task4_final/runs" / args.run_name
    ck = run_dir / "checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    best = float("inf"); bad = 0; hist = []
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for x, y, _ in tr:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * x.size(0)
        vm = evaluate(model, va, device)
        hist.append({"epoch": ep, "train_loss": tot / len(tr.dataset), "val": vm})
        print(f"ep {ep:3d} | loss {tot/len(tr.dataset):.3f} | val mean_PMA {vm['mean_PMA']:.2f}% "
              f"(fat {vm['fat_PMA']:.1f})", flush=True)
        if vm["mean_PMA"] < best:
            best = vm["mean_PMA"]; bad = 0
            torch.save({"model_state": model.state_dict(), "epoch": ep, "use_depth": args.use_depth,
                        "val_mean_PMA": best}, ck / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ {ep}", flush=True); break

    state = torch.load(ck / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    test_m = evaluate(model, te, device)
    save_json(run_dir / "training_history.json", hist)
    save_json(run_dir / "test_metrics.json", {**test_m, "best_val_mean_PMA": best,
              "best_epoch": state["epoch"], "use_depth": args.use_depth,
              "n_test_dishes": len(te_d)})
    print(f"DONE {args.run_name} | best_val {best:.2f}% | TEST mean_PMA {test_m['mean_PMA']:.2f}% "
          f"(cal {test_m['calories_PMA']:.1f} fat {test_m['fat_PMA']:.1f} "
          f"carb {test_m['carb_PMA']:.1f} prot {test_m['protein_PMA']:.1f})", flush=True)


if __name__ == "__main__":
    main()
