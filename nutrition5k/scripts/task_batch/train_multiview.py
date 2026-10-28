"""Multi-view fusion experiment, built on the RGB-D winner.

Controlled ablation on the SAME dishes (those with overhead RGB-D AND side frames):
  --views overhead : 1 view  (overhead RGB-D, 4ch)              -> control
  --views multi    : overhead RGB-D + K side RGB views (4ch ea) -> treatment

Shared ResNet50 (4ch) encodes every view; an attention pool combines per-view
2048-d features; one head -> 4 nutrients. Side views have no depth, so their 4th
channel is zeros. Proven recipe (raw targets + Softplus + SmoothL1, 224px, 40ep).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from PIL import Image

ROOT = Path("/home/students/cs/202520906/nutrition5k_pipeline")
sys.path.insert(0, str(ROOT))
from nutrition5k_cbam.utils import save_json, seed_everything  # noqa: E402
from scripts.task_batch.train_depth import (  # noqa: E402
    OV, NUTR, IMN_MEAN, IMN_STD, dish_targets, _ok, pma,
)

SA = Path("/home/students/cs/202520906/nutrition5k/imagery/side_angles")
SIDE_CAMERAS = ["A", "B", "C"]  # K=3 distinct angles


def mv_dishes(split):
    """Dishes with overhead RGB-D AND >=1 side frame -> (list, targets)."""
    tg = dish_targets(split)
    out = []
    for d in tg:
        if not (_ok(OV / d / "rgb.png") and _ok(OV / d / "depth_raw.png")):
            continue
        fd = SA / d / "frames_sampled5"
        if fd.exists() and any(fd.glob("*.jpeg")):
            out.append(d)
    return out, tg


def load_rgb(path, res, flip):
    rgb = Image.open(path).convert("RGB").resize((res, res), Image.BILINEAR)
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    if flip:
        rgb = rgb[:, ::-1, :].copy()
    rgb = (rgb - IMN_MEAN) / IMN_STD
    return rgb.transpose(2, 0, 1)  # (3,H,W)


def load_depth(path, res, flip):
    try:
        dep = Image.open(path).resize((res, res), Image.NEAREST)
        dep = np.asarray(dep, dtype=np.float32)
    except Exception:
        dep = np.zeros((res, res), dtype=np.float32)
    mask = dep > 0
    if mask.sum() > 10:
        m, s = dep[mask].mean(), dep[mask].std() + 1e-6
        dep = np.where(mask, (dep - m) / s, 0.0).astype(np.float32)
    else:
        dep = np.zeros((res, res), dtype=np.float32)
    if flip:
        dep = dep[:, ::-1].copy()
    return dep[None, :, :]  # (1,H,W)


class MVDataset(Dataset):
    def __init__(self, dishes, targets, views, res=224, train=False):
        self.dishes = dishes; self.targets = targets; self.views = views
        self.res = res; self.train = train
        # precompute side frames per dish per camera
        self.side = {}
        for d in dishes:
            fd = SA / d / "frames_sampled5"
            cams = {}
            for cam in SIDE_CAMERAS:
                fs = sorted(fd.glob(f"camera_{cam}_frame_*.jpeg")) if fd.exists() else []
                if fs:
                    cams[cam] = fs
            self.side[d] = cams

    @property
    def num_views(self):
        return 1 if self.views == "overhead" else 1 + len(SIDE_CAMERAS)

    def __len__(self):
        return len(self.dishes)

    def __getitem__(self, i):
        d = self.dishes[i]
        flip = self.train and bool(torch.rand(1).item() < 0.5)
        views = []
        # overhead RGB-D
        o_rgb = load_rgb(OV / d / "rgb.png", self.res, flip)
        o_dep = load_depth(OV / d / "depth_raw.png", self.res, flip)
        views.append(np.concatenate([o_rgb, o_dep], axis=0))  # (4,H,W)
        if self.views == "multi":
            cams = self.side[d]
            avail = [c for c in SIDE_CAMERAS if c in cams]
            for cam in SIDE_CAMERAS:
                src = cam if cam in cams else (avail[0] if avail else None)
                if src is None:
                    s_rgb = np.zeros((3, self.res, self.res), dtype=np.float32)
                else:
                    fs = cams[src]
                    idx = int(torch.randint(len(fs), (1,)).item()) if self.train else len(fs) // 2
                    s_rgb = load_rgb(fs[idx], self.res, flip)
                zerod = np.zeros((1, self.res, self.res), dtype=np.float32)  # no depth for side
                views.append(np.concatenate([s_rgb, zerod], axis=0))
        x = np.stack(views, axis=0)  # (V,4,H,W)
        y = torch.tensor(self.targets[d], dtype=torch.float32)
        return torch.from_numpy(x), y, d


class MVNet(nn.Module):
    def __init__(self, pretrained=True, dropout=0.2):
        super().__init__()
        bb = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        old = bb.conv1.weight.data
        new = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old
            new.weight[:, 3:] = old.mean(dim=1, keepdim=True)
        bb.conv1 = new
        bb.fc = nn.Identity()
        self.backbone = bb  # -> 2048
        self.attn = nn.Sequential(nn.Linear(2048, 256), nn.Tanh(), nn.Linear(256, 1))
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2048, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(512, 4), nn.Softplus(),
        )

    def forward(self, x):  # x: (B,V,4,H,W)
        B, V = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).view(B, V, -1)  # (B,V,2048)
        w = torch.softmax(self.attn(f).squeeze(-1), dim=1)  # (B,V)
        fused = (f * w.unsqueeze(-1)).sum(dim=1)  # (B,2048)
        return self.head(fused), w


def evaluate(model, loader, device):
    model.eval(); P, T, W = [], [], []
    with torch.no_grad():
        for x, y, _ in loader:
            with torch.autocast("cuda"):
                out, w = model(x.to(device))
            P.append(out.float().cpu().numpy()); T.append(y.numpy()); W.append(w.float().cpu().numpy())
    m = pma(np.concatenate(P), np.concatenate(T))
    m["mean_attn_weights"] = np.concatenate(W).mean(0).tolist()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", choices=["overhead", "multi"], required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda")
    print(f"multiview run={args.run_name} views={args.views}", flush=True)

    tr_d, tr_t = mv_dishes("train"); va_d, va_t = mv_dishes("val"); te_d, te_t = mv_dishes("test")
    print(f"dishes train {len(tr_d)} val {len(va_d)} test {len(te_d)}", flush=True)

    mk = lambda dd, tt, tr: MVDataset(dd, tt, args.views, train=tr)
    tr = DataLoader(mk(tr_d, tr_t, True), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va = DataLoader(mk(va_d, va_t, False), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    te = DataLoader(mk(te_d, te_t, False), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    model = MVNet(pretrained=True).to(device)
    crit = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    run_dir = ROOT / "outputs/task_runs/task4_final/runs" / args.run_name
    ckp = run_dir / "checkpoints"; ckp.mkdir(parents=True, exist_ok=True)
    best = float("inf"); bad = 0; hist = []
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for x, y, _ in tr:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda"):
                out, _ = model(x); loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * x.size(0)
        vm = evaluate(model, va, device)
        hist.append({"epoch": ep, "train_loss": tot / len(tr.dataset), "val": vm})
        print(f"ep {ep:3d} | loss {tot/len(tr.dataset):.3f} | val mean_PMA {vm['mean_PMA']:.2f}% "
              f"(fat {vm['fat_PMA']:.1f}) | attn {np.round(vm['mean_attn_weights'],2)}", flush=True)
        if vm["mean_PMA"] < best:
            best = vm["mean_PMA"]; bad = 0
            torch.save({"model_state": model.state_dict(), "epoch": ep, "views": args.views,
                        "val_mean_PMA": best}, ckp / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ {ep}", flush=True); break

    state = torch.load(ckp / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    test_m = evaluate(model, te, device)
    save_json(run_dir / "training_history.json", hist)
    save_json(run_dir / "test_metrics.json", {**test_m, "best_val_mean_PMA": best,
              "best_epoch": state["epoch"], "views": args.views, "n_test_dishes": len(te_d)})
    print(f"DONE {args.run_name} | best_val {best:.2f}% | TEST mean_PMA {test_m['mean_PMA']:.2f}% "
          f"(cal {test_m['calories_PMA']:.1f} fat {test_m['fat_PMA']:.1f} "
          f"carb {test_m['carb_PMA']:.1f} prot {test_m['protein_PMA']:.1f}) attn {np.round(test_m['mean_attn_weights'],2)}", flush=True)


if __name__ == "__main__":
    main()
