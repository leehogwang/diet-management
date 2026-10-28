"""Task4 empirical test: same ResNet50 GramNet (RGB-D + gram aux) but trained on
plate-ROI CROPPED images (background removed) instead of full frames.
Everything else identical to train_grams.py so the only variable is the crop.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

ROOT = Path("/home/students/cs/202520906/nutrition5k_pipeline")
sys.path.insert(0, str(ROOT))
from nutrition5k_cbam.utils import save_json, seed_everything  # noqa
from scripts.task_batch.train_depth import OV, _ok, dish_targets, pma  # noqa
from scripts.task_batch.train_grams import parse_grams, usable, GramNet, evaluate, VOCAB_K  # noqa

IMN_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMN_STD = np.array([0.229, 0.224, 0.225], np.float32)
BBOX = json.load((ROOT / "outputs/task_runs/task4_final/crop_bbox.json").open())

# --- 크롭 박스 통일 실험 (2026-08-20) -------------------------------------------
# 현행 크롭은 접시 검출이 성공하면 검출박스, 실패(면적<15% 또는 >97%)하면 고정박스를 쓴다.
# 실측: 3,261접시 중 2,870장(88.0%)이 고정박스, 391장(12.0%)이 제각각 다른 검출박스.
# FIXED_BOX 를 튜플로 세팅하면 **모든 접시가 같은 박스**를 쓴다(= 박스 통일).
# preprocess_plate_roi.FALLBACK 과 같은 값이며 verify_fixed_box.py 가 일치를 검증한다.
FIXED_BOX = None
FALLBACK_BOX = (110, 15, 575, 470)   # x0,y0,x1,y1 on the 640x480 overhead frame


def box_for(dish):
    """이 접시에 쓸 크롭 박스. FIXED_BOX 가 세팅돼 있으면 접시와 무관하게 그 박스."""
    return FIXED_BOX if FIXED_BOX is not None else BBOX.get(dish)


def crop_rgb(path, dish, res, flip):
    img = Image.open(path).convert("RGB")
    bb = box_for(dish)
    if bb:
        img = img.crop((bb[0], bb[1], bb[2], bb[3]))
    img = img.resize((res, res), Image.BILINEAR)
    rgb = np.asarray(img, np.float32) / 255.0
    if flip:
        rgb = rgb[:, ::-1, :].copy()
    rgb = (rgb - IMN_MEAN) / IMN_STD
    return rgb.transpose(2, 0, 1)


def crop_depth(path, dish, res, flip):
    try:
        img = Image.open(path)
        bb = box_for(dish)
        if bb:
            img = img.crop((bb[0], bb[1], bb[2], bb[3]))
        dep = np.asarray(img.resize((res, res), Image.NEAREST), np.float32)
    except Exception:
        dep = np.zeros((res, res), np.float32)
    mask = dep > 0
    if mask.sum() > 10:
        m, s = dep[mask].mean(), dep[mask].std() + 1e-6
        dep = np.where(mask, (dep - m) / s, 0.0).astype(np.float32)
    else:
        dep = np.zeros((res, res), np.float32)
    if flip:
        dep = dep[:, ::-1].copy()
    return dep[None, :, :]


class CropDS(Dataset):
    def __init__(self, dishes, targets, grams, vocab, gmu, gsd, res=224, train=False):
        self.d = dishes; self.t = targets; self.g = grams
        self.vocab = vocab; self.gmu = gmu; self.gsd = gsd; self.res = res; self.train = train

    def __len__(self):
        return len(self.d)

    def gvec(self, d):
        g = self.g.get(d, {})
        v = np.array([g.get(n, 0.0) for n in self.vocab], np.float32)
        return ((np.log1p(v) - self.gmu) / self.gsd).astype(np.float32)

    def __getitem__(self, i):
        d = self.d[i]
        flip = self.train and bool(torch.rand(1).item() < 0.5)
        rgb = crop_rgb(OV / d / "rgb.png", d, self.res, flip)
        dep = crop_depth(OV / d / "depth_raw.png", d, self.res, flip)
        x = np.concatenate([rgb, dep], axis=0)
        y = torch.tensor(self.t[d], dtype=torch.float32)
        gy = torch.from_numpy(self.gvec(d))
        return torch.from_numpy(x), y, gy, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda")
    print(f"CROP run={args.run_name} lam={args.lam} seed={args.seed}", flush=True)

    grams = parse_grams()
    tr_d, tr_t = usable("train"); va_d, va_t = usable("val"); te_d, te_t = usable("test")
    df = Counter()
    for d in tr_d:
        for n in grams.get(d, {}):
            df[n] += 1
    vocab = [n for n, _ in df.most_common(VOCAB_K)]
    G = np.array([[np.log1p(grams.get(d, {}).get(n, 0.0)) for n in vocab] for d in tr_d], np.float64)
    gmu, gsd = G.mean(0), G.std(0) + 1e-6
    print(f"dishes train {len(tr_d)} val {len(va_d)} test {len(te_d)} | vocab {len(vocab)}", flush=True)

    mk = lambda dd, tt, tr: CropDS(dd, tt, grams, vocab, gmu, gsd, train=tr)
    tr = DataLoader(mk(tr_d, tr_t, True), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va = DataLoader(mk(va_d, va_t, False), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    te = DataLoader(mk(te_d, te_t, False), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    model = GramNet(use_aux=True, pretrained=True).to(device)
    crit = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    run_dir = ROOT / "outputs/task_runs/task4_final/runs" / args.run_name
    ckp = run_dir / "checkpoints"; ckp.mkdir(parents=True, exist_ok=True)
    best = float("inf"); bad = 0; hist = []
    for ep in range(1, args.epochs + 1):
        model.train(); tot = lnut = laux = 0.0
        for x, y, gy, _ in tr:
            x, y, gy = x.to(device), y.to(device), gy.to(device)
            with torch.autocast("cuda"):
                nutr, aux = model(x)
                lo_n = crit(nutr, y); lo_a = crit(aux, gy)
                loss = lo_n + args.lam * lo_a
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()*x.size(0); lnut += lo_n.item()*x.size(0); laux += lo_a.item()*x.size(0)
        vm = evaluate(model, va, device)
        hist.append({"epoch": ep, "loss": tot/len(tr.dataset), "nutr_loss": lnut/len(tr.dataset),
                     "aux_loss": laux/len(tr.dataset), "val": vm})
        print(f"ep {ep:3d} | nutr {lnut/len(tr.dataset):.2f} | val mean_PMA {vm['mean_PMA']:.2f}% "
              f"(cal {vm['calories_PMA']:.1f})", flush=True)
        if vm["mean_PMA"] < best:
            best = vm["mean_PMA"]; bad = 0
            torch.save({"model_state": model.state_dict(), "epoch": ep, "use_aux": 1,
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
              "best_epoch": state["epoch"], "use_aux": 1, "lam": args.lam,
              "n_test_dishes": len(te_d), "vocab_k": len(vocab), "cropped": 1})
    print(f"DONE {args.run_name} | best_val {best:.2f}% | TEST mean {test_m['mean_PMA']:.2f}% "
          f"(cal {test_m['calories_PMA']:.1f})", flush=True)


if __name__ == "__main__":
    main()
