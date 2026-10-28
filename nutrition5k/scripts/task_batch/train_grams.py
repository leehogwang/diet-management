"""Ingredient-gram MULTI-TASK experiment, built on the RGB-D winner.

The earlier text experiment failed because ingredient *names* carry no *quantity*.
Here we add the quantity as an AUXILIARY task: the model must also predict each
ingredient's grams (over a top-K vocab) from the image. Grams are supervision, not
input (feeding grams would be cheating). At inference only the nutrient head is used.

Controlled ablation on the SAME overhead RGB-D dishes (507 test), proven recipe
(raw nutrient targets + Softplus + SmoothL1, 224px, 40ep):
  --use-aux 0 : RGB-D, nutrients only            (control == depth_rgbd setup)
  --use-aux 1 : RGB-D + ingredient-gram aux head (multi-task)
total loss = SmoothL1(nutrients) + lam * SmoothL1(standardized log1p grams)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

ROOT = Path("/home/students/cs/202520906/nutrition5k_pipeline")
DSMETA = Path("/home/students/cs/202520906/nutrition5k/metadata")
sys.path.insert(0, str(ROOT))
from nutrition5k_cbam.utils import save_json, seed_everything  # noqa: E402
from scripts.task_batch.train_depth import OV, NUTR, _ok, dish_targets, pma  # noqa: E402
from scripts.task_batch.train_multiview import load_rgb, load_depth  # noqa: E402

VOCAB_K = 80


def parse_grams():
    """dish_id -> {ingredient: grams}."""
    out = {}
    for cafe in ["dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"]:
        for line in (DSMETA / cafe).open():
            p = line.rstrip("\n").split(",")
            if not p or not p[0].startswith("dish_"):
                continue
            g = {}
            i = 6
            while i + 6 < len(p):
                name = p[i + 1].strip()
                try:
                    g[name] = g.get(name, 0.0) + float(p[i + 2])
                except ValueError:
                    pass
                i += 7
            out[p[0]] = g
    return out


def usable(split):
    tg = dish_targets(split)
    return [d for d in tg if _ok(OV / d / "rgb.png") and _ok(OV / d / "depth_raw.png")], tg


class GramsDS(Dataset):
    def __init__(self, dishes, targets, grams, vocab, gmu, gsd, res=224, train=False):
        self.dishes = dishes; self.targets = targets; self.grams = grams
        self.vocab = vocab; self.gmu = gmu; self.gsd = gsd; self.res = res; self.train = train

    def __len__(self):
        return len(self.dishes)

    def gram_vec(self, d):
        g = self.grams.get(d, {})
        v = np.array([g.get(name, 0.0) for name in self.vocab], dtype=np.float32)
        return ((np.log1p(v) - self.gmu) / self.gsd).astype(np.float32)

    def __getitem__(self, i):
        d = self.dishes[i]
        flip = self.train and bool(torch.rand(1).item() < 0.5)
        rgb = load_rgb(OV / d / "rgb.png", self.res, flip)
        dep = load_depth(OV / d / "depth_raw.png", self.res, flip)
        x = np.concatenate([rgb, dep], axis=0)  # (4,H,W)
        y = torch.tensor(self.targets[d], dtype=torch.float32)
        gy = torch.from_numpy(self.gram_vec(d))
        return torch.from_numpy(x), y, gy, d


class GramNet(nn.Module):
    def __init__(self, use_aux, vocab_k=VOCAB_K, pretrained=True, dropout=0.2):
        super().__init__()
        self.use_aux = use_aux
        bb = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        old = bb.conv1.weight.data
        new = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old
            new.weight[:, 3:] = old.mean(dim=1, keepdim=True)
        bb.conv1 = new
        bb.fc = nn.Identity()
        self.backbone = bb
        self.nutr_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2048, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(512, 4), nn.Softplus(),
        )
        if use_aux:
            self.aux_head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(2048, 512), nn.ReLU(inplace=True),
                nn.Linear(512, vocab_k),
            )

    def forward(self, x):
        f = self.backbone(x)
        nutr = self.nutr_head(f)
        aux = self.aux_head(f) if self.use_aux else None
        return nutr, aux


def evaluate(model, loader, device):
    model.eval(); P, T = [], []
    with torch.no_grad():
        for x, y, gy, _ in loader:
            with torch.autocast("cuda"):
                nutr, _ = model(x.to(device))
            P.append(nutr.float().cpu().numpy()); T.append(y.numpy())
    return pma(np.concatenate(P), np.concatenate(T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-aux", type=int, choices=[0, 1], required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda")
    print(f"grams run={args.run_name} use_aux={args.use_aux} lam={args.lam}", flush=True)

    grams = parse_grams()
    tr_d, tr_t = usable("train"); va_d, va_t = usable("val"); te_d, te_t = usable("test")
    # vocab from TRAIN dishes (doc-frequency), normalization stats from train
    df = Counter()
    for d in tr_d:
        for n in grams.get(d, {}):
            df[n] += 1
    vocab = [n for n, _ in df.most_common(VOCAB_K)]
    G = np.array([[np.log1p(grams.get(d, {}).get(n, 0.0)) for n in vocab] for d in tr_d], dtype=np.float64)
    gmu, gsd = G.mean(0), G.std(0) + 1e-6
    print(f"dishes train {len(tr_d)} val {len(va_d)} test {len(te_d)} | vocab {len(vocab)}", flush=True)

    mk = lambda dd, tt, tr: GramsDS(dd, tt, grams, vocab, gmu, gsd, train=tr)
    tr = DataLoader(mk(tr_d, tr_t, True), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va = DataLoader(mk(va_d, va_t, False), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    te = DataLoader(mk(te_d, te_t, False), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    model = GramNet(use_aux=bool(args.use_aux), pretrained=True).to(device)
    crit = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    run_dir = ROOT / "outputs/task_runs/task4_final/runs" / args.run_name
    ckp = run_dir / "checkpoints"; ckp.mkdir(parents=True, exist_ok=True)
    best = float("inf"); bad = 0; hist = []
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0; lnut = 0.0; laux = 0.0
        for x, y, gy, _ in tr:
            x, y, gy = x.to(device), y.to(device), gy.to(device)
            with torch.autocast("cuda"):
                nutr, aux = model(x)
                lo_n = crit(nutr, y)
                lo_a = crit(aux, gy) if model.use_aux else torch.zeros((), device=device)
                loss = lo_n + args.lam * lo_a
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * x.size(0); lnut += lo_n.item() * x.size(0); laux += lo_a.item() * x.size(0)
        vm = evaluate(model, va, device)
        hist.append({"epoch": ep, "loss": tot/len(tr.dataset), "nutr_loss": lnut/len(tr.dataset),
                     "aux_loss": laux/len(tr.dataset), "val": vm})
        print(f"ep {ep:3d} | nutr {lnut/len(tr.dataset):.2f} aux {laux/len(tr.dataset):.3f} | "
              f"val mean_PMA {vm['mean_PMA']:.2f}% (fat {vm['fat_PMA']:.1f})", flush=True)
        if vm["mean_PMA"] < best:
            best = vm["mean_PMA"]; bad = 0
            torch.save({"model_state": model.state_dict(), "epoch": ep, "use_aux": args.use_aux,
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
              "best_epoch": state["epoch"], "use_aux": args.use_aux, "lam": args.lam,
              "n_test_dishes": len(te_d), "vocab_k": len(vocab)})
    print(f"DONE {args.run_name} | best_val {best:.2f}% | TEST mean_PMA {test_m['mean_PMA']:.2f}% "
          f"(cal {test_m['calories_PMA']:.1f} fat {test_m['fat_PMA']:.1f} "
          f"carb {test_m['carb_PMA']:.1f} prot {test_m['protein_PMA']:.1f})", flush=True)


if __name__ == "__main__":
    main()
