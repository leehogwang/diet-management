"""side_angles 프레임으로 백본 사전적응 (2026-08-22, 단계 0).

## 아이디어
현행 챔피언 체인은 ImageNet 가중치에서 출발해 overhead 접시(2,485장)로 적응한다.
그런데 라벨은 있는데 overhead 가 없어 **안 쓰던 접시가 1,156장** 더 있고, 전부 side_angles 를
갖고 있다. 그래서 단계 0 을 하나 끼워 넣는다:

  단계 0  side 프레임으로 영양 회귀 사전학습 (train split 3,535접시 = 현행 대비 +42%)
  단계 1  그 백본에서 출발해 overhead RGB-D 384 크롭으로 적응  (기존과 동일)
  단계 2  RMSE 기준 미세조정                                   (기존과 동일)

챔피언과의 차이는 **단계 1 의 출발점이 ImageNet 이냐 side-사전학습이냐** 하나뿐이다.

## 과거 멀티뷰(Task 7, 음성)와 무엇이 다른가
Task 7 은 **같은 접시에 side 뷰를 더해 융합**하는 실험이었다(접시 수 불변). 여기는 반대로
**뷰를 융합하지 않고 접시 수를 늘린다.** 축이 다르므로 음성 결론이 그대로 적용되지 않는다.

## 누수 차단
- train split 접시만 쓴다. val/test 접시의 side 프레임은 **한 장도 넣지 않는다**(인덱스로 강제).
- 검증은 train 에서 떼어낸 홀드아웃(접시 단위)으로 하고, 진짜 val/test 는 건드리지 않는다.

## 입력 형태
side 는 깊이가 없다. 챔피언 백본은 4채널 stem 이므로 **4번째 채널을 0 으로 채워** 형태를 맞춘다
(단계 1 에서 그 채널이 실제 깊이로 채워지며 학습된다). 매 에폭 접시마다 프레임 한 장을 무작위로
뽑으므로 회전 프레임 ~58장이 자연스러운 뷰 증강이 된다.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/home/students/cs/202520906/nutrition5k_pipeline")
sys.path.insert(0, str(ROOT))
from nutrition5k_cbam.utils import save_json, seed_everything  # noqa: E402
from scripts.task_batch.train_depth import NUTR, pma  # noqa: E402
from scripts.task_batch.train_grams import parse_grams, VOCAB_K  # noqa: E402
from scripts.task_batch.train_strong import StrongNet, _nrmse  # noqa: E402
from scripts.task_batch.task2_common import dish_targets5  # noqa: E402

IDX = ROOT / "outputs/task_runs/task_side/side_frames.json"
RUNS = ROOT / "outputs/task_runs/task_side/runs"
IMN_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMN_STD = np.array([0.229, 0.224, 0.225], np.float32)


class SideDS(Dataset):
    """접시 하나 = 한 샘플. 매 호출마다 그 접시의 프레임 중 하나를 무작위로 뽑는다."""

    def __init__(self, dishes, targets, grams, vocab, gmu, gsd, index,
                 res=384, train=False):
        self.d = dishes; self.t = targets; self.g = grams
        self.vocab = vocab; self.gmu = gmu; self.gsd = gsd
        self.ix = index["dishes"]; self.root = Path(index["root"]); self.sub = index["sub"]
        self.res = res; self.train = train

    def __len__(self):
        return len(self.d)

    def gvec(self, d):
        g = self.g.get(d, {})
        v = np.array([g.get(n, 0.0) for n in self.vocab], np.float32)
        return ((np.log1p(v) - self.gmu) / self.gsd).astype(np.float32)

    def __getitem__(self, i):
        d = self.d[i]
        fr = self.ix[d]["frames"]
        # 학습이면 무작위 프레임(뷰 증강), 평가면 항상 가운데 프레임(결정론적)
        name = random.choice(fr) if self.train else fr[len(fr) // 2]
        try:
            img = Image.open(self.root / d / self.sub / name).convert("RGB")
            img = img.resize((self.res, self.res), Image.BILINEAR)
            rgb = np.asarray(img, np.float32) / 255.0
        except Exception:
            rgb = np.zeros((self.res, self.res, 3), np.float32)
        if self.train and random.random() < 0.5:
            rgb = rgb[:, ::-1, :].copy()
        rgb = ((rgb - IMN_MEAN) / IMN_STD).transpose(2, 0, 1)
        # 4번째 채널 = 0 (side 에는 깊이가 없다). 단계 1 에서 실제 깊이로 채워진다.
        x = np.concatenate([rgb, np.zeros((1, self.res, self.res), np.float32)], 0)
        y = torch.tensor(self.t[d], dtype=torch.float32)
        return torch.from_numpy(x), y, torch.from_numpy(self.gvec(d)), d


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); P, T = [], []
    for x, y, _, _ in loader:
        with torch.autocast("cuda"):
            nutr, _ = model(x.to(device))
        P.append(nutr.float().cpu().numpy()); T.append(y.numpy())
    P = np.concatenate(P); T = np.concatenate(T)
    return {**pma(P, T), **_nrmse(P, T)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--backbone", default="convnext_small")
    ap.add_argument("--resolution", type=int, default=384)
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", type=float, default=0.08,
                    help="train 에서 떼어내는 내부 홀드아웃 비율(진짜 val/test 는 건드리지 않는다)")
    args = ap.parse_args()
    seed_everything(args.seed); random.seed(args.seed)
    device = torch.device("cuda")

    index = json.loads(IDX.read_text())
    tgt5 = {}
    for sp in ["train", "val", "test"]:
        tgt5.update(dish_targets5(sp))
    # ★ train split 접시만. val/test 는 인덱스에 있어도 제외한다(누수 차단).
    pool = sorted([d for d, m in index["dishes"].items()
                   if m["split"] == "train" and d in tgt5])
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    n_hold = int(len(pool) * args.holdout)
    hold, tr_d = pool[:n_hold], pool[n_hold:]
    assert not any(index["dishes"][d]["split"] != "train" for d in pool), "train 외 접시가 섞였다"
    tgt = {d: tgt5[d][:4] for d in pool}

    grams = parse_grams()
    df = Counter()
    for d in tr_d:
        for n in grams.get(d, {}):
            df[n] += 1
    vocab = [n for n, _ in df.most_common(VOCAB_K)]
    G = np.array([[np.log1p(grams.get(d, {}).get(n, 0.0)) for n in vocab] for d in tr_d], np.float64)
    gmu, gsd = G.mean(0), G.std(0) + 1e-6
    n_new = sum(1 for d in tr_d if not index["dishes"][d]["has_overhead"])
    print(f"side 사전학습 | 학습 {len(tr_d)}접시 (그중 overhead 없어 기존에 못 쓰던 것 {n_new}) "
          f"| 내부 홀드아웃 {len(hold)} | vocab {len(vocab)}", flush=True)
    print(f"프레임 총 {sum(index['dishes'][d]['n_frames'] for d in tr_d):,}장 "
          f"(접시당 평균 {np.mean([index['dishes'][d]['n_frames'] for d in tr_d]):.0f}장)", flush=True)

    mk = lambda dd, tr: SideDS(dd, tgt, grams, vocab, gmu, gsd, index,
                               res=args.resolution, train=tr)
    tr = DataLoader(mk(tr_d, True), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    ho = DataLoader(mk(hold, False), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    model = StrongNet(args.backbone, use_aux=True, pretrained=True).to(device)
    crit = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    run_dir = RUNS / args.run_name
    ckp = run_dir / "checkpoints"; ckp.mkdir(parents=True, exist_ok=True)
    best = float("inf"); bad = 0; hist = []; stopped_early = False
    for ep in range(1, args.epochs + 1):
        model.train(); ln = la = 0.0
        for x, y, gy, _ in tr:
            x, y, gy = x.to(device), y.to(device), gy.to(device)
            with torch.autocast("cuda"):
                nutr, aux = model(x)
                loss = crit(nutr, y) + args.lam * crit(aux, gy)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ln += loss.item() * x.size(0)
        hm = evaluate(model, ho, device)
        hist.append({"epoch": ep, "loss": ln / len(tr.dataset), "holdout": hm})
        print(f"ep {ep:3d} | loss {ln/len(tr.dataset):.2f} | "
              f"holdout PMA {hm['mean_PMA']:.2f}% nRMSE {hm['mean_nRMSE']:.2f}%", flush=True)
        if hm["mean_PMA"] < best:
            best = hm["mean_PMA"]; bad = 0
            torch.save({"model_state": model.state_dict(), "epoch": ep,
                        "holdout": hm, "backbone": args.backbone}, ckp / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ {ep}", flush=True); stopped_early = True; break

    save_json(run_dir / "training_history.json", hist)
    save_json(run_dir / "pretrain_metrics.json", {
        "best_holdout_mean_PMA": best, "best_epoch": int(np.argmin(
            [h["holdout"]["mean_PMA"] for h in hist]) + 1),
        "epochs_run": len(hist), "epoch_cap": args.epochs,
        "stopped_early": stopped_early, "hit_epoch_cap": (not stopped_early),
        "n_train_dishes": len(tr_d), "n_new_dishes": n_new, "n_holdout": len(hold),
        "backbone": args.backbone, "resolution": args.resolution, "seed": args.seed})
    flag = "" if stopped_early else "  [!] 에폭 상한 도달"
    print(f"DONE {args.run_name} | best holdout PMA {best:.2f}% "
          f"ep{np.argmin([h['holdout']['mean_PMA'] for h in hist])+1}/{len(hist)}{flag}", flush=True)


if __name__ == "__main__":
    main()
