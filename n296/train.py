"""단계별 학습 (stage1 텍스트 → stage2 이미지 → stage3 융합).

  python -m n296.train --config configs/stage3_296.yaml
  python -m n296.train --config configs/stage3_296.yaml --init-ckpt <pth>        # 기존 가중치에서 이어서
  python -m n296.train --config configs/stage3_296.yaml --only-stage stage3_fusion

스모크 테스트용 옵션: --epochs 1 --max-steps 5 --limit 256
출력: <output_dir>/<stage>_best.pth, <output_dir>/model_stage3_296_final.pth, history.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import TARGETS, load_config, public_config, resolve, seed_everything
from .dataset import Nutrition296Dataset, compute_target_stats, load_manifests, split_train_val
from .metrics import collect, compute_metrics
from .model import Stage3NutritionModel


def set_trainable(model: Stage3NutritionModel, modules, cfg: dict) -> list[str]:
    if modules == "all":
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        for name in modules:
            obj = getattr(model, name)
            params = [obj] if isinstance(obj, torch.nn.Parameter) else list(obj.parameters())
            for p in params:
                p.requires_grad = True
    return model.freeze_pretrained_layers(cfg)


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def loss_fn(pred: dict, batch: dict, cfg: dict) -> torch.Tensor:
    w = cfg["training"].get("loss_weights", {})
    total, denom = 0.0, 0.0
    for j, k in enumerate(TARGETS):
        m = batch["mask"][:, j]
        if not m.any():
            continue
        p, y = pred[k].squeeze(1)[m].float(), batch["y_norm"][:, j][m]
        l = F.smooth_l1_loss(p, y) if cfg["training"].get("loss") == "smooth_l1" else F.l1_loss(p, y)
        total = total + w.get(k, 1.0) * l
        denom += w.get(k, 1.0)
    return total / max(denom, 1e-8)


@torch.no_grad()
def evaluate(model, loader, device, stats, image_mode="full", amp=True):
    model.eval()
    outs = []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            pred = model(batch["image"], batch["text_emb"], batch["has_image"], image_mode=image_mode)
        outs.append({"pred": pred, "y": batch["y"], "mask": batch["mask"]})
    pred, gt = collect(outs, stats)
    return compute_metrics(pred, gt), pred, gt


def run_stage(model, stage, cfg, train_loader, val_loader, device, stats, out_dir, args, history):
    tr = cfg["training"]
    name = stage["name"]
    frozen = set_trainable(model, stage.get("train_modules", "all"), cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"\n=== {name}: 학습 파라미터 {n_train / 1e6:.1f}M (사전학습 고정: {frozen}) ===")
    lr = tr["lr"] * stage.get("lr_scale", 1.0)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)  # [추정]
    scaler = torch.cuda.amp.GradScaler(enabled=tr.get("amp", True) and device.type == "cuda")
    image_mode = stage.get("image_mode", "full")
    for ds in (train_loader.dataset, val_loader.dataset):
        ds.load_images = image_mode != "masked"

    epochs = args.epochs or stage.get("epochs", tr["epochs"])
    best, best_epoch, bad = float("inf"), -1, 0
    best_path = out_dir / f"{name}_best.pth"
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        pbar = tqdm(train_loader, desc=f"{name} ep{epoch}", leave=False)
        for step, batch in enumerate(pbar):
            if args.max_steps and step >= args.max_steps:
                break
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                pred = model(batch["image"], batch["text_emb"], batch["has_image"],
                             modality_dropout=stage.get("modality_dropout", False), image_mode=image_mode)
                loss = loss_fn(pred, batch, cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        metrics, _, _ = evaluate(model, val_loader, device, stats, image_mode, scaler.is_enabled())
        sched.step(metrics["mean_pmae"])
        row = {"stage": name, "epoch": epoch, "train_loss": float(np.mean(losses)), "lr": opt.param_groups[0]["lr"],
               "sec": round(time.time() - t0, 1), **metrics}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(f"[{name}] ep{epoch} loss={row['train_loss']:.4f} mean_pmae={metrics['mean_pmae']:.2f} "
              f"cal={metrics['cal']:.2f} fat={metrics['fat']:.2f} prot={metrics['prot']:.2f} "
              f"carb_head={metrics['carb_head']:.2f} carb_atwater={metrics['carb_atwater']:.2f} ({row['sec']}s)")

        if metrics["mean_pmae"] < best:
            best, best_epoch, bad = metrics["mean_pmae"], epoch, 0
            save_ckpt(best_path, model, cfg, stats, stage_idx=args.stage_index[name], epoch=epoch,
                      best=best, metrics=metrics, stage_name=name)
        else:
            bad += 1
            if bad >= tr["patience"]:
                print(f"[{name}] early stop (patience {tr['patience']})")
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[{name}] best epoch {best_epoch}, mean_pmae {best:.2f} → 다음 단계에 이 가중치 사용")
    return best_path


def save_ckpt(path, model, cfg, stats, stage_idx, epoch, best, metrics, stage_name):
    # 원본 체크포인트와 같은 키 + 역정규화용 target_stats
    torch.save({
        "state_dict": model.state_dict(),
        "stage": stage_idx,
        "stage_name": stage_name,
        "epoch": epoch,
        "best_val_pmae": best,
        "val_metrics": metrics,
        "config": public_config(cfg),
        "target_stats": stats,
        "targets": TARGETS,
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-ckpt", default=None, help="시작 가중치 (예: 기존 model_stage3_296_final.pth)")
    ap.add_argument("--only-stage", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="모든 단계 epoch 수 덮어쓰기")
    ap.add_argument("--max-steps", type=int, default=None, help="epoch 당 최대 step (스모크 테스트)")
    ap.add_argument("--limit", type=int, default=None, help="샘플 수 제한 (스모크 테스트)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    out_dir = resolve(args.output_dir or cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifests(cfg, resolve(cfg["paths"]["manifest"]), resolve(cfg["paths"].get("text_only_manifest")))
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=cfg["seed"])
    train_df, val_df = split_train_val(df, cfg, cfg["seed"])
    stats = compute_target_stats(train_df)
    print(f"train {len(train_df)} (텍스트 전용 {int((train_df.has_image == 0).sum())}) / val {len(val_df)}; "
          f"val 음식: {sorted(val_df['food_name'].unique())[:10]}")
    (out_dir / "split.json").write_text(json.dumps({
        "train_food_codes": sorted(train_df["food_code"].astype(str).unique()),
        "val_food_codes": sorted(val_df["food_code"].astype(str).unique()),
        "target_stats": stats,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    emb_cache = resolve(cfg["paths"]["emb_cache"])
    img_cache = resolve(cfg["paths"].get("image_cache_dir"))
    train_ds = Nutrition296Dataset(train_df, cfg, emb_cache, stats, train=True, image_cache_dir=img_cache)
    val_ds = Nutrition296Dataset(val_df, cfg, emb_cache, stats, train=False, image_cache_dir=img_cache)
    tr = cfg["training"]
    nw = tr.get("image_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True, num_workers=nw,
                              pin_memory=True, drop_last=len(train_ds) > tr["batch_size"])  # BatchNorm: 1개 배치 방지
    val_loader = DataLoader(val_ds, batch_size=tr["batch_size"], shuffle=False, num_workers=nw, pin_memory=True)

    model = Stage3NutritionModel(cfg).to(device)
    if args.init_ckpt:
        sd = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(sd.get("state_dict", sd))
        print(f"초기 가중치 로드: {args.init_ckpt}")
        if "target_stats" not in sd:
            print("  주의: 이 체크포인트에는 target_stats 가 없어 출력 스케일이 현재 정규화와 다를 수 있음")

    stages = cfg["stages"]
    args.stage_index = {s["name"]: i + 1 for i, s in enumerate(stages)}
    if args.only_stage:
        stages = [s for s in stages if s["name"] in args.only_stage]
    history, last = [], None
    for st in stages:
        last = run_stage(model, st, cfg, train_loader, val_loader, device, stats, out_dir, args, history)

    final = out_dir / "model_stage3_296_final.pth"
    ck = torch.load(last, map_location="cpu", weights_only=False)
    torch.save(ck, final)
    print(f"\n최종 저장: {final}\nval_metrics: {json.dumps(ck['val_metrics'], indent=1)}")


if __name__ == "__main__":
    main()
