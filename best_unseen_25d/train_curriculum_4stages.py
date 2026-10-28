"""
Unified 4-Stage Zero-Leakage Curriculum Training Engine.

Stages:
- Stage 0: kFood 140-Class Vision Backbone Pretraining (CrossEntropyLoss)
  -> checkpoints_curriculum_4stages/checkpoint_stage0_backbone.pth
- Stage 1: OmniFood8K Physical Nutrition Learning (5,101 train plates, macros + mass)
  -> checkpoints_curriculum_4stages/checkpoint_stage1_omnifood.pth
- Stage 2: AIHub 122 369-Class Standard Nutrition Scale Learning (70k images + 20% OmniFood buffer)
  -> checkpoints_curriculum_4stages/checkpoint_stage2_122_std.pth
- Stage 3: AIHub 122 50-Class Portion (Q1~Q5) Estimation (122 Q train + 10% 122 std + 10% OmniFood buffer)
  -> checkpoints_curriculum_4stages/model_stage3_final.pth
"""

import argparse
import functools
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms

print = functools.partial(print, flush=True)

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "text_baseline_the_best"))

from train_multimodal import MultimodalNutritionModel
from models.vision_classifier import VisionFoodClassifier
from dataset_omnifood8k import load_omnifood_dataset, OmniFoodDataset
from dataset_aihub122_classification_nutrition import load_122_classification_dataset, AIHub122ClassificationDataset
from dataset_aihub122_multimodal import load_aihub122_dataset, AIHub122MultimodalDataset
from dataset_aihub import ResizeWithPadding


def pmae(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (pred - gt).abs() / (gt.abs() + 1.0)


def predict_abs_5(preds: dict) -> dict:
    """Convert model predictions from log1p space to absolute scale."""
    return {
        "calories": torch.expm1(preds["calories"].squeeze(-1).clamp(0, 12)).clamp(min=0),
        "fat": torch.expm1(preds["fat"].squeeze(-1).clamp(0, 10)).clamp(min=0),
        "protein": torch.expm1(preds["protein"].squeeze(-1).clamp(0, 10)).clamp(min=0),
        "carb": torch.expm1(preds["carb"].squeeze(-1).clamp(0, 10)).clamp(min=0),
        "mass": torch.expm1(preds["mass"].squeeze(-1).clamp(0, 12)).clamp(min=0),
    }


# -------------------------------------------------------------
# Stage 0: kFood Dataset for Vision Classification
# -------------------------------------------------------------
class KFoodPretrainDataset(Dataset):
    def __init__(self, parquet_path: Path, train: bool = True, val_ratio: float = 0.05, seed: int = 42):
        df = pd.read_parquet(parquet_path)
        rng = random.Random(seed)
        shuffled_indices = list(range(len(df)))
        rng.shuffle(shuffled_indices)
        val_len = int(len(df) * val_ratio)
        if train:
            self.data = df.iloc[shuffled_indices[val_len:]].reset_index(drop=True)
        else:
            self.data = df.iloc[shuffled_indices[:val_len]].reset_index(drop=True)

        self.transform = transforms.Compose([
            ResizeWithPadding(224),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(0.1, 0.1) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.data))
        return self.transform(img), torch.tensor(row["label"], dtype=torch.long)


# -------------------------------------------------------------
# Unified Multi-Dataset Rehearsal Wrapper
# -------------------------------------------------------------
class MultiRehearsalDataset(Dataset):
    def __init__(self, main_ds, buffer_ds_list: list, buffer_weights: list):
        self.main_ds = main_ds
        self.buffers = buffer_ds_list
        self.weights = buffer_weights

    def __len__(self):
        return len(self.main_ds)

    def __getitem__(self, idx: int):
        r = random.random()
        cum = 0.0
        for b_ds, w in zip(self.buffers, self.weights):
            cum += w
            if r < cum and len(b_ds) > 0:
                b_idx = random.randint(0, len(b_ds) - 1)
                return b_ds[b_idx]
        return self.main_ds[idx]


# -------------------------------------------------------------
# Unified Collate Function for Multimodal Datasets
# -------------------------------------------------------------
def unified_nutrition_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    has_image = torch.tensor([
        float(b["has_image"]) if not isinstance(b["has_image"], torch.Tensor)
        else b["has_image"].item()
        for b in batch
    ], dtype=torch.bool)

    text_embs = []
    for b in batch:
        te = b["text_emb"]
        if isinstance(te, torch.Tensor):
            text_embs.append(te)
        elif isinstance(te, np.ndarray):
            text_embs.append(torch.from_numpy(te.astype(np.float32)))
        elif isinstance(te, list):
            text_embs.append(torch.tensor(te, dtype=torch.float32))
        else:
            text_embs.append(torch.zeros(3072, dtype=torch.float32))
    text_emb = torch.stack(text_embs)

    targets_list = []
    for b in batch:
        t = b.get("targets", b.get("target"))
        if isinstance(t, torch.Tensor):
            targets_list.append(t)
        elif isinstance(t, np.ndarray):
            targets_list.append(torch.from_numpy(t.astype(np.float32)))
        elif isinstance(t, list):
            targets_list.append(torch.tensor(t, dtype=torch.float32))
        else:
            raise ValueError(f"Unknown target type: {type(t)}")
    targets = torch.stack(targets_list)

    return {
        "images": images,
        "has_image": has_image,
        "text_emb": text_emb,
        "targets": targets,
        "portion": torch.tensor([b.get("portion_label", -1) for b in batch], dtype=torch.long),  # -1 = 라벨 없음(버퍼)
    }


# -------------------------------------------------------------
# Evaluation Helper
# -------------------------------------------------------------
@torch.no_grad()
def evaluate_nutrition_model(model, loader, device, include_mass: bool = True):
    model.eval()
    all_cal, all_fat, all_prot, all_carb, all_mass = [], [], [], [], []

    for batch in loader:
        images = batch["images"].to(device)
        has_image = batch["has_image"].to(device)
        text_emb = batch["text_emb"].to(device)
        targets = batch["targets"].to(device)

        preds = model(images, has_image, text_emb, apply_dropout=False)
        pred_abs = predict_abs_5(preds)

        pred_cal = pred_abs["calories"]
        pred_fat = pred_abs["fat"]
        pred_prot = pred_abs["protein"]
        pred_carb = pred_abs["carb"]
        pred_mass = pred_abs["mass"]

        gt_cal = targets[:, 0]
        gt_fat = targets[:, 1]
        gt_prot = targets[:, 2]
        gt_carb = targets[:, 3]
        gt_mass = targets[:, 4]

        all_cal.append(pmae(pred_cal, gt_cal).cpu())
        all_fat.append(pmae(pred_fat, gt_fat).cpu())
        all_prot.append(pmae(pred_prot, gt_prot).cpu())
        all_carb.append(pmae(pred_carb, gt_carb).cpu())
        all_mass.append(pmae(pred_mass, gt_mass).cpu())

    cal_m = torch.cat(all_cal).mean().item() * 100
    fat_m = torch.cat(all_fat).mean().item() * 100
    prot_m = torch.cat(all_prot).mean().item() * 100
    carb_m = torch.cat(all_carb).mean().item() * 100
    mass_m = torch.cat(all_mass).mean().item() * 100

    if include_mass:
        mean_all = (cal_m + fat_m + prot_m + carb_m + mass_m) / 5.0
    else:
        mean_all = (cal_m + fat_m + prot_m + carb_m) / 4.0

    return {
        "mean_pmae": mean_all,
        "cal": cal_m,
        "fat": fat_m,
        "prot": prot_m,
        "carb": carb_m,
        "mass": mass_m,
    }


# -------------------------------------------------------------
# Main Runner
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified 4-Stage Zero-Leakage Curriculum Trainer")
    parser.add_argument("--stage", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_dir", type=str, default="checkpoints_curriculum_4stages")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--config", type=str, default="text_baseline_the_best/config_multimodal_296_text64_ingredients.yaml")
    parser.add_argument("--no_mass", action="store_true", help="Train without mass target and disable mass feedback")
    parser.add_argument("--backbone_path", type=str, default=None, help="Path to Stage 0 backbone checkpoint")
    parser.add_argument("--fold", type=int, default=None, help="Stage 3 k-fold index: use this fold's foods as validation")
    parser.add_argument("--folds_file", type=str, default="foodlist_folds_122.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--portion_aux", type=float, default=0.0, help="Stage 3: weight of Q1~Q5 portion classification aux loss (0=off)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 65)
    print(f"  Curriculum Stage {args.stage} Launching (Device: {device})")
    print(f"  Max Epochs: {args.epochs} | Batch Size: {args.batch_size} | Patience: {args.patience}")
    print("=" * 65)

    # ---------------------------------------------------------
    # STAGE 0: kFood 140-Class Vision Pretraining
    # ---------------------------------------------------------
    if args.stage == 0:
        manifest_p = ROOT_DIR / "data/kfood_pretrain_manifest.parquet"
        train_ds = KFoodPretrainDataset(manifest_p, train=True)
        val_ds = KFoodPretrainDataset(manifest_p, train=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

        model = VisionFoodClassifier(num_classes=140, pretrained=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr or 1e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        patience = 0
        save_path = out_dir / "checkpoint_stage0_backbone.pth"

        print(f"[Stage 0] Training on {len(train_ds)} images across 140 classes (Val: {len(val_ds)})...")
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            t0 = time.time()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y)
                preds = out.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += len(y)

            train_acc = correct / total * 100

            # Val
            model.eval()
            v_correct, v_total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    v_correct += (out.argmax(dim=-1) == y).sum().item()
                    v_total += len(y)
            val_acc = v_correct / v_total * 100
            elapsed = time.time() - t0

            print(f"==> [Stage 0] Epoch {epoch:3d} ({elapsed:.1f}s) | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            if val_acc > best_acc:
                best_acc = val_acc
                patience = 0
                torch.save({
                    "epoch": epoch,
                    "backbone_state_dict": model.extract_backbone_state_dict(),
                    "val_acc": val_acc
                }, save_path)
                print(f"  ★ [Saved Best Backbone] -> {save_path} (Val Acc: {val_acc:.2f}%)")
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"[EarlyStop] Reached patience {args.patience} at Epoch {epoch}. Best Val Acc: {best_acc:.2f}%")
                    break

        print("[Stage 0 Complete] Best Backbone saved successfully.")
        return

    # ---------------------------------------------------------
    # STAGE 1, 2, 3: Multimodal Nutrition Regression Stages
    # ---------------------------------------------------------
    model = MultimodalNutritionModel(
        image_dim=cfg["model"]["image_dim"],
        text_in_dim=3072,
        text_hidden1=cfg["model"]["text_hidden1"],
        text_hidden2=cfg["model"]["text_hidden2"],
        head_hidden=cfg["model"]["head_hidden"],
        dropout=cfg["model"]["dropout"],
        freeze_backbone_layers=cfg["model"]["freeze_backbone_layers"],
        modality_dropout_prob=cfg["model"]["modality_dropout_prob"],
        freq_cfg=cfg.get("freq", None),
        mass_feedback=not args.no_mass,
    ).to(device)

    # Load Stage 0 Backbone if starting Stage 1 fresh
    if args.stage == 1 and not args.resume:
        s0_path = Path(args.backbone_path) if args.backbone_path else (out_dir / "checkpoint_stage0_backbone.pth")
        if s0_path.exists():
            s0_ckpt = torch.load(s0_path, map_location=device, weights_only=False)
            model.image_encoder.load_state_dict(s0_ckpt["backbone_state_dict"], strict=False)
            print(f"[transfer] Successfully transferred Stage 0 Korean Backbone from: {s0_path}")
        else:
            print(f"[transfer] Warning: {s0_path} not found, initializing backbone from ImageNet.")

    if args.no_mass:
        print("[model] Configured NutritionHead with mass_feedback=False (No Mass Mode)")

    # Resume/Transfer logic
    if args.resume:
        r_path = Path(args.resume)
        if r_path.exists():
            ckpt = torch.load(r_path, map_location=device, weights_only=False)
            sd = ckpt.get("state_dict", ckpt)
            model.load_state_dict(sd, strict=False)
            print(f"[transfer] Loaded weights from: {r_path}")
        else:
            print(f"[transfer] Warning: Resume path {r_path} not found.")

    # 분량(Q1~Q5) 분류 보조 헤드: NutritionHead 은닉층(ReLU+Dropout 이후)을 hook으로 받아 5-way 분류
    head_hidden_feat = {}
    if args.portion_aux > 0:
        model.portion_head = nn.Linear(cfg["model"]["head_hidden"], 5).to(device)
        model.nutrition_head.dropout.register_forward_hook(lambda m, i, o: head_hidden_feat.__setitem__("h", o))
        print(f"[portion_aux] enabled (weight={args.portion_aux})")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr or (3e-4 if args.stage == 1 else (1e-4 if args.stage == 2 else 5e-5)),
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = nn.SmoothL1Loss()

    # Prepare datasets per stage
    if args.stage == 1:
        # OmniFood8K
        splits, embs = load_omnifood_dataset()
        train_ds = OmniFoodDataset(splits["train"], embs, train=True)
        val_loader = DataLoader(OmniFoodDataset(splits["val"], embs, train=False), batch_size=args.batch_size, shuffle=False, collate_fn=unified_nutrition_collate_fn, num_workers=args.workers)
        test_loader = DataLoader(OmniFoodDataset(splits["test"], embs, train=False), batch_size=args.batch_size, shuffle=False, collate_fn=unified_nutrition_collate_fn, num_workers=args.workers)
        save_ckpt = out_dir / "checkpoint_stage1_omnifood.pth"
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=unified_nutrition_collate_fn, num_workers=args.workers, pin_memory=True)

    elif args.stage == 2:
        # AIHub 122 369-Class Classification Nutrition + 20% OmniFood Buffer
        splits_122, embs_122, vars_122 = load_122_classification_dataset()
        ds_122_train = AIHub122ClassificationDataset(splits_122["train"], embs_122, vars_122, train=True)
        val_loader = DataLoader(AIHub122ClassificationDataset(splits_122["val"], embs_122, vars_122, train=False), batch_size=args.batch_size, shuffle=False, collate_fn=unified_nutrition_collate_fn, num_workers=args.workers)

        # Buffer from Stage 1
        splits_omni, embs_omni = load_omnifood_dataset()
        omni_buffer = OmniFoodDataset(splits_omni["train"][:500], embs_omni, train=True)
        train_ds = MultiRehearsalDataset(ds_122_train, [omni_buffer], [0.2])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=unified_nutrition_collate_fn, num_workers=args.workers, pin_memory=True)
        save_ckpt = out_dir / "checkpoint_stage2_122_std.pth"
        test_loader = None

    elif args.stage == 3:
        # AIHub 122 50-Class Portion (Q1~Q5) Estimation + Buffers
        val_foods = None
        if args.fold is not None:
            val_foods = json.loads(Path(args.folds_file).read_text(encoding="utf-8"))["folds"][args.fold]
            print(f"[fold] Stage 3 fold {args.fold}: {len(val_foods)} val foods -> {val_foods}")
        splits_122_q, embs_122_q = load_aihub122_dataset(emb_cache_path=Path("text_baseline_the_best/emb_cache_with_ingredients.json"), val_foods=val_foods)
        ds_q_train = AIHub122MultimodalDataset(splits_122_q["train"], embs_122_q, train=True)
        val_loader = DataLoader(AIHub122MultimodalDataset(splits_122_q["val"], embs_122_q, train=False), batch_size=args.batch_size, shuffle=False, collate_fn=unified_nutrition_collate_fn, num_workers=args.workers)
        test_loader = DataLoader(AIHub122MultimodalDataset(splits_122_q["test"], embs_122_q, train=False), batch_size=args.batch_size, shuffle=False, collate_fn=unified_nutrition_collate_fn, num_workers=args.workers)

        # Rehearsal Buffers
        splits_std, embs_std, vars_std = load_122_classification_dataset()
        buffer_std = AIHub122ClassificationDataset(splits_std["train"][:500], embs_std, vars_std, train=True)
        splits_omni, embs_omni = load_omnifood_dataset()
        buffer_omni = OmniFoodDataset(splits_omni["train"][:500], embs_omni, train=True)

        train_ds = MultiRehearsalDataset(ds_q_train, [buffer_std, buffer_omni], [0.1, 0.1])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=unified_nutrition_collate_fn, num_workers=args.workers, pin_memory=True)
        save_ckpt = out_dir / "model_stage3_final.pth"

    # Train Loop
    best_val_pmae = float("inf")
    patience = 0

    print(f"[Stage {args.stage}] Training started (Max Epochs: {args.epochs}, Patience: {args.patience})...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            images = batch["images"].to(device)
            has_image = batch["has_image"].to(device)
            text_emb = batch["text_emb"].to(device)
            targets = batch["targets"].to(device)

            targets_log = torch.log1p(targets.clamp(min=0))

            optimizer.zero_grad()
            preds = model(images, has_image, text_emb, apply_dropout=True)

            # Targets: [cal, fat, prot, carb, mass]
            loss_cal = criterion(preds["calories"].squeeze(-1), targets_log[:, 0])
            loss_fat = criterion(preds["fat"].squeeze(-1), targets_log[:, 1])
            loss_prot = criterion(preds["protein"].squeeze(-1), targets_log[:, 2])
            loss_carb = criterion(preds["carb"].squeeze(-1), targets_log[:, 3])

            if args.no_mass:
                loss = loss_cal + loss_fat + loss_prot + loss_carb
            else:
                loss_mass = criterion(preds["mass"].squeeze(-1), targets_log[:, 4])
                loss = loss_cal + loss_fat + loss_prot + loss_carb + loss_mass

            if args.portion_aux > 0:
                portion = batch["portion"].to(device)
                if (portion >= 0).any():
                    logits = model.portion_head(head_hidden_feat["h"])
                    loss = loss + args.portion_aux * nn.functional.cross_entropy(logits, portion, ignore_index=-1)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(images)

        train_loss = total_loss / len(train_ds)
        val_res = evaluate_nutrition_model(model, val_loader, device, include_mass=not args.no_mass)
        scheduler.step(val_res["mean_pmae"])
        elapsed = time.time() - t0

        if args.no_mass:
            print(f"==> [Stage {args.stage}] Epoch {epoch:3d} ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Mean PMAE: {val_res['mean_pmae']:.2f}% | cal: {val_res['cal']:.1f}% | fat: {val_res['fat']:.1f}% | prot: {val_res['prot']:.1f}% | carb: {val_res['carb']:.1f}%")
        else:
            print(f"==> [Stage {args.stage}] Epoch {epoch:3d} ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Mean PMAE: {val_res['mean_pmae']:.2f}% | cal: {val_res['cal']:.1f}% | fat: {val_res['fat']:.1f}% | prot: {val_res['prot']:.1f}% | carb: {val_res['carb']:.1f}% | mass: {val_res['mass']:.1f}%")

        if val_res["mean_pmae"] < best_val_pmae:
            best_val_pmae = val_res["mean_pmae"]
            patience = 0
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_pmae": best_val_pmae,
            }, save_ckpt)
            print(f"  ★ [Saved Best Checkpoint] -> {save_ckpt} (Val PMAE: {best_val_pmae:.2f}%)")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[EarlyStop] Reached patience {args.patience} at Epoch {epoch}. Best Val PMAE: {best_val_pmae:.2f}%")
                break

    # If Stage 3, evaluate on Test Set
    if args.stage == 3 and test_loader is not None:
        print("=" * 65)
        print("  Stage 3 Completed! Evaluating Best Model on AIHub 122 Test Set (23 Foods)")
        print("=" * 65)
        best_ckpt = torch.load(save_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["state_dict"])
        test_res = evaluate_nutrition_model(model, test_loader, device, include_mass=not args.no_mass)
        print("Test Set Results:")
        print(f"  Mean PMAE:     {test_res['mean_pmae']:.2f}%")
        print(f"  Calories PMAE: {test_res['cal']:.2f}%")
        print(f"  Fat PMAE:      {test_res['fat']:.2f}%")
        print(f"  Protein PMAE:  {test_res['prot']:.2f}%")
        print(f"  Carb PMAE:     {test_res['carb']:.2f}%")
        if not args.no_mass:
            print(f"  Mass PMAE:     {test_res['mass']:.2f}%")


if __name__ == "__main__":
    main()
