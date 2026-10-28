"""
Dataset loader for OmniFood8K with camera_0 (overhead/main view) and macros.
Targets: [cal(kcal), fat(g), prot(g), carb(g), mass(g)]
"""

import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms

from dataset_aihub import ResizeWithPadding

OMNIFOOD_ROOT = Path("/mnt/data/food_leehogwang/omnifood8k/0-OminiFood8k")
DEFAULT_EMB_CACHE = Path(__file__).resolve().parent / "text_baseline_the_best/emb_cache_with_ingredients.json"

_TRANSFORMS = {
    "train": transforms.Compose([
        ResizeWithPadding(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        ResizeWithPadding(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}


def load_omnifood_dataset(
    root: Path = OMNIFOOD_ROOT,
    emb_cache_path: Optional[Path] = DEFAULT_EMB_CACHE,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict[str, List[dict]], Dict[str, torch.Tensor]]:
    excel_path = root / "8k-result.xlsx"
    df = pd.read_excel(excel_path).dropna(subset=["ID"])
    df["ID"] = df["ID"].astype(int)

    # Load ID -> ingredient name mapping from ingred-nutrition.xlsx
    nut_xlsx = root / "ingred-nutrition.xlsx"
    id_to_name = {}
    if nut_xlsx.exists():
        nut_df = pd.read_excel(nut_xlsx)
        for _, row in nut_df.iterrows():
            if pd.notna(row["ID"]) and pd.notna(row["Ingredients"]):
                id_to_name[int(row["ID"])] = str(row["Ingredients"]).strip()

    condiments = {
        "cooking oil", "light soy sauce", "dark soy sauce", "vinegar",
        "cooking wine", "doubanjiang", "oyster sauce", "ginger", "garlic",
        "scallion", "salt", "sugar", "water", "chicken essence"
    }

    # Load emb_cache if available
    emb_cache = {}
    if emb_cache_path and emb_cache_path.exists():
        with open(emb_cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            for k, v in raw.items():
                emb_cache[k] = torch.tensor(v, dtype=torch.float32)

    records = []
    for _, row in df.iterrows():
        dish_id = int(row["ID"])
        img_p = root / f"8036/{dish_id}/camera_0.jpg"
        if not img_p.exists():
            continue

        try:
            p = float(pd.to_numeric(row["Total Protein"], errors="coerce"))
            f = float(pd.to_numeric(row["Total Fat"], errors="coerce"))
            c = float(pd.to_numeric(row["Total Carbohydrates"], errors="coerce"))
            m = float(pd.to_numeric(row["Total Weight"], errors="coerce"))
            if np.isnan(p) or np.isnan(f) or np.isnan(c) or np.isnan(m):
                continue
        except Exception:
            continue
        # Atwater derived kcal (since table energy is in kJ)
        kcal = 4.0 * p + 9.0 * f + 4.0 * c

        # Collect top visible ingredients mapped to real names
        ings = []
        for i in range(1, 9):
            col_ing = f"Ingredient {i}"
            val = row.get(col_ing)
            if pd.notna(val):
                try:
                    iid = int(val)
                    name = id_to_name.get(iid)
                    if name and name.lower() not in condiments:
                        ings.append(name.lower())
                except Exception:
                    pass
        dish_text = f"Asian dish, {', '.join(ings)}" if ings else "Asian dish"

        records.append({
            "dish_id": dish_id,
            "image_path": str(img_p),
            "text": dish_text,
            "ingredients": ings,
            "targets": np.array([kcal, f, p, c, m], dtype=np.float32),
            "mass": m,
        })

    # Official train/test splits from train_new333.txt and test_new333.txt
    train_file = root / "train_new333.txt"
    test_file = root / "test_new333.txt"
    train_ids = set(int(line.split()[0]) for line in open(train_file) if line.strip()) if train_file.exists() else set()
    test_ids = set(int(line.split()[0]) for line in open(test_file) if line.strip()) if test_file.exists() else set()

    train_recs = [r for r in records if r["dish_id"] in train_ids]
    test_recs = [r for r in records if r["dish_id"] in test_ids]

    # Split train into train and val
    rng = random.Random(seed)
    rng.shuffle(train_recs)
    val_len = int(len(train_recs) * val_ratio)
    val_recs = train_recs[:val_len]
    train_recs = train_recs[val_len:]

    splits = {
        "train": train_recs,
        "val": val_recs,
        "test": test_recs,
    }
    print(f"[OmniFood8K] Loaded: train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)}")
    return splits, emb_cache


class OmniFoodDataset(Dataset):
    def __init__(self, records: List[dict], emb_cache: Dict[str, torch.Tensor], train: bool = False, default_emb_dim: int = 3072):
        self.records = records
        self.emb_cache = emb_cache
        self.train = train
        self.transform = _TRANSFORMS["train"] if train else _TRANSFORMS["val"]
        self.default_emb = torch.zeros(default_emb_dim, dtype=torch.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.records))
        tensor_img = self.transform(img)

        # Text augmentation: with 30% probability in training, shuffle ingredient order if multiple
        text = rec["text"]
        if self.train and len(rec.get("ingredients", [])) > 1 and random.random() < 0.3:
            shuffled_ings = list(rec["ingredients"])
            random.shuffle(shuffled_ings)
            text_cand = f"Asian dish, {', '.join(shuffled_ings)}"
            if text_cand in self.emb_cache:
                text = text_cand

        text_emb = self.emb_cache.get(text, self.emb_cache.get(rec["text"], self.default_emb))

        return {
            "image": tensor_img,
            "has_image": torch.tensor(1.0, dtype=torch.float32),
            "text_emb": text_emb,
            "targets": torch.tensor(rec["targets"], dtype=torch.float32),
            "mass": torch.tensor(rec["mass"], dtype=torch.float32),
            "dish_id": rec["dish_id"],
        }


def omnifood_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    has_image = torch.stack([b["has_image"] for b in batch])
    text_emb = torch.stack([b["text_emb"] for b in batch])
    targets = torch.stack([b["targets"] for b in batch])
    mass = torch.stack([b["mass"] for b in batch])
    return {
        "images": images,
        "has_image": has_image,
        "text_emb": text_emb,
        "targets": targets,
        "mass": mass,
    }
