"""
Dataset loader for AIHub 122 400-class food classification images paired with
standard 1-serving nutrition DB (44.음식분류_AI_데이터_영양DB.xlsx).
Strictly filters out Test Holdout (23 foods) and Val Holdout (8 foods) to keep
369 non-test food classes for Stage 2 scale learning.
Enriched with Gemini whole-food ingredients and synonym text variants augmentation.
"""

import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms

from dataset_aihub import ResizeWithPadding

DEFAULT_122_ROOT = Path("/mnt/data/food_leehogwang/122.음식_분류를_위한_음식종류_및_양에_따른_칼로리_데이터셋(재료,_양념,_완제품_등)")
DEFAULT_MANIFEST = Path(__file__).parent / "data/aihub122_recognition_manifest.parquet"
DEFAULT_NUTRITION_XLSX = DEFAULT_122_ROOT / "01.데이터/9.칼로리데이터셋/44.음식분류_AI_데이터_영양DB.xlsx"
DEFAULT_EMB_CACHE = Path(__file__).resolve().parent / "text_baseline_the_best/emb_cache_with_ingredients.json"
DEFAULT_VARIANTS_JSON = Path(__file__).resolve().parent / "ingredient_extraction/food_to_ingredient_texts.json"

_TRANSFORMS = {
    "train": transforms.Compose([
        ResizeWithPadding(224),
        transforms.RandomHorizontalFlip(),
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


def load_122_classification_dataset(
    manifest_path: Path = DEFAULT_MANIFEST,
    nutrition_xlsx: Path = DEFAULT_NUTRITION_XLSX,
    emb_cache_path: Optional[Path] = DEFAULT_EMB_CACHE,
    variants_json_path: Optional[Path] = DEFAULT_VARIANTS_JSON,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[Dict[str, List[dict]], Dict[str, torch.Tensor], Dict[str, dict]]:
    manifest_df = pd.read_parquet(manifest_path)
    # pretrain split only contains the 369 non-test holdout food classes
    pretrain_df = manifest_df[manifest_df["split"] == "pretrain"].copy()

    nut_df = pd.read_excel(nutrition_xlsx)
    nut_df["음 식 명"] = nut_df["음 식 명"].astype(str).str.strip()
    nut_map = {}
    for _, row in nut_df.iterrows():
        fname = row["음 식 명"]
        cal = float(row["에너지(kcal)"])
        fat = float(row["지방(g)"])
        prot = float(row["단백질(g)"])
        carb = float(row["탄수화물(g)"])
        mass = float(row["중량(g)"])
        nut_map[fname] = np.array([cal, fat, prot, carb, mass], dtype=np.float32)

    # Load emb_cache if available
    emb_cache = {}
    if emb_cache_path and emb_cache_path.exists():
        with open(emb_cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            for k, v in raw.items():
                emb_cache[k] = torch.tensor(v, dtype=torch.float32)

    # Load variants json
    food_variants = {}
    if variants_json_path and variants_json_path.exists():
        with open(variants_json_path, "r", encoding="utf-8") as f:
            food_variants = json.load(f)

    records = []
    for _, row in pretrain_df.iterrows():
        fname = row["food_name"]
        if fname not in nut_map:
            continue
        records.append({
            "image_path": row["image_path"],
            "food_name": fname,
            "targets": nut_map[fname],
            "mass": nut_map[fname][4],
        })

    rng = random.Random(seed)
    rng.shuffle(records)
    val_len = int(len(records) * val_ratio)
    val_recs = records[:val_len]
    train_recs = records[val_len:]

    splits = {
        "train": train_recs,
        "val": val_recs,
    }
    print(f"[AIHub 122 Classification 369-Class] Loaded: train={len(train_recs)}, val={len(val_recs)} (Test/Val Holdout 31 classes strictly excluded)")
    return splits, emb_cache, food_variants


class AIHub122ClassificationDataset(Dataset):
    def __init__(
        self,
        records: List[dict],
        emb_cache: Dict[str, torch.Tensor],
        food_variants: Optional[Dict[str, dict]] = None,
        train: bool = False,
        default_emb_dim: int = 3072
    ):
        self.records = records
        self.emb_cache = emb_cache
        self.food_variants = food_variants or {}
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

        food_name = rec["food_name"]
        text_info = self.food_variants.get(food_name, {})
        base_text = text_info.get("base", food_name)
        variants = text_info.get("variants", [])

        # Training text augmentation: 50% chance to pick a synonym variant
        chosen_text = base_text
        if self.train and variants and random.random() < 0.5:
            chosen_text = random.choice(variants)

        # Lookup embedding: variant/base -> base -> food_name -> default
        text_emb = self.emb_cache.get(
            chosen_text,
            self.emb_cache.get(base_text, self.emb_cache.get(food_name, self.default_emb))
        )

        return {
            "image": tensor_img,
            "has_image": torch.tensor(1.0, dtype=torch.float32),
            "text_emb": text_emb,
            "targets": torch.tensor(rec["targets"], dtype=torch.float32),
            "mass": torch.tensor(rec["mass"], dtype=torch.float32),
            "food_name": food_name,
        }
