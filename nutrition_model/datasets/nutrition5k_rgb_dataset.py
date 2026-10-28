"""Nutrition5k overhead RGB dataset for regression."""

import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).parent.parent

METADATA_CSVS = [
    ROOT / "data" / "nutrition5k" / "metadata" / "dish_metadata_cafe1.csv",
    ROOT / "data" / "nutrition5k" / "metadata" / "dish_metadata_cafe2.csv",
]
IMAGERY_ROOT = ROOT / "data" / "nutrition5k" / "imagery" / "realsense_overhead"
DEPTH_FEATURES_JSON = ROOT / "outputs" / "nutrition5k_depth_features.json"

TARGETS = ["kcal", "mass_g", "fat_g", "carb_g", "protein_g"]

TRAIN_MEAN = [0.485, 0.456, 0.406]
TRAIN_STD  = [0.229, 0.224, 0.225]


def load_gt_dict() -> dict[str, dict]:
    """dish_id → {kcal, mass_g, fat_g, carb_g, protein_g}"""
    gt = {}
    for csv_path in METADATA_CSVS:
        with open(csv_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                dish_id = parts[0]
                try:
                    gt[dish_id] = {
                        "kcal":      float(parts[1]),
                        "mass_g":    float(parts[2]),
                        "fat_g":     float(parts[3]),
                        "carb_g":    float(parts[4]),
                        "protein_g": float(parts[5]),
                    }
                except ValueError:
                    continue
    return gt


def load_split_ids(split: str) -> list[str]:
    """depth_features.json 기준 split dish_id 목록."""
    data = json.load(open(DEPTH_FEATURES_JSON))
    return [r["dish_id"] for r in data["results"] if r.get("split") == split]


class Nutrition5kRGBDataset(Dataset):
    def __init__(
        self,
        dish_ids: list[str],
        gt_dict: dict[str, dict],
        augment: bool = False,
    ):
        self.gt_dict = gt_dict
        self.augment = augment

        # GT 있고 이미지 존재하는 dish만
        self.samples: list[tuple[Path, dict]] = []
        for did in dish_ids:
            if did not in gt_dict:
                continue
            img_path = IMAGERY_ROOT / did / "rgb.png"
            if not img_path.exists():
                continue
            self.samples.append((img_path, gt_dict[did]))

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, gt = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x   = self.transform(img)
        # log1p 변환 — 넓은 범위의 양수 타깃을 스케일 통일
        y   = torch.tensor(
            [math.log1p(gt[t]) for t in TARGETS], dtype=torch.float32
        )
        return x, y

    @staticmethod
    def decode(log_pred: torch.Tensor) -> torch.Tensor:
        """log1p 예측값 → 원래 단위 (expm1)."""
        return torch.expm1(log_pred.clamp(min=0))


def make_train_val_split(
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """train split을 다시 train/val로 나눔 (depth_features에 val이 없으므로)."""
    train_ids = load_split_ids("train")
    rng = random.Random(seed)
    rng.shuffle(train_ids)
    n_val = int(len(train_ids) * val_ratio)
    return train_ids[n_val:], train_ids[:n_val]
