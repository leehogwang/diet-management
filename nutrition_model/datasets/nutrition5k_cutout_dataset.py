"""Nutrition5k cutout (배경 제거) 이미지 기반 dataset."""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from datasets.nutrition5k_rgb_dataset import load_gt_dict, load_split_ids, TARGETS

CUTOUT_DIR = Path(__file__).parent.parent / "data" / "nutrition5k_seg_labels" / "segmentation-labels" / "cutout"

TRAIN_MEAN = [0.485, 0.456, 0.406]
TRAIN_STD  = [0.229, 0.224, 0.225]


class Nutrition5kCutoutDataset(Dataset):
    """
    cutout 이미지(배경 픽셀=0) 입력, 원본 스케일 GT 출력.
    log1p 미적용 — normalized MAE loss와 함께 사용.
    """

    def __init__(self, dish_ids: list[str], gt_dict: dict, augment: bool = False):
        self.samples: list[tuple[Path, list[float]]] = []
        for did in dish_ids:
            if did not in gt_dict:
                continue
            img_path = CUTOUT_DIR / f"{did}_cutout.png"
            if not img_path.exists():
                continue
            gt = gt_dict[did]
            self.samples.append((img_path, [gt[t] for t in TARGETS]))

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_vals = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(gt_vals, dtype=torch.float32)
        return x, y


def load_cutout_split_ids() -> tuple[list[str], list[str]]:
    """cutout 파일이 존재하는 dish_id를 train/test로 반환."""
    all_cutout_ids = {p.stem.replace("_cutout", "") for p in CUTOUT_DIR.glob("*_cutout.png")}
    train_ids = [d for d in load_split_ids("train") if d in all_cutout_ids]
    test_ids  = [d for d in load_split_ids("test")  if d in all_cutout_ids]
    return train_ids, test_ids
