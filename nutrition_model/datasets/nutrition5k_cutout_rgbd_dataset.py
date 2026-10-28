"""cutout RGB + RealSense raw depth → 4채널 RGBD dataset.

- RGB: cutout(배경=0), ImageNet 정규화
- Depth: depth_raw.png(uint16, 0.1mm), food 영역만, 전역 표준화, 배경/홀=0(중립)
- RGB/Depth에 동일한 geometric 증강 적용 (crop/flip 동기화)
"""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import RandomCrop
from PIL import Image

from datasets.nutrition5k_rgb_dataset import TARGETS
from datasets.nutrition5k_cutout_dataset import CUTOUT_DIR

IMG_ROOT = Path(__file__).parent.parent / "data" / "nutrition5k" / "imagery" / "realsense_overhead"

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]
# food 영역 depth 전역 통계 (표본 추정)
DEPTH_MEAN = 3240.0
DEPTH_STD  = 714.0


class Nutrition5kCutoutRGBDDataset(Dataset):
    def __init__(self, dish_ids: list[str], gt_dict: dict, augment: bool = False):
        self.augment = augment
        self.samples: list[tuple[str, list[float]]] = []
        for did in dish_ids:
            if did not in gt_dict:
                continue
            if not (CUTOUT_DIR / f"{did}_cutout.png").exists():
                continue
            if not (IMG_ROOT / did / "depth_raw.png").exists():
                continue
            self.samples.append((did, [gt_dict[did][t] for t in TARGETS]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        did, gt_vals = self.samples[idx]
        rgb = Image.open(CUTOUT_DIR / f"{did}_cutout.png").convert("RGB")
        rgb_np = np.array(rgb)                       # (H,W,3)
        food = rgb_np.sum(2) > 0                     # cutout 전경 = food

        try:                                          # 손상/빈 depth → 중립(0) 채널로 대체
            dep = np.array(Image.open(IMG_ROOT / did / "depth_raw.png")).astype(np.float32)
            if dep.shape != rgb_np.shape[:2]:
                raise ValueError("shape mismatch")
        except Exception:
            dep = np.full(rgb_np.shape[:2], DEPTH_MEAN, dtype=np.float32)
        dep = np.where(food & (dep > 0), dep, DEPTH_MEAN)
        dep = (dep - DEPTH_MEAN) / DEPTH_STD         # 배경/홀 → 0

        rgb_t = TF.to_tensor(rgb)                    # (3,H,W) in [0,1]
        dep_t = torch.from_numpy(dep)[None]          # (1,H,W)
        x = torch.cat([rgb_t, dep_t], 0)             # (4,H,W)

        # geometric (RGB/Depth 동시): resize 256 → crop 224 (+train시 flip)
        x = TF.resize(x, [256, 256], antialias=True)
        if self.augment:
            i, j, h, w = RandomCrop.get_params(x, (224, 224))
            x = TF.crop(x, i, j, h, w)
            if random.random() < 0.5:
                x = TF.hflip(x)
        else:
            x = TF.center_crop(x, [224, 224])

        # RGB 3채널만 ImageNet 정규화 (depth는 이미 표준화됨)
        x[:3] = TF.normalize(x[:3], RGB_MEAN, RGB_STD)
        return x, torch.tensor(gt_vals, dtype=torch.float32)
