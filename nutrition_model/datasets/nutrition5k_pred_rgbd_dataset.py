"""원본 RGB + Depth Anything 예측 depth → 4채널 RGBD dataset (배포 시나리오).

raw RGBD와의 차이: depth가 RealSense 실측이 아니라 단안 예측(npy).
- 예측 depth는 hole 없음, 상대 스케일 (Depth Anything은 값이 클수록 가까움)
- 전역 표준화 통계는 gen_pred_depth.py가 만든 stats.json에서 로드
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import RandomCrop
from PIL import Image

from datasets.nutrition5k_rgb_dataset import TARGETS

IMG_ROOT = Path(__file__).parent.parent / "data" / "nutrition5k" / "imagery" / "realsense_overhead"
PRED_DIR = Path(__file__).parent.parent / "data" / "nutrition5k_pred_depth"

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]


def _load_stats():
    s = json.loads((PRED_DIR / "stats.json").read_text())
    return s["mean"], s["std"]


class Nutrition5kPredRGBDDataset(Dataset):
    def __init__(self, dish_ids: list[str], gt_dict: dict, augment: bool = False):
        self.augment = augment
        self.d_mean, self.d_std = _load_stats()
        self.samples: list[tuple[str, list[float]]] = []
        for did in dish_ids:
            if did not in gt_dict:
                continue
            if not (IMG_ROOT / did / "rgb.png").exists():
                continue
            if not (PRED_DIR / f"{did}.npy").exists():
                continue
            self.samples.append((did, [gt_dict[did][t] for t in TARGETS]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        did, gt_vals = self.samples[idx]
        rgb = Image.open(IMG_ROOT / did / "rgb.png").convert("RGB")
        dep = np.load(PRED_DIR / f"{did}.npy").astype(np.float32)
        dep = (dep - self.d_mean) / self.d_std

        x = torch.cat([TF.to_tensor(rgb), torch.from_numpy(dep)[None]], 0)

        # raw RGBD와 동일한 geometric 파이프라인
        x = TF.resize(x, [256, 256], antialias=True)
        if self.augment:
            i, j, h, w = RandomCrop.get_params(x, (224, 224))
            x = TF.crop(x, i, j, h, w)
            if random.random() < 0.5:
                x = TF.hflip(x)
        else:
            x = TF.center_crop(x, [224, 224])

        x[:3] = TF.normalize(x[:3], RGB_MEAN, RGB_STD)
        return x, torch.tensor(gt_vals, dtype=torch.float32)
