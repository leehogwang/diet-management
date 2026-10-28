"""원본(세그멘테이션 없음) RGB + RealSense raw depth → 4채널 RGBD dataset.

cutout RGBD와의 차이:
- RGB: rgb.png 원본 (배경·접시 포함)
- Depth: depth_raw.png 전체 맵 (food 마스킹 없음), hole(0)만 중립값 대체
- 전역 표준화 통계는 전체 이미지 기준 (mean 3754 / std 941)

배포 시나리오(사진→바로 예측, 세그멘테이션 불필요) 검증용.
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

IMG_ROOT = Path(__file__).parent.parent / "data" / "nutrition5k" / "imagery" / "realsense_overhead"

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD  = [0.229, 0.224, 0.225]
# 전체 이미지 depth 통계 (train 300접시 표본, hole 제외)
DEPTH_MEAN = 3754.0
DEPTH_STD  = 941.0


class Nutrition5kRawRGBDDataset(Dataset):
    """cache=True면 resize까지 끝낸 256x256 4채널을 float16으로 들고 있는다.

    PNG 디코딩이 병목이라 에폭당 41초가 나왔다. 캐시하면 학습만 남는다.
    3,224장 전부라도 ~1.7GB이고 DataLoader 워커는 fork로 공유한다.
    crop·flip·normalize는 캐시 뒤에 그대로 적용되므로 증강 효과는 변하지 않는다.
    """
    def __init__(self, dish_ids: list[str], gt_dict: dict, augment: bool = False,
                 cache: bool = False):
        self.augment = augment
        self.samples: list[tuple[str, list[float]]] = []
        for did in dish_ids:
            if did not in gt_dict:
                continue
            if not (IMG_ROOT / did / "rgb.png").exists():
                continue
            if not (IMG_ROOT / did / "depth_raw.png").exists():
                continue
            self.samples.append((did, [gt_dict[did][t] for t in TARGETS]))
        self.cache = None
        if cache and self.samples:
            self.cache = torch.empty(len(self.samples), 4, 256, 256, dtype=torch.float16)
            for i, (did, _) in enumerate(self.samples):
                self.cache[i] = self._load(did).half()

    def __len__(self):
        return len(self.samples)

    def _load(self, did):
        rgb = Image.open(IMG_ROOT / did / "rgb.png").convert("RGB")
        rgb_np = np.array(rgb)

        try:                                          # 손상 depth → 중립 채널
            dep = np.array(Image.open(IMG_ROOT / did / "depth_raw.png")).astype(np.float32)
            if dep.shape != rgb_np.shape[:2]:
                raise ValueError("shape mismatch")
        except Exception:
            dep = np.full(rgb_np.shape[:2], DEPTH_MEAN, dtype=np.float32)
        dep = np.where(dep > 0, dep, DEPTH_MEAN)      # hole만 중립값
        dep = (dep - DEPTH_MEAN) / DEPTH_STD

        rgb_t = TF.to_tensor(rgb)                     # (3,H,W)
        dep_t = torch.from_numpy(dep)[None]           # (1,H,W)
        x = torch.cat([rgb_t, dep_t], 0)

        # cutout RGBD와 동일한 geometric 파이프라인
        return TF.resize(x, [256, 256], antialias=True)

    def __getitem__(self, idx):
        did, gt_vals = self.samples[idx]
        x = self.cache[idx].float() if self.cache is not None else self._load(did)
        if self.augment:
            i, j, h, w = RandomCrop.get_params(x, (224, 224))
            x = TF.crop(x, i, j, h, w)
            if random.random() < 0.5:
                x = TF.hflip(x)
        else:
            x = TF.center_crop(x, [224, 224])

        x[:3] = TF.normalize(x[:3], RGB_MEAN, RGB_STD)
        return x, torch.tensor(gt_vals, dtype=torch.float32)
