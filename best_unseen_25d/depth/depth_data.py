"""깊이(UniDepth V2 사전계산) 데이터 연결부. 기존 데이터셋 파일은 수정하지 않고 감싸서 사용한다.

- 저장 형식: DEPTH_DIR/depth_all.npy (N,112,112 float16, 미터 단위, 패딩=0), depth_keys.json, depth_meta.npy (N,2: [crop 기준 fx(px@112), 원본 이미지 긴 변 px])
- 기존 데이터셋의 뒤집기 증강을 끄고(transform 을 평가용으로 교체), 이미지와 깊이를 같은 난수로 함께 뒤집는다.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DEPTH_DIR = Path(os.environ.get("DEPTH_DIR", "/mnt/data/food_leehogwang/depth_cache_unidepthv2"))  # 테스트 시 환경변수로 교체
DEPTH_SIZE = 112


def record_key(r: dict) -> str:
    if "zip_path" in r:
        return f"{r['zip_path']}::{r['filename']}"
    return str(r["image_path"])


class DepthStore:
    """사전계산된 깊이를 memmap 으로 열어 key -> (depth[112,112], meta[2]) 제공. 워커 간 공유 가능."""

    def __init__(self, depth_dir: Path = DEPTH_DIR):
        self.depth = np.load(depth_dir / "depth_all.npy", mmap_mode="r")
        self.meta = np.load(depth_dir / "depth_meta.npy")
        keys = json.loads((depth_dir / "depth_keys.json").read_text(encoding="utf-8"))
        self.index = {k: i for i, k in enumerate(keys)}

    def get(self, r: dict):
        i = self.index.get(record_key(r))
        if i is None:  # 깊이 없음 → 0 맵 (모델이 has_depth=False 로 처리)
            return np.zeros((DEPTH_SIZE, DEPTH_SIZE), np.float32), np.zeros(2, np.float32), False
        return np.asarray(self.depth[i], dtype=np.float32), self.meta[i].astype(np.float32), True


class WithDepth(Dataset):
    """기존 데이터셋 인스턴스를 감싸 깊이를 추가. flip_v=True 면 상하 뒤집기도 (OmniFood 학습 증강과 동일)."""

    def __init__(self, base: Dataset, store: DepthStore, eval_transform, train: bool, flip_v: bool = False):
        self.base, self.store, self.train, self.flip_v = base, store, train, flip_v
        self.base.transform = eval_transform  # 기존 무작위 뒤집기 제거 (색상 증강은 포기, 텍스트 증강은 유지)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        depth, meta, ok = self.store.get(self.base.records[idx])
        img, dep = item["image"], torch.from_numpy(depth)[None]
        if self.train:
            if random.random() < 0.5:
                img, dep = img.flip(-1), dep.flip(-1)
            if self.flip_v and random.random() < 0.5:
                img, dep = img.flip(-2), dep.flip(-2)
        item["image"], item["depth"], item["depth_meta"], item["has_depth"] = img, dep, torch.from_numpy(meta), ok
        return item


def depth_collate(batch, base_collate):
    out = base_collate(batch)
    out["depth"] = torch.stack([b["depth"] for b in batch])
    out["depth_meta"] = torch.stack([b["depth_meta"] for b in batch])
    out["has_depth"] = torch.tensor([bool(b["has_depth"]) for b in batch])
    return out
