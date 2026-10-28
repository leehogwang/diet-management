"""manifest(CSV) → (image, text_emb, has_image, target, mask) 배치."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .common import TARGETS
from .embeddings import load_cache, render_text


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_csv(path)


def load_manifests(cfg: dict, main_path: Path, text_only_path: Path | None) -> pd.DataFrame:
    df = _read_table(main_path)
    df["has_image"] = 1
    # 이미지 파일이 없는 행은 제외 (예: AIHub 122 는 라벨만 먼저 받고 이미지는 일부만 받는 경우)
    exists = df["image_path"].fillna("").astype(str).map(lambda p: bool(p) and os.path.exists(p))
    if not exists.all():
        print(f"이미지 파일이 없는 {int((~exists).sum())}/{len(df)}행 제외")
        df = df[exists]
    if df.empty:
        raise SystemExit(f"이미지가 있는 샘플이 없습니다: {main_path}. 이미지를 받은 뒤 manifest 를 다시 만드세요.")
    if cfg["data"].get("enable_text_only") and text_only_path and Path(text_only_path).exists():
        t = _read_table(text_only_path)
        t["has_image"] = 0
        df = pd.concat([df, t], ignore_index=True)
    df["text"] = [render_text(r, cfg["data"]["text_template"]) for _, r in df.iterrows()]
    return df


def split_train_val(df: pd.DataFrame, cfg: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """이미지 샘플을 train/val 로 나누고, 텍스트 전용 샘플은 모두 train 에 넣는다.

    split_by
      food_code : 음식 단위 분할. 296 라벨은 음식마다 영양값이 하나라서 이미지 단위로 나누면
                  val 에 같은 음식(=같은 정답)이 들어가 성능이 부풀려진다.
      official  : manifest 의 source_split(Training/Validation) 을 그대로 사용 (AIHub 122 기본값)
      group     : split_group 단위 무작위 분할 (같은 접시의 side/top 이 갈라지지 않게)
      image     : 이미지 단위 무작위 분할
    """
    img, txt = df[df.has_image == 1], df[df.has_image == 0]
    rng = np.random.default_rng(seed)
    ratio = cfg["data"]["val_ratio"]
    split_by = cfg["data"]["split_by"]
    if split_by == "official":
        is_val = img["source_split"] == "Validation"
    elif split_by == "group":
        groups = np.array(sorted(img["split_group"].unique()))
        rng.shuffle(groups)
        val_groups = set(groups[:max(1, int(round(len(groups) * ratio)))])
        is_val = img["split_group"].isin(val_groups)
    elif split_by == "food_code":
        codes = np.array(sorted(img["food_code"].unique()))
        rng.shuffle(codes)
        n_val = max(1, int(round(len(codes) * ratio)))
        val_codes = set(codes[:n_val])
        is_val = img["food_code"].isin(val_codes)
        # 텍스트 전용 샘플 중 val 음식과 이름이 같은 것은 누수 방지를 위해 제거
        val_names = set(img.loc[is_val, "food_name"])
        txt = txt[~txt["food_name"].isin(val_names)]
    else:
        is_val = pd.Series(rng.random(len(img)) < ratio, index=img.index)
    return pd.concat([img[~is_val], txt]), img[is_val]


class Nutrition296Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: dict, emb_cache_path: Path, target_stats: dict,
                 train: bool, image_cache_dir: Path | None = None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        size = cfg["data"]["image_size"]
        self.size = size
        self.use_bbox = cfg["data"].get("use_bbox_crop", False)
        self.cache_dir = image_cache_dir
        self.cache_max_side = cfg["data"].get("cache_max_side", 384)
        self.load_images = True  # 이미지 분기를 쓰지 않는 단계(stage1)에서는 False 로 두어 디코딩을 건너뜀
        cache = load_cache(emb_cache_path)
        missing = sorted(set(self.df["text"]) - set(cache))
        if missing:
            raise SystemExit(f"임베딩 캐시에 없는 텍스트 {len(missing)}개 (예: {missing[:3]}). n296.embeddings 를 먼저 실행하세요.")
        self.text_emb = {t: torch.tensor(cache[t], dtype=torch.float32) for t in set(self.df["text"])}
        self.mean = torch.tensor([target_stats[k]["mean"] for k in TARGETS], dtype=torch.float32)
        self.std = torch.tensor([target_stats[k]["std"] for k in TARGETS], dtype=torch.float32)

        if train and cfg.get("augmentation", {}).get("enabled"):
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
                transforms.ToTensor(),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(int(size * 256 / 224)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.df)

    def _open(self, r) -> Image.Image:
        src = Path(r["image_path"])
        key = src.name
        if self.use_bbox:
            key = f"bbox_{key}"
        if self.cache_dir is not None:
            sub = hashlib.md5(key.encode()).hexdigest()[:2]
            cp = self.cache_dir / sub / key
            if cp.exists():
                return Image.open(cp).convert("RGB")
        img = Image.open(src).convert("RGB")
        if self.use_bbox and not pd.isna(r.get("bbox_w", np.nan)):
            x, y, w, h = (float(r[k]) for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
            pad = 0.1 * max(w, h)
            img = img.crop((max(0, x - pad), max(0, y - pad), min(img.width, x + w + pad), min(img.height, y + h + pad)))
        if self.cache_dir is not None:
            img.thumbnail((self.cache_max_side, self.cache_max_side), Image.BICUBIC)
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_name(f".{os.getpid()}_{cp.name}")
            img.save(tmp, quality=95)
            tmp.replace(cp)
        return img

    def __getitem__(self, i):
        r = self.df.iloc[i]
        has_image = bool(r["has_image"])
        if has_image and self.load_images:
            image = self.tf(self._open(r))
        else:
            image = torch.zeros(3, self.size, self.size)
        y = torch.tensor([float(r[k]) if k in r and not pd.isna(r[k]) else np.nan for k in TARGETS], dtype=torch.float32)
        mask = ~torch.isnan(y)
        y_norm = torch.where(mask, (y - self.mean) / self.std, torch.zeros_like(y))
        return {
            "image": image,
            "text_emb": self.text_emb[r["text"]],
            "has_image": torch.tensor(has_image),
            "y": torch.nan_to_num(y), "y_norm": y_norm, "mask": mask,
            "food_code": str(r.get("food_code", "")), "food_name": str(r.get("food_name", "")),
        }


def compute_target_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for k in TARGETS:
        v = pd.to_numeric(df[k], errors="coerce").dropna()
        stats[k] = {"mean": float(v.mean()), "std": float(v.std() + 1e-6)}
    return stats
