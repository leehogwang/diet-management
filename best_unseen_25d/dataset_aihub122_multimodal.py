"""
AIHub 122 한식 도메인 멀티모달 데이터셋 (Zip 기반 고속 로더).

- 음식명(food_name) 기준 Zero-Leakage Split:
  foodlist_train_122.txt (58종) vs foodlist_test_122.txt (25종) 완벽 격리.
- 58종 학습 음식군 내에서 결정론적(seed=42)으로 50종 Train / 8종 Val로 분할.
- 측면(side) 뷰만 사용.
- bbox_cache.json에 사전 계산된 YOLO BBox 적용.
- Q1~Q5 배율(0.25~1.25)을 적용한 4대 영양소(cal, fat, prot, carb) 절대량 지도학습.
- carb_mask = 1.0 (4대 영양소 전량 지도학습).
"""
import io
import json
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset_aihub import ResizeWithPadding, FOOD_NAME_ALIASES, PORTION_FACTORS, _load_nutrition_table

DEFAULT_122_ROOT = Path("/mnt/data/food_leehogwang/122.음식_분류를_위한_음식종류_및_양에_따른_칼로리_데이터셋(재료,_양념,_완제품_등)")
DEFAULT_TRAIN_IMAGE_ROOT = DEFAULT_122_ROOT / "01.데이터/1.Training/원천데이터/양추정_이미지_TRAIN/image"
DEFAULT_VAL_IMAGE_ROOT = DEFAULT_122_ROOT / "01.데이터/2.Validation/원천데이터/image"
DEFAULT_NUTRITION_XLSX = DEFAULT_122_ROOT / "01.데이터/9.칼로리데이터셋/44.음식분류_AI_데이터_영양DB.xlsx"
DEFAULT_BBOX_CACHE = Path(__file__).parent / "text_baseline_the_best/bbox_cache.json"


def collect_122_records_from_cache(
    bbox_cache_path: Path,
    train_image_root: Path,
    val_image_root: Path,
    nutrition_table: dict,
    allowed_food_names: set,
) -> List[dict]:
    cache = json.loads(bbox_cache_path.read_text())
    records = []

    for k, box in cache.items():
        p = Path(k)
        food_raw = p.parent.parent.name
        if food_raw not in allowed_food_names:
            continue

        q_dir = p.parent.name  # "Q1" ~ "Q5"
        portion = q_dir[1:] if q_dir.startswith("Q") else q_dir
        if portion not in PORTION_FACTORS:
            continue

        food_std = FOOD_NAME_ALIASES.get(food_raw, food_raw)
        if food_std not in nutrition_table:
            continue

        fname = p.name
        if "2.Validation" in k:
            zip_path = val_image_root / food_raw / q_dir / f"{q_dir}.zip"
        else:
            zip_path = train_image_root / food_raw / q_dir / f"{q_dir}.zip"
        if not zip_path.exists():
            continue

        factor = float(PORTION_FACTORS[portion])
        nut = nutrition_table[food_std]
        target = np.array([
            nut["calories"] * factor,
            nut["fat"] * factor,
            nut["protein"] * factor,
            nut["carb"] * factor,
            nut["mass"] * factor,
        ], dtype=np.float32)

        records.append({
            "zip_path": str(zip_path),
            "filename": fname,
            "raw_food_name": food_raw,
            "food_name": food_std,
            "portion": portion,
            "bbox": box,
            "targets": target,
            "mass": nut["mass"] * factor,
        })

    return records


_IMAGE_TRANSFORM = {
    "train": transforms.Compose([
        ResizeWithPadding(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "eval": transforms.Compose([
        ResizeWithPadding(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}


class AIHub122MultimodalDataset(Dataset):
    def __init__(self, records: List[dict], name_to_emb: dict, train: bool = True, text_variants: Optional[dict] = None):
        self.records = records
        self.name_to_emb = name_to_emb
        self.train = train
        self.text_variants = text_variants
        self.transform = _IMAGE_TRANSFORM["train" if train else "eval"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        with zipfile.ZipFile(rec["zip_path"], "r") as z:
            fname = rec["filename"]
            try:
                data = z.read(fname)
            except KeyError:
                try:
                    cp437_fname = fname.encode("cp949").decode("cp437", errors="ignore")
                    data = z.read(cp437_fname)
                except KeyError:
                    suffix = fname.split("_")[-1]
                    matches = [n for n in z.namelist() if n.endswith(suffix)]
                    if matches:
                        data = z.read(matches[0])
                    else:
                        raise KeyError(f"Cannot find {fname} or {suffix} in {rec['zip_path']}")
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")

        box = rec.get("bbox")
        if box is not None and len(box) == 4:
            x1, y1, x2, y2 = box
            w, h = image.size
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= w + 10 and y2 <= h + 10:
                image = image.crop((x1, y1, x2, y2))

        image = self.transform(image)
        
        emb = None
        if self.train and self.text_variants:
            food_info = self.text_variants.get(rec["food_name"])
            if food_info and food_info.get("variants") and random.random() < 0.5:
                v_text = random.choice(food_info["variants"])
                emb = self.name_to_emb.get(v_text)
        if emb is None:
            emb = self.name_to_emb.get(rec["food_name"], self.name_to_emb.get(rec["raw_food_name"]))

        food_raw = rec["raw_food_name"]
        portion = rec["portion"]
        fname = rec["filename"]
        return {
            "dish_id": f"{food_raw}_Q{portion}_{fname}",
            "food_name": rec["food_name"],
            "image": image,
            "has_image": True,
            "target": rec["targets"],
            "carb_mask": 1.0,
            "text_emb": emb,
            "portion_label": int(portion) - 1,  # Q1~Q5 -> 0~4 (분량 분류 보조 손실용)
        }


def collate_fn(batch: List[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch])
    has_image = torch.tensor([b["has_image"] for b in batch], dtype=torch.bool)
    food_names = [b["food_name"] for b in batch]
    dish_ids = [b.get("dish_id", "") for b in batch]
    targets = torch.from_numpy(np.stack([b["target"] for b in batch]))
    carb_mask = torch.tensor([b["carb_mask"] for b in batch], dtype=torch.float32)
    text_emb = torch.from_numpy(np.stack([b["text_emb"] for b in batch]).astype(np.float32))
    return {
        "images": images,
        "has_image": has_image,
        "food_names": food_names,
        "dish_ids": dish_ids,
        "targets": targets,
        "carb_mask": carb_mask,
        "text_emb": text_emb,
    }


def load_aihub122_dataset(
    train_food_list_path: Path = Path("foodlist_train_122.txt"),
    test_food_list_path: Path = Path("foodlist_test_122.txt"),
    nutrition_xlsx: Path = DEFAULT_NUTRITION_XLSX,
    bbox_cache_path: Path = DEFAULT_BBOX_CACHE,
    emb_cache_path: Optional[Path] = None,
    val_foods_count: int = 8,
    seed: int = 42,
    val_foods: Optional[List[str]] = None,  # 지정 시 무작위 분할 대신 이 음식들을 검증셋으로 (k-fold용)
) -> Tuple[Dict[str, List[dict]], dict]:
    train_foods_all = [x.strip() for x in train_food_list_path.read_text().splitlines() if x.strip()]
    test_foods = set(x.strip() for x in test_food_list_path.read_text().splitlines() if x.strip())

    if val_foods is not None:
        val_foods = set(val_foods)
        assert val_foods <= set(train_foods_all), f"val_foods not in train list: {val_foods - set(train_foods_all)}"
        train_foods = set(train_foods_all) - val_foods
    else:
        rng = random.Random(seed)
        train_foods_shuffled = list(dict.fromkeys(train_foods_all))
        rng.shuffle(train_foods_shuffled)
        val_foods = set(train_foods_shuffled[:val_foods_count])
        train_foods = set(train_foods_shuffled[val_foods_count:])

    assert len(train_foods & test_foods) == 0, "Leakage detected: train_foods overlaps test_foods!"
    assert len(val_foods & test_foods) == 0, "Leakage detected: val_foods overlaps test_foods!"
    assert len(train_foods & val_foods) == 0, "Leakage detected: train_foods overlaps val_foods!"

    print(f"[aihub122] Zero-Leakage Split: {len(train_foods)} Train foods, {len(val_foods)} Val foods, {len(test_foods)} Test foods.")

    nut_table = _load_nutrition_table(nutrition_xlsx)
    train_root = DEFAULT_TRAIN_IMAGE_ROOT
    val_root = DEFAULT_VAL_IMAGE_ROOT

    train_records = collect_122_records_from_cache(bbox_cache_path, train_root, val_root, nut_table, train_foods)
    val_records = collect_122_records_from_cache(bbox_cache_path, train_root, val_root, nut_table, val_foods)
    test_records = collect_122_records_from_cache(bbox_cache_path, train_root, val_root, nut_table, test_foods)

    print(f"[aihub122] side records: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    if emb_cache_path is None:
        emb_cache_path = Path(__file__).parent / "text_baseline_the_best/emb_cache.json"
    name_to_emb = json.loads(emb_cache_path.read_text()) if (emb_cache_path and emb_cache_path.exists()) else {}

    splits = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }
    return splits, name_to_emb
