"""
멀티모달(이미지+음식명) 절대량 영양 회귀용 데이터 파이프라인.

text_baseline_the_best의 텍스트 전용 영양DB(load_combined_db)와
AIHub 296 데이터셋의 이미지(dataset_296.py)를 하나의 food_name 공간에서 합친다.

타깃은 100g당 값이 아니라 **1인분 절대량**이다. 296은 Q레벨이 없고 대신 샘플셋별
실측 중량(image_info.weight)이 있어 그 값으로 직접 스케일한다:
  abs_target = 100g당 값 × weight(g) / 100
- 이미지 샘플: weight = 296 라벨의 실측 중량
- 텍스트 전용 샘플(이미지 없음): weight = DB의 1회제공량(mass)

296 라벨에는 탄수화물이 없다. 칼로리(cal)/지방(fat)/단백질(prot)만 직접 지도학습 타깃으로 쓰고,
carb는 Atwater 역산 참고용 GT로만 따로 들고 다닌다 (TARGET_COLS에는 없음).

Split은 dataset_296.load_split()이 정한 폴더 기반 분할을 그대로 쓴다
(Training-only 음식 = test, Training∩Validation 음식 = train/val).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset_aihub import ResizeWithPadding  # noqa: E402
import dataset_296  # noqa: E402

import importlib.util
_spec = importlib.util.spec_from_file_location("tb_train", Path(__file__).parent / "train.py")
tb_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tb_train)  # load_combined_db, get_embeddings 재사용

TARGET_COLS = ["cal", "fat", "prot"]  # 순서 고정 — 모델 출력 순서와 일치. carb는 Atwater 역산이라 제외


def build_dataset(cfg_data: dict, view: str = dataset_296.VIEW_SIDE):
    """전체 파이프라인 실행: 영양DB 로드 + 296 이미지 매니페스트 로드/필터링.

    Returns:
        splits: dict split -> {"image": list[record], "text": DataFrame}
    """
    nutrition_df = tb_train.load_combined_db(cfg_data)

    splits_296 = dataset_296.load_split(
        root=cfg_data.get("root_296", dataset_296.DEFAULT_296_ROOT),
        view=view,
        rebuild=cfg_data.get("rebuild_manifest", False),
    )
    for name, df in splits_296.items():
        print(f"[multimodal-296] {name}: rows={len(df)} food_codes={df['food_code'].nunique()} "
              f"food_names={df['food_name'].nunique()}")

    # 텍스트 전용 브랜치는 기본 ON — 296에 없는 음식명으로 텍스트 인코더에 다양성을 준다.
    # test(구조적 train-only) 음식명은 반드시 제외해 텍스트 경로로도 누수되지 않게 한다.
    enable_text_only = cfg_data.get("enable_text_only", True)
    if enable_text_only:
        test_food_names = set(splits_296["test"]["food_name"].unique())
        text_only_df = nutrition_df[~nutrition_df["식품명"].isin(test_food_names)].reset_index(drop=True)
    else:
        text_only_df = nutrition_df.iloc[0:0].reset_index(drop=True)
    print(f"[multimodal-296] text-only rows(train에 배정)={len(text_only_df)} "
          f"(enable_text_only={enable_text_only})")

    empty_text = nutrition_df.iloc[0:0].reset_index(drop=True)
    splits = {
        "train": {"image": splits_296["train"].to_dict("records"), "text": text_only_df},
        "val": {"image": splits_296["val"].to_dict("records"), "text": empty_text},
        "test": {"image": splits_296["test"].to_dict("records"), "text": empty_text},
    }
    for split_name, s in splits.items():
        print(f"[multimodal] split={split_name}: image_samples={len(s['image'])}, "
              f"text_only_rows={len(s['text'])}")

    return splits, nutrition_df


def build_name_to_emb(splits: dict, emb_cfg: dict) -> dict:
    """splits에 등장하는 모든 food_name의 임베딩을 한 번만 로드해 메모리 dict로 반환.

    get_embeddings()는 캐시에 없는 이름이 있으면 API를 호출한다 — 여기서 한 번만 호출되고
    이후 Dataset.__getitem__은 이 dict만 참조하므로 학습 중 추가 API 호출/디스크 재읽기가 없다.
    """
    all_names = set()
    for split in splits.values():
        all_names.update(r["food_name"] for r in split["image"])
        all_names.update(split["text"]["식품명"])
    names = sorted(all_names)
    api_key_env = emb_cfg.get("api_key_env", "GEMINI_KEY")
    embs = tb_train.get_embeddings(names, emb_cfg["model"], emb_cfg["cache_path"], api_key_env)
    return {n: e for n, e in zip(names, embs)}


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


class MultimodalNutritionDataset(Dataset):
    """이미지 샘플(296) + 텍스트 전용 행(영양DB)을 하나의 인덱스 공간에서 섞어 낸다.

    각 아이템: {"food_name", "image"(Tensor|None), "has_image"(bool),
    "target"(np.ndarray[3] = cal,fat,prot), "carb_gt"(float, Atwater 역산 참고용, 학습엔 안 씀),
    "text_emb"(np.ndarray)}. 임베딩은 디스크 캐시를 배치마다 다시 읽지 않도록 미리 메모리에
    올려서 (`name_to_emb`) 넘겨받는다 — text_baseline_the_best/train.py의 emb_cache 워밍업 패턴과 동일.
    """

    def __init__(self, image_records: list[dict], text_df: pd.DataFrame,
                 nutrition_df: pd.DataFrame, name_to_emb: dict, train: bool):
        self.image_records = image_records
        self.text_rows = text_df.to_dict("records")
        # mass는 타깃이 아니지만 절대량 스케일링(100g당 값 × mass)에 필요해서 조회 컬럼엔 남긴다.
        self.nutrition_by_name = nutrition_df.set_index("식품명")[TARGET_COLS + ["carb", "mass"]].to_dict("index")
        self.name_to_emb = name_to_emb
        self.transform = _IMAGE_TRANSFORM["train" if train else "eval"]

    def __len__(self):
        return len(self.image_records) + len(self.text_rows)

    def _image_target(self, rec: dict) -> tuple[np.ndarray, float, float]:
        """296 레코드: e100/fat100/pro100/carb100 × weight(실측 g)/100 = 절대량."""
        scale = rec["weight"] / 100.0
        target = np.array([rec["e100"] * scale, rec["fat100"] * scale, rec["pro100"] * scale],
                           dtype=np.float32)
        carb_gt = float(rec["carb100"] * scale)
        mass_gt = float(rec["weight"])
        return target, carb_gt, mass_gt

    def _text_target(self, food_name: str) -> tuple[np.ndarray, float, float]:
        """DB 전용 행: 100g당 값 × 1회제공량(mass)/100 = 절대량 (portion_factor는 항상 1.0)."""
        row = self.nutrition_by_name[food_name]
        mass = float(row["mass"])
        target = np.array([row[k] * mass / 100 for k in TARGET_COLS], dtype=np.float32)
        carb_gt = float(row["carb"] * mass / 100)
        return target, carb_gt, mass

    def __getitem__(self, idx):
        if idx < len(self.image_records):
            rec = self.image_records[idx]
            image = ImageOps.exif_transpose(Image.open(rec["image_path"])).convert("RGB")
            bbox = (rec.get("bbox_x"), rec.get("bbox_y"), rec.get("bbox_w"), rec.get("bbox_h"))
            if all(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in bbox):
                x, y, w, h = bbox
                image = image.crop((x, y, x + w, y + h))
            image = self.transform(image)
            target, carb_gt, mass_gt = self._image_target(rec)
            emb = self.name_to_emb[rec["food_name"]]
            return {"food_name": rec["food_name"], "image": image, "has_image": True,
                    "target": target, "carb_gt": carb_gt, "mass_gt": mass_gt, "text_emb": emb}
        else:
            row = self.text_rows[idx - len(self.image_records)]
            target, carb_gt, mass_gt = self._text_target(row["식품명"])
            emb = self.name_to_emb[row["식품명"]]
            return {"food_name": row["식품명"], "image": None, "has_image": False,
                    "target": target, "carb_gt": carb_gt, "mass_gt": mass_gt, "text_emb": emb}


def collate_fn(batch: list[dict]) -> dict:
    images = torch.stack([
        b["image"] if b["has_image"] else torch.zeros(3, 224, 224) for b in batch
    ])
    has_image = torch.tensor([b["has_image"] for b in batch], dtype=torch.bool)
    food_names = [b["food_name"] for b in batch]
    targets = torch.from_numpy(np.stack([b["target"] for b in batch]))
    carb_gt = torch.tensor([b["carb_gt"] for b in batch], dtype=torch.float32)
    text_emb = torch.from_numpy(np.stack([b["text_emb"] for b in batch]).astype(np.float32))
    return {"images": images, "has_image": has_image, "food_names": food_names,
            "targets": targets, "carb_gt": carb_gt, "text_emb": text_emb}
