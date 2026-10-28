import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from ingredient_augmentation import IngredientAugmentor


DEFAULT_AIHUB_ROOT = (
    "/home/202421012/food/122.음식_분류를_위한_음식종류_및_양에_따른_칼로리_데이터셋"
    "(재료,_양념,_완제품_등)"
)
DEFAULT_AIHUB_TRAIN_ROOT = (
    DEFAULT_AIHUB_ROOT + "/01.데이터/1.Training/원천데이터/양추정_이미지_TRAIN/image"
)
DEFAULT_AIHUB_VALIDATION_ROOT = (
    DEFAULT_AIHUB_ROOT + "/01.데이터/2.Validation/원천데이터/image"
)
DEFAULT_AIHUB_NUTRITION_XLSX = (
    DEFAULT_AIHUB_ROOT + "/01.데이터/9.칼로리데이터셋/44.음식분류_AI_데이터_영양DB.xlsx"
)
DEFAULT_TRAIN_FOOD_LIST = "foodlist_train_122.txt"
DEFAULT_TEST_FOOD_LIST = "foodlist_test_122.txt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIEW_RE = re.compile(r"^(side|top)_", re.IGNORECASE)
PORTION_RE = re.compile(r"_Q([1-5])(?:[_\s]|$)", re.IGNORECASE)
Q_DIR_RE = re.compile(r"^Q([1-5])$", re.IGNORECASE)

PORTION_FACTORS = {
    "1": 0.25,
    "2": 0.50,
    "3": 0.75,
    "4": 1.00,
    "5": 1.25,
}


class ResizeWithPadding:
    """Resize an RGB PIL image to fit inside a square without cropping context."""

    def __init__(self, size=224, fill=(124, 116, 104)):
        self.size = int(size)
        self.fill = tuple(fill)

    def __call__(self, image):
        width, height = image.size
        scale = min(self.size / width, self.size / height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(
            resized,
            ((self.size - resized.width) // 2, (self.size - resized.height) // 2),
        )
        return canvas

FOOD_NAME_ALIASES = {
    "감자튀김(스틱형)": "감자튀김",
    "감자튀김(웨지감자)": "감자튀김",
    "김밥": "일반김밥",
    "물회": "생선물회",
    "영양돌솥밥": "돌솥밥",
    "오므라이스(사각배달)": "오므라이스",
    "오므라이스(접시)": "오므라이스",
    "토마토스파게티": "토마토소스스파게티",
}


def _load_nutrition_table(nutrition_xlsx):
    df = pd.read_excel(nutrition_xlsx)
    required = ["음 식 명", "중량(g)", "에너지(kcal)", "탄수화물(g)", "지방(g)", "단백질(g)"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing nutrition columns in {nutrition_xlsx}: {missing}")

    table = {}
    for _, row in df.iterrows():
        food_name = str(row["음 식 명"]).strip()
        if not food_name or food_name == "nan":
            continue
        table[food_name] = {
            "mass": float(row["중량(g)"]),
            "calories": float(row["에너지(kcal)"]),
            "carb": float(row["탄수화물(g)"]),
            "fat": float(row["지방(g)"]),
            "protein": float(row["단백질(g)"]),
        }
    return table


def _food_name_from_path(path):
    q_match = Q_DIR_RE.match(path.parent.name)
    if q_match and path.parent.parent.name:
        return path.parent.parent.name
    return None


def _portion_from_path(path):
    q_match = Q_DIR_RE.match(path.parent.name)
    if q_match:
        return q_match.group(1)
    return None


def _fallback_food_name_from_stem(stem):
    without_view = VIEW_RE.sub("", stem, count=1)
    before_q = re.split(r"_Q[1-5](?:[_\s]|$)", without_view, maxsplit=1, flags=re.IGNORECASE)[0]
    parts = [part.strip() for part in before_q.split("_") if part.strip()]
    return parts[-1] if parts else ""


def _parse_aihub_filename(path):
    view_match = VIEW_RE.match(path.name)
    if view_match is None:
        return None

    view = view_match.group(1).lower()
    stem = path.stem
    portion_match = PORTION_RE.search(stem)
    portion = portion_match.group(1) if portion_match else _portion_from_path(path)
    if portion not in PORTION_FACTORS:
        return None

    raw_food_name = _food_name_from_path(path) or _fallback_food_name_from_stem(stem)
    if not raw_food_name:
        return None

    food_name = FOOD_NAME_ALIASES.get(raw_food_name, raw_food_name)

    return {
        "view": view,
        "portion": portion,
        "portion_factor": PORTION_FACTORS[portion],
        "raw_food_name": raw_food_name,
        "food_name": food_name,
    }


def _normalize_roots(roots):
    if roots is None:
        return []
    if isinstance(roots, (str, Path)):
        return [Path(roots)]
    return [Path(root) for root in roots]


def _iter_unique_image_paths(root):
    """
    AIHub validation folders can contain duplicate filenames both at the top
    level and under food/Q subdirectories. Keep one path per filename and
    prefer the shortest relative path so the top-level copy wins when present.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"AIHub image root not found: {root}")

    def path_score(path):
        has_food_folder = _food_name_from_path(path) is not None
        return (0 if has_food_folder else 1, len(path.relative_to(root).parts), str(path))

    by_name = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        current = by_name.get(path.name)
        if current is None:
            by_name[path.name] = path
            continue

        if path_score(path) < path_score(current):
            by_name[path.name] = path

    return [by_name[name] for name in sorted(by_name)]


def _parse_food_list_line(line):
    line = line.strip()
    if not line:
        return None
    parts = line.split("\t", maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else parts[0].strip()


def _load_food_list(food_list_path):
    if food_list_path is None:
        return None
    path = Path(food_list_path)
    if not path.exists():
        raise FileNotFoundError(f"Food list not found: {path}")
    foods = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            food = _parse_food_list_line(line)
            if food:
                foods.append(food)
    return set(dict.fromkeys(foods))


def _filter_records_by_food_list(records, food_names):
    if not food_names:
        return list(records)
    allowed = set(food_names)
    return [row for row in records if row["raw_food_name"] in allowed]


def _build_records(roots, nutrition_table, view="side", food_names=None):
    records = []
    skipped = Counter()

    for root_index, root in enumerate(_normalize_roots(roots)):
        for image_path in _iter_unique_image_paths(root):
            parsed = _parse_aihub_filename(image_path)
            if parsed is None:
                skipped["unparsed_filename"] += 1
                continue
            if parsed["view"] != view:
                skipped[f"view:{parsed['view']}"] += 1
                continue
            if food_names and parsed["raw_food_name"] not in food_names:
                skipped[f"food_list:{parsed['raw_food_name']}"] += 1
                continue
            if parsed["food_name"] not in nutrition_table:
                skipped[f"missing_nutrition:{parsed['food_name']}"] += 1
                continue

            rel_path = str(image_path.relative_to(root))
            records.append(
                {
                    "image_path": str(image_path),
                    "rel_path": f"root{root_index}:{rel_path}",
                    "source_root": str(root),
                    "source_root_index": root_index,
                    **parsed,
                }
            )

    return records, skipped


class FoodBatchSampler(Sampler):
    """Yield batches containing samples from one raw AIHub food folder only."""

    def __init__(self, samples, batch_size, shuffle=True, drop_last=False, seed=42):
        self.samples = samples
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

    def __iter__(self):
        rng = random.Random(self.seed)
        groups = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            groups[sample["raw_food_name"]].append(idx)

        food_names = sorted(groups)
        if self.shuffle:
            rng.shuffle(food_names)

        for food_name in food_names:
            indices = list(groups[food_name])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self):
        total = 0
        groups = defaultdict(int)
        for sample in self.samples:
            groups[sample["raw_food_name"]] += 1
        for count in groups.values():
            if self.drop_last:
                total += count // self.batch_size
            else:
                total += (count + self.batch_size - 1) // self.batch_size
        return total


class FoodBatchSamplerShuffleQ(Sampler):
    """Yield batches for one food at a time, using only the Q-level subset assigned for this epoch.

    mode='train': per food, uses 3 Q levels (all if fewer than 3 available).
    mode='val':   per food, uses the remaining Q levels (2 when 5 are available).
    Q split is deterministic per (food_name, epoch, base_seed) so train and val are always
    complementary without requiring any shared state between the two samplers.
    """

    def __init__(self, samples, batch_size, mode, base_seed=42, epoch=0, drop_last=False):
        if mode not in ('train', 'val'):
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")
        self.samples = samples
        self.batch_size = int(batch_size)
        self.mode = mode
        self.base_seed = int(base_seed)
        self.epoch = int(epoch)
        self.drop_last = bool(drop_last)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _q_split_for_food(self, food_name, available_q_levels):
        rng = random.Random(f"{food_name}_{self.epoch}_{self.base_seed}")
        available = sorted(available_q_levels)
        n_train = min(3, len(available))
        train_q = set(rng.sample(available, n_train))
        return train_q if self.mode == 'train' else set(available) - train_q

    def __iter__(self):
        rng = random.Random(f"order_{self.epoch}_{self.base_seed}")

        groups = defaultdict(list)
        food_q_levels = defaultdict(set)
        for idx, sample in enumerate(self.samples):
            groups[sample["raw_food_name"]].append(idx)
            food_q_levels[sample["raw_food_name"]].add(sample["portion"])

        food_names = sorted(groups)
        if self.mode == 'train':
            rng.shuffle(food_names)

        for food_name in food_names:
            allowed_q = self._q_split_for_food(food_name, food_q_levels[food_name])
            if not allowed_q:
                continue
            indices = [idx for idx in groups[food_name]
                       if self.samples[idx]["portion"] in allowed_q]
            if not indices:
                continue
            if self.mode == 'train':
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self):
        groups = defaultdict(list)
        food_q_levels = defaultdict(set)
        for idx, sample in enumerate(self.samples):
            groups[sample["raw_food_name"]].append(idx)
            food_q_levels[sample["raw_food_name"]].add(sample["portion"])

        total = 0
        for food_name in groups:
            allowed_q = self._q_split_for_food(food_name, food_q_levels[food_name])
            count = sum(1 for idx in groups[food_name]
                        if self.samples[idx]["portion"] in allowed_q)
            if self.drop_last:
                total += count // self.batch_size
            else:
                total += (count + self.batch_size - 1) // self.batch_size
        return total


class AIHubFoodDataset(Dataset):
    """AIHub 122 food amount dataset using raw food-folder holdouts."""

    def __init__(
        self,
        aihub_train_root=DEFAULT_AIHUB_TRAIN_ROOT,
        aihub_validation_root=DEFAULT_AIHUB_VALIDATION_ROOT,
        aihub_test_roots=None,
        nutrition_xlsx=DEFAULT_AIHUB_NUTRITION_XLSX,
        food_list_path=None,
        split="train",
        transform=None,
        view="side",
        enable_ingredient_augmentation=False,
        synonyms_json="ingredient_synonyms_ollama_chatgpt4o.json",
        synonym_replace_prob=0.5,
        ingredient_sample_prob=0.5,
        bbox_cropper=None,
        factorized_q=False,
    ):
        self.bbox_cropper = bbox_cropper
        self.factorized_q = bool(factorized_q)
        self.aihub_train_root = Path(aihub_train_root)
        self.aihub_validation_root = Path(aihub_validation_root)
        self.aihub_test_roots = _normalize_roots(
            aihub_test_roots
            if aihub_test_roots is not None
            else [self.aihub_train_root, self.aihub_validation_root]
        )
        self.nutrition_xlsx = Path(nutrition_xlsx)
        self.food_list_path = Path(food_list_path) if food_list_path else None
        self.split = split
        self.view = view

        self.ingredient_augmentor = IngredientAugmentor(
            enabled=(split == "train" and enable_ingredient_augmentation),
            synonyms_json=synonyms_json,
            synonym_replace_prob=synonym_replace_prob,
            ingredient_sample_prob=ingredient_sample_prob,
        )

        if transform:
            self.transform = transform
        elif split == "train":
            self.transform = self._train_transform()
        else:
            self.transform = self._test_transform()

        self.nutrition_table = _load_nutrition_table(self.nutrition_xlsx)
        food_names = _load_food_list(self.food_list_path)

        if split == "train":
            roots = [self.aihub_train_root]
        elif split == "val":
            roots = [self.aihub_validation_root]
        elif split == "test":
            roots = self.aihub_test_roots
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'")

        self.samples, skipped = _build_records(
            roots=roots,
            nutrition_table=self.nutrition_table,
            view=view,
            food_names=food_names,
        )
        self._print_split_summary(roots, skipped)

        if split == "train" and self.ingredient_augmentor.enabled:
            print(
                "  Ingredient augmentation: enabled "
                f"(synonym_replace_prob={synonym_replace_prob}, ingredient_sample_prob={ingredient_sample_prob})"
            )

    def _print_split_summary(self, roots, skipped):
        print(f"AIHub {self.split} split (raw food list filter, view={self.view}):")
        if self.food_list_path:
            print(f"  Food list: {self.food_list_path}")
        for root in roots:
            print(f"  Root: {root}")
        print(f"  Matched images: {len(self.samples)}")
        foods = sorted({row["raw_food_name"] for row in self.samples})
        print(f"  Matched foods: {len(foods)}")
        self._print_skipped_summary(skipped)

    @staticmethod
    def _print_skipped_summary(skipped):
        if not skipped:
            return
        print("  Skipped samples:")
        for key, count in sorted(skipped.items()):
            print(f"    {key}: {count}")

    def _train_transform(self):
        if self.factorized_q:
            return transforms.Compose([
                ResizeWithPadding(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            ),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _test_transform(self):
        if self.factorized_q:
            return transforms.Compose([
                ResizeWithPadding(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        image_path = Path(row["image_path"])
        try:
            original_image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))
        crop_image = None
        if self.factorized_q:
            image = self.transform(original_image) if self.transform else original_image
            if self.bbox_cropper is not None:
                crop_pil = self.bbox_cropper.crop(original_image.copy())
                crop_image = self.transform(crop_pil) if self.transform else crop_pil
        else:
            image = original_image
            if self.bbox_cropper is not None:
                image = self.bbox_cropper.crop(image)
            if self.transform:
                image = self.transform(image)

        nutrition = self.nutrition_table[row["food_name"]]
        factor = float(row["portion_factor"])
        targets = {
            "calories": torch.tensor([nutrition["calories"] * factor], dtype=torch.float32),
            "mass": torch.tensor([nutrition["mass"] * factor], dtype=torch.float32),
            "fat": torch.tensor([nutrition["fat"] * factor], dtype=torch.float32),
            "carb": torch.tensor([nutrition["carb"] * factor], dtype=torch.float32),
            "protein": torch.tensor([nutrition["protein"] * factor], dtype=torch.float32),
        }

        ingredient_names = self.ingredient_augmentor.augment([row["raw_food_name"]])
        dish_id = f"{row['raw_food_name']}_Q{row['portion']}"

        sample = {
            "images": image,
            "clip_feat": torch.zeros(512, dtype=torch.float32),
            "is_synthetic": torch.tensor(False, dtype=torch.bool),
            "ingredient_names": ingredient_names,
            "targets": targets,
            "dish_id": dish_id,
            "image_path": str(image_path),
            "food_name": row["food_name"],
            "raw_food_name": row["raw_food_name"],
            "portion": row["portion"],
            "source_root": row["source_root"],
        }
        if crop_image is not None:
            sample["crop_image"] = crop_image
        return sample


def collate_fn(batch):
    images_batch = torch.stack([sample["images"] for sample in batch])
    clip_feats_batch = torch.stack([sample["clip_feat"] for sample in batch])
    is_synthetic_batch = torch.stack([sample["is_synthetic"] for sample in batch])
    ingredient_names = [sample["ingredient_names"] for sample in batch]
    dish_ids = [sample["dish_id"] for sample in batch]

    targets = {
        "calories": [],
        "mass": [],
        "fat": [],
        "carb": [],
        "protein": [],
    }
    for sample in batch:
        for key in targets:
            targets[key].append(sample["targets"][key])
    for key in targets:
        targets[key] = torch.stack(targets[key], dim=0)

    return {
        "images": images_batch,
        "clip_feats": clip_feats_batch,
        "is_synthetic": is_synthetic_batch,
        "ingredient_names": ingredient_names,
        "targets": targets,
        "dish_ids": dish_ids,
    }
