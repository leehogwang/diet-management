"""공통 설정/유틸."""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 상대 경로의 기준. load_config 가 config 의 paths.storage_root 로 바꾼다.
# /home 은 mergerfs(minfreespace=250G)라 하위 디스크 여유가 250G 미만이면 새 파일 생성이 거부된다.
# storage_root 를 하위 디스크(/mnt/nvme2/...)의 같은 프로젝트 경로로 두면 제한 없이 쓰고,
# mergerfs 로 합쳐진 /home 경로에서도 같은 파일이 보인다.
STORAGE_ROOT = PROJECT_ROOT

# 모델 헤드 순서와 체크포인트 key 이름을 그대로 따른다.
TARGETS = ["calories", "mass", "fat", "carb", "protein"]
# 체크포인트의 mean_pmae 는 cal/fat/prot 세 개의 평균과 일치한다(22.74).
MEAN_PMAE_KEYS = ["cal", "fat", "prot"]


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(Path(path).resolve())
    global STORAGE_ROOT
    root = cfg.get("paths", {}).get("storage_root")
    STORAGE_ROOT = Path(root) if root else PROJECT_ROOT
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    return cfg


def resolve(p: str | Path | None) -> Path | None:
    """상대 경로는 STORAGE_ROOT(config paths.storage_root, 없으면 프로젝트 루트) 기준으로 해석한다."""
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else STORAGE_ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atwater_carb(calories, fat, protein):
    """칼로리/지방/단백질에서 탄수화물 역산: (kcal - 9·fat - 4·protein) / 4, 음수는 0."""
    if isinstance(calories, torch.Tensor):
        return torch.clamp((calories - 9 * fat - 4 * protein) / 4, min=0)
    return np.clip((np.asarray(calories) - 9 * np.asarray(fat) - 4 * np.asarray(protein)) / 4, 0, None)


def public_config(cfg: dict) -> dict:
    """체크포인트에 저장할 config (내부 키 제거)."""
    out = copy.deepcopy(cfg)
    out.pop("_config_path", None)
    return out
