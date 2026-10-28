"""AIHub 122 양추정 이미지 → 음식 양 등급 Q1~Q5 분류기 (timm ConvNeXt-Tiny 미세조정).

파이프라인(n296.pipeline_infer)에서 음식 분류기와 같은 사진으로 양 등급을 예측하고,
영양 모델이 낸 1인분(Q4) 기준 영양값에 등급 비율을 곱해 사진 속 양으로 보정한다.
  Q1 0.25 · Q2 0.5 · Q3 0.75 · Q4 1.0 · Q5 1.25  (공식 활용가이드, build_manifest_122.Q_RATIO)

데이터
  data/manifest_122.csv 의 task == quantity 행 (build_manifest_122 가 라벨 zip 에서 만든 것, 78종).
  이미지는 양추정 이미지 zip 을 풀지 않고 zip 에서 바로 읽어 --max-side 로 줄인 사본만 저장한다
  (zip 48GB 를 풀면 같은 만큼 디스크를 더 쓴다). 이미 푼 폴더를 줘도 된다.
분할
  공식 Training → train/val (split_group 단위 9:1, 같은 접시의 side/top 은 같은 쪽), 공식 Validation → test.
  Validation 음식은 모두 Training 에도 있다 (사진은 다름).
전처리
  양을 판단하려면 접시 전체와 크기 기준 물체(숟가락·포크·동전)가 보여야 하므로 가운데 자르기 없이
  정사각형으로 리사이즈한다. 학습 증강도 크기를 바꾸지 않는 것(좌우 반전, 작은 회전·이동, 색)만 쓴다.
확신도 보정
  food_classifier 와 같이 학습 후 val 로 temperature 를 맞춘다.

사용 예
  python -m n296.quantity_estimator prepare --config configs/aihub122.yaml \
      --images-root <양추정_이미지_TRAIN_0422_add.zip> <양추정_이미지_VAL_0422_add.zip>
  CUDA_VISIBLE_DEVICES=0 python -m n296.quantity_estimator train --config configs/aihub122.yaml
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import struct
import sys
import tarfile
import threading
import zipfile
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .build_manifest_122 import IMAGE_EXTS, Q_RATIO, _zip_name, index_images
from .common import load_config, resolve, seed_everything
from .food_category import CATEGORY_NAMES, FOOD_CATEGORY, UNKNOWN, normalize_category
from .food_classifier import MEAN, STD, fit_temperature

DATA_DIR = "data/quantity_122"
RUN_DIR = "runs/quantity_122_cat"
MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k_384"  # 384px 로 미세조정된 가중치 (입력 크기와 맞춤)
Q_LEVELS = list(Q_RATIO)  # ["Q1", ..., "Q5"]
RATIOS = torch.tensor([Q_RATIO[q] for q in Q_LEVELS])
UNKNOWN_IDX = CATEGORY_NAMES.index(UNKNOWN)


# ─── 데이터 준비 ─────────────────────────────────────────────────────────────
_ZIPS: dict[str, zipfile.ZipFile] = {}
_INNER: dict[tuple[str, str], zipfile.ZipFile] = {}  # 마지막으로 연 안쪽 zip 하나만 (메모리 절약)


def _read_member(src: str, member: str | tuple[str, str]) -> bytes:
    """member 가 (안쪽 zip 멤버, 이미지 멤버) 이면 zip 속 zip 에서 읽는다."""
    if src not in _ZIPS:  # 워커마다 zip 을 한 번만 연다
        _ZIPS[src] = zipfile.ZipFile(src)
    if isinstance(member, str):
        return _ZIPS[src].read(member)
    inner, name = member
    if (src, inner) not in _INNER:
        _INNER.clear()
        _INNER[(src, inner)] = zipfile.ZipFile(io.BytesIO(_ZIPS[src].read(inner)))
    return _INNER[(src, inner)].read(name)


def _resize_one(job: tuple[str, str | tuple[str, str] | None, str, int]) -> tuple[str, str] | None:
    """job = (원본 파일 또는 zip, zip 안 멤버 이름 / (안쪽 zip, 멤버) / None, 저장 경로, 최대 변 길이)."""
    src, member, dst, max_side = job
    try:
        if member is None:
            img = Image.open(src)
        else:
            img = Image.open(io.BytesIO(_read_member(src, member)))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_side, max_side), Image.BICUBIC)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, quality=92)
        return None
    except Exception as e:  # noqa: BLE001 — 깨진 이미지는 건너뜀
        return dst, f"{src}:{member}: {type(e).__name__}: {e}"


def index_sources(roots: list[str]) -> dict[str, tuple[str, str | None]]:
    """파일명(소문자) → (zip 경로, 멤버) 또는 (파일 경로, None).
    AIHub 양추정 이미지 zip 은 image/<음식>/<Q>/<Q>.zip 처럼 zip 안에 zip 이 있어, 그 경우 멤버는 (안쪽 zip, 이미지)."""
    idx = {}
    for r in roots:
        p = Path(r)
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                for info in tqdm(z.infolist(), desc=f"index {p.name}", leave=False):
                    name = os.path.basename(_zip_name(info))
                    ext = os.path.splitext(name)[1].lower()
                    if ext in IMAGE_EXTS:
                        idx[name.lower()] = (str(p), info.filename)
                    elif ext == ".zip":
                        with zipfile.ZipFile(z.open(info)) as iz:
                            for ii in iz.infolist():
                                n = os.path.basename(_zip_name(ii))
                                if os.path.splitext(n)[1].lower() in IMAGE_EXTS:
                                    idx[n.lower()] = (str(p), (info.filename, ii.filename))
        elif p.is_dir():
            idx.update({k: (v, None) for k, v in index_images(p).items()})
        else:
            raise SystemExit(f"이미지 경로가 없습니다: {p}")
    return idx


def prepare(args, cfg):
    m = pd.read_csv(resolve(args.manifest))
    m = m[m["task"] == "quantity"].copy()
    src = index_sources(args.images_root)
    m["src"] = m["file_name"].str.lower().map(src)
    missing = m["src"].isna()
    print(f"양추정 라벨 {len(m)}장 중 이미지 찾음 {(~missing).sum()}장, 없음 {missing.sum()}장")
    m = m[~missing]
    if m.empty:
        raise SystemExit("찾은 이미지가 없습니다. --images-root 를 확인하세요.")

    rng = np.random.default_rng(args.seed)
    split = {}
    tr = m[m["source_split"] == "Training"]
    for _, g in tr.groupby(["food_name", "q_level"]):
        groups = g["split_group"].unique()
        rng.shuffle(groups)
        n_val = max(1, round(len(groups) * args.val_ratio))
        split.update({gr: "val" if i < n_val else "train" for i, gr in enumerate(groups)})
    m["split"] = np.where(m["source_split"] == "Training", m["split_group"].map(split), "test")

    out = resolve(DATA_DIR)
    m["path"] = [str(out / "img" / f"{Path(f).stem}.jpg") for f in m["file_name"]]
    jobs = [(s[0], s[1], d, args.max_side) for s, d in zip(m["src"], m["path"]) if not Path(d).exists()]
    jobs.sort(key=lambda j: (j[0], str(j[1])))  # 같은 안쪽 zip 의 이미지가 한 워커에 몰리도록
    print(f"새로 줄일 것 {len(jobs)}장 → {out}")
    with Pool(args.workers) as pool:
        errors = [e for e in tqdm(pool.imap_unordered(_resize_one, jobs, chunksize=16), total=len(jobs)) if e]
    if errors:
        print(f"실패 {len(errors)}장 (제외): {[msg for _, msg in errors[:3]]}")
        m = m[~m["path"].isin({dst for dst, _ in errors})]
    cols = ["file_name", "food_name", "q_level", "q_ratio", "view", "split_group", "source_split", "split", "path",
            "mass", "calories", "fat", "carb", "protein"]
    m[cols].to_csv(out / "split.csv", index=False)
    print(m.groupby(["split", "q_level"]).size().unstack().to_string())


# ─── 다운로드 스트림에서 바로 줄인 사본 만들기 (zip 을 디스크에 두지 않음) ─────────
class _PartStream(io.RawIOBase):
    """AIHub 다운로드 tar 스트림(<zip>.partOFFSET 조각이 순서대로 옴)을 이어 붙인 zip 바이트 스트림."""

    def __init__(self, fileobj):
        self.tf = tarfile.open(fileobj=fileobj, mode="r|")
        self.cur, self.pos = None, 0

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self.cur is None:
                m = self.tf.next()
                if m is None:
                    return 0
                if not m.isfile():
                    continue
                mo = re.search(r"\.part(\d+)$", m.name)
                if (int(mo.group(1)) if mo else 0) != self.pos:
                    raise IOError(f"조각 순서가 어긋남: {m.name} (기대 offset {self.pos})")
                self.cur = self.tf.extractfile(m)
            n = self.cur.readinto(b)
            if n:
                self.pos += n
                return n
            self.cur = None


def _read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise IOError(f"스트림이 중간에 끊김 ({len(b)}/{n} 바이트)")
    return b


def _iter_zip_stream(f):
    """순차 zip 스트림의 local header 를 따라 (이름, 압축 푼 바이트) 를 낸다. 중앙 디렉터리에서 끝."""
    while f.read(4) == b"PK\x03\x04":
        _, flag, method, _, _, _, csize, usize, nlen, xlen = struct.unpack("<HHHHHIIIHH", _read_exact(f, 26))
        raw, extra = _read_exact(f, nlen), _read_exact(f, xlen)
        name = raw.decode("utf-8" if flag & 0x800 else "cp949", errors="replace")
        if flag & 0x08:
            raise IOError(f"data descriptor 를 쓰는 zip 은 지원하지 않음: {name}")
        i = 0
        while i + 4 <= len(extra):  # ZIP64 확장 필드
            hid, hl = struct.unpack("<HH", extra[i:i + 4])
            if hid == 1:
                vals = struct.unpack(f"<{hl // 8}Q", extra[i + 4:i + 4 + hl // 8 * 8])
                j = 0
                if usize == 0xFFFFFFFF:
                    usize, j = vals[j], j + 1
                if csize == 0xFFFFFFFF:
                    csize = vals[j]
            i += 4 + hl
        data = _read_exact(f, csize)
        if method == 8:
            data = zlib.decompress(data, -15)
        elif method != 0:
            raise IOError(f"지원하지 않는 압축 방식 {method}: {name}")
        yield name, data


_WANTED: dict[str, str] = {}


def _stream_init(wanted: dict[str, str]):
    global _WANTED
    _WANTED = wanted


def _resize_bytes(data: bytes, dst: str, max_side: int) -> None:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img.thumbnail((max_side, max_side), Image.BICUBIC)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=92)


def _stream_job(job: tuple[str, bytes, int]) -> tuple[int, list[str]]:
    """job = (항목 이름, 바이트, 최대 변). 안쪽 zip 이면 그 안의 필요한 이미지를 모두 줄여 저장."""
    name, data, max_side = job
    saved, errors = 0, []
    try:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as iz:
                items = [(os.path.basename(_zip_name(ii)).lower(), ii) for ii in iz.infolist()]
                for key, ii in items:
                    if key in _WANTED and not Path(_WANTED[key]).exists():
                        try:
                            _resize_bytes(iz.read(ii), _WANTED[key], max_side)
                            saved += 1
                        except Exception as e:  # noqa: BLE001
                            errors.append(f"{name}:{ii.filename}: {type(e).__name__}: {e}")
        else:
            key = os.path.basename(name).lower()
            if key in _WANTED and not Path(_WANTED[key]).exists():
                _resize_bytes(data, _WANTED[key], max_side)
                saved += 1
    except Exception as e:  # noqa: BLE001
        errors.append(f"{name}: {type(e).__name__}: {e}")
    return saved, errors


def stream(args, cfg):
    """curl ... | python -m n296.quantity_estimator stream : AIHub 다운로드 스트림(tar)에서 양추정 이미지를 바로
    --max-side 로 줄여 data/quantity_122/img/ 에 저장한다. 그다음 prepare --images-root data/quantity_122/img ..."""
    m = pd.read_csv(resolve(args.manifest))
    m = m[m["task"] == "quantity"]
    out = resolve(DATA_DIR) / "img"
    wanted = {f.lower(): str(out / f"{Path(f).stem}.jpg") for f in m["file_name"]}
    src = io.BufferedReader(_PartStream(sys.stdin.buffer), 8 << 20)
    saved, errors, entries = 0, [], 0
    slots = threading.BoundedSemaphore(args.workers * 2)  # 메모리에 올려 둘 안쪽 zip 수 제한

    def done(res):
        nonlocal saved
        saved += res[0]
        errors.extend(res[1])
        slots.release()

    with Pool(args.workers, initializer=_stream_init, initargs=(wanted,)) as pool:
        bar = tqdm(_iter_zip_stream(src), desc="stream", unit="entry")
        for name, data in bar:
            entries += 1
            if not data:
                continue  # 폴더
            slots.acquire()
            pool.apply_async(_stream_job, ((name, data, args.max_side),), callback=done)
            bar.set_postfix(saved=saved, GB=f"{src.raw.pos / 1e9:.1f}")
        pool.close()
        pool.join()
    print(f"zip 항목 {entries}개, 새로 저장 {saved}장 → {out}, 실패 {len(errors)}장 {errors[:3]}")
    if errors:
        sys.exit(1)


# ─── 모델 ────────────────────────────────────────────────────────────────────
class QuantityNet(nn.Module):
    """이미지 특징(ConvNeXt) + 음식 대분류 임베딩 → Q1~Q5.
    n_categories == 0 이면 대분류 없이 이미지만 쓴다 (비교 실험용)."""

    def __init__(self, model_name: str, n_classes: int, n_categories: int, cat_dim: int = 64,
                 pretrained: bool = False, drop_path_rate: float = 0.0):
        super().__init__()
        import timm

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate)
        d = self.backbone.num_features
        self.cat_emb = nn.Embedding(n_categories, cat_dim) if n_categories else None
        d += cat_dim if n_categories else 0
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_classes))

    def forward(self, x, cat=None):
        f = self.backbone(x)
        if self.cat_emb is not None:
            f = torch.cat([f, self.cat_emb(cat)], dim=1)
        return self.head(f)


# ─── 학습 ────────────────────────────────────────────────────────────────────
class QuantityDataset(Dataset):
    """(이미지, 대분류 번호, Q 번호). cat_dropout 확률로 대분류를 기타로 바꿔, 대분류를 모를 때도 동작하게 한다."""

    def __init__(self, df: pd.DataFrame, tf, cat_dropout: float = 0.0):
        self.paths = df["path"].tolist()
        self.cats = [CATEGORY_NAMES.index(c) for c in df["category"]]
        self.labels = [Q_LEVELS.index(q) for q in df["q_level"]]
        self.tf, self.cat_dropout = tf, cat_dropout

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        c = self.cats[i]
        if self.cat_dropout and torch.rand(1).item() < self.cat_dropout:
            c = UNKNOWN_IDX
        return self.tf(Image.open(self.paths[i]).convert("RGB")), c, self.labels[i]


def eval_transform(size: int):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def train_transform(size: int):
    return transforms.Compose([
        transforms.Resize((size, size)), transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


@torch.no_grad()
def predict_logits(model, loader, device, mask_category: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """mask_category: 대분류를 모두 기타로 바꿔 예측 (대분류 효과 확인용)."""
    model.eval()
    out, ys = [], []
    for x, c, y in loader:
        if mask_category:
            c = torch.full_like(c, UNKNOWN_IDX)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(x.to(device, non_blocking=True), c.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(out), torch.cat(ys)


def quantity_report(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> dict:
    prob = F.softmax(logits / temperature, dim=1)
    pred = prob.argmax(1)
    expected = prob @ RATIOS
    true_ratio = RATIOS[labels]
    conf = torch.zeros(len(Q_LEVELS), len(Q_LEVELS), dtype=torch.long)
    for t, p in zip(labels, pred):
        conf[t, p] += 1
    return {
        "n": len(labels),
        "acc": float((pred == labels).float().mean()),
        "acc_within1": float(((pred - labels).abs() <= 1).float().mean()),  # 한 등급 차이까지 허용
        "ratio_mae_argmax": float((RATIOS[pred] - true_ratio).abs().mean()),
        "ratio_mae_expected": float((expected - true_ratio).abs().mean()),
        "confusion": {"rows=true, cols=pred": Q_LEVELS, "matrix": conf.tolist()},
    }


def per_food(df: pd.DataFrame, logits: torch.Tensor, labels: torch.Tensor) -> pd.DataFrame:
    pred = logits.argmax(1)
    per = pd.DataFrame({"food_name": df["food_name"].values, "category": df["category"].values,
                        "correct": (pred == labels).numpy(), "within1": ((pred - labels).abs() <= 1).numpy()})
    return per.groupby(["food_name", "category"]).agg(acc=("correct", "mean"), within1=("within1", "mean"),
                                                      n=("correct", "size")).sort_values("acc").reset_index()


def pick_holdout(df: pd.DataFrame, seed: int) -> list[str]:
    """음식이 3종 이상인 대분류마다 1종씩 뽑아 학습에서 뺀다 (처음 보는 음식 일반화 실험)."""
    rng = np.random.default_rng(seed)
    foods = df.drop_duplicates("food_name")
    out = []
    for c, g in foods.groupby("category"):
        if c != UNKNOWN and len(g) >= 3:
            out.append(str(rng.choice(sorted(g["food_name"]))))
    return sorted(out)


def train(args, cfg):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(resolve(DATA_DIR) / "split.csv")
    df["category"] = df["food_name"].map(FOOD_CATEGORY)
    if df["category"].isna().any():
        raise SystemExit(f"대분류가 없는 음식: {sorted(df[df['category'].isna()]['food_name'].unique())} (food_category.py)")
    holdout = pick_holdout(df, args.seed) if args.holdout else []
    unseen = df[df["food_name"].isin(holdout)]
    df = df[~df["food_name"].isin(holdout)]
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    print(f"train {len(tr)} / val {len(va)} / test {len(te)} 장, {df['food_name'].nunique()}종, "
          f"대분류 {'사용' if args.use_category else '안 씀'}")
    if holdout:
        print(f"학습에서 뺀 음식 {len(holdout)}종 ({len(unseen)}장): {', '.join(holdout)}")

    mk = lambda d, tf, sh, cd=0.0: DataLoader(QuantityDataset(d, tf, cd), batch_size=args.batch_size, shuffle=sh,
                                              num_workers=args.workers, pin_memory=True, drop_last=sh,
                                              persistent_workers=True)
    train_loader = mk(tr, train_transform(args.img_size), True, args.cat_dropout if args.use_category else 0.0)
    val_loader = mk(va, eval_transform(args.img_size), False)

    n_cat = len(CATEGORY_NAMES) if args.use_category else 0
    model = QuantityNet(MODEL_NAME, len(Q_LEVELS), n_cat, pretrained=True, drop_path_rate=0.1).to(device)
    # 새로 붙인 헤드·임베딩은 백본보다 학습률을 크게
    new = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": args.lr},
                             {"params": new, "lr": args.lr * 10}], weight_decay=0.05)
    steps = args.epochs * len(train_loader)
    warm = len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    meta = {"model_name": MODEL_NAME, "q_levels": Q_LEVELS, "q_ratio": Q_RATIO, "img_size": args.img_size,
            "categories": CATEGORY_NAMES if args.use_category else [], "cat_dim": 64,
            "foods": sorted(tr["food_name"].unique()), "holdout_foods": holdout}

    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best, bad, history = -1.0, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for step, (x, c, y) in enumerate(tqdm(train_loader, desc=f"ep{ep}", leave=False)):
            if args.max_steps and step >= args.max_steps:
                break
            x, c, y = x.to(device, non_blocking=True), c.to(device), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = crit(model(x, c), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        logits, labels = predict_logits(model, val_loader, device)
        acc = float((logits.argmax(1) == labels).float().mean())
        history.append({"epoch": ep, "train_loss": tot / n, "val_acc": acc})
        print(f"ep{ep:02d} loss {tot / n:.4f}  val acc {acc:.4f}", flush=True)
        if acc > best:
            best, bad = acc, 0
            torch.save({**meta, "state_dict": model.state_dict(), "temperature": 1.0, "epoch": ep}, out / "best.pth")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop (best val acc {best:.4f})")
                break

    ck = torch.load(out / "best.pth", map_location="cpu")
    model.load_state_dict(ck["state_dict"])
    val_logits, val_labels = predict_logits(model, val_loader, device)
    ck["temperature"] = fit_temperature(val_logits, val_labels)
    torch.save(ck, out / "best.pth")
    T = ck["temperature"]
    report = {"best_epoch": ck["epoch"], "temperature": T, "use_category": args.use_category,
              "val": quantity_report(val_logits, val_labels, T), "history": history}
    for name, d in (("test", te), ("unseen", unseen)):
        if d.empty:
            continue
        loader = mk(d, eval_transform(args.img_size), False)
        logits, labels = predict_logits(model, loader, device)
        report[name] = quantity_report(logits, labels, T)
        pf = per_food(d, logits, labels)
        if args.use_category:  # 같은 사진에서 대분류를 가렸을 때 (대분류가 얼마나 도움이 되는지)
            m_logits, _ = predict_logits(model, loader, device, mask_category=True)
            report[f"{name}_category_masked"] = quantity_report(m_logits, labels, T)
            pf = pf.merge(per_food(d, m_logits, labels)[["food_name", "acc", "within1"]]
                          .rename(columns={"acc": "acc_masked", "within1": "within1_masked"}), on="food_name")
        pf.to_csv(out / f"{name}_per_food.csv", index=False, encoding="utf-8-sig")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "confusion"} if isinstance(v, dict) else v
                      for k, v in report.items() if k != "history"}, ensure_ascii=False, indent=2))
    print(f"저장: {out}")


# ─── 추론 (파이프라인용) ─────────────────────────────────────────────────────
class QuantityEstimator:
    def __init__(self, ckpt_path: str | Path, device: torch.device | str = "cpu"):
        ck = torch.load(ckpt_path, map_location="cpu")
        self.q_levels = ck["q_levels"]
        self.ratios = torch.tensor([ck["q_ratio"][q] for q in self.q_levels])
        self.temperature = float(ck.get("temperature", 1.0))
        self.foods = set(ck.get("foods", []))  # 학습한 음식
        self.categories = ck.get("categories", [])  # 비어 있으면 대분류를 쓰지 않는 모델
        self.device = torch.device(device)
        self.model = QuantityNet(ck["model_name"], len(self.q_levels), len(self.categories), ck.get("cat_dim", 64))
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()
        self.tf = eval_transform(ck["img_size"])

    @torch.no_grad()
    def predict(self, image: Image.Image, category: str | None = None) -> dict:
        """category: 음식 대분류 (food_category.CATEGORIES). 목록에 없거나 None 이면 기타."""
        x = self.tf(ImageOps.exif_transpose(image).convert("RGB"))[None].to(self.device)
        cat = normalize_category(category)
        c = torch.tensor([self.categories.index(cat) if self.categories else 0], device=self.device)
        prob = F.softmax(self.model(x, c).float() / self.temperature, dim=1)[0].cpu()
        i = int(prob.argmax())
        return {"q_level": self.q_levels[i], "q_prob": float(prob[i]), "q_ratio": float(self.ratios[i]),
                "q_ratio_expected": float(prob @ self.ratios), "q_category": cat,
                "q_probs": {q: round(float(p), 4) for q, p in zip(self.q_levels, prob)}}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", required=True)
    p.add_argument("--images-root", nargs="+", required=True, help="양추정 이미지 zip 또는 zip 을 푼 폴더 (여러 개 가능)")
    p.add_argument("--manifest", default="data/manifest_122.csv")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("stream", help="AIHub 다운로드 tar 스트림(stdin)에서 바로 줄인 사본 저장")
    s.add_argument("--config", required=True)
    s.add_argument("--manifest", default="data/manifest_122.csv")
    s.add_argument("--max-side", type=int, default=512)
    s.add_argument("--workers", type=int, default=16)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--out-dir", default=RUN_DIR)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--patience", type=int, default=6)
    t.add_argument("--batch-size", type=int, default=48)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--img-size", type=int, default=384)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--no-category", dest="use_category", action="store_false", help="대분류 없이 이미지만 (비교용)")
    t.add_argument("--cat-dropout", type=float, default=0.2, help="학습 때 대분류를 기타로 바꾸는 확률")
    t.add_argument("--holdout", action="store_true",
                   help="대분류마다 음식 1종을 학습에서 빼고 그 음식으로 처음 보는 음식 성능(unseen)을 잰다")
    t.add_argument("--max-steps", type=int, default=0, help="스모크 테스트용: epoch 당 최대 step (0 = 제한 없음)")
    t.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config(args.config)  # storage_root 설정 (상대 경로 기준)
    {"prepare": prepare, "stream": stream, "train": train}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
