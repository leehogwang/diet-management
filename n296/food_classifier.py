"""AIHub 122 음식분류 이미지 → 음식명 분류기 (timm ConvNeXt-Tiny 미세조정).

파이프라인의 "이미지 → 음식명" 단계에서 쓴다 (n296.pipeline_infer). 확신도가 낮으면 Gemini 로 넘긴다.

데이터
  음식분류 이미지 zip 을 푼 폴더: <images_root>/<음식코드 8자리>/<파일>.jpg
  코드 → 음식명은 build_manifest_122.classification_code2name (정렬 순서 = 영양DB 행 순서 [추정]).
  같은 음식을 여러 장 찍은 사진은 파일명 끝 _0, _1 만 다르다. 이것을 한 그룹으로 묶어 train/val/test 로 나눈다
  (같은 접시가 학습과 평가에 동시에 들어가 정확도가 부풀려지는 것을 막음).

확신도 보정
  학습 후 val 로 temperature 하나를 맞춘다 (temperature scaling). 파이프라인의 70% 기준(--threshold)은 보정된 softmax 에 적용된다.

사용 예
  python -m n296.food_classifier prepare --config configs/aihub122.yaml --images-root <이미지 폴더>
  CUDA_VISIBLE_DEVICES=0 python -m n296.food_classifier train --config configs/aihub122.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import re
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

from .common import load_config, resolve, seed_everything

DATA_DIR = "data/food_cls_122"
RUN_DIR = "runs/food_cls_122"
MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ─── 데이터 준비 ─────────────────────────────────────────────────────────────
def _resize_one(job: tuple[str, str, int]) -> str | None:
    src, dst, max_side = job
    try:
        img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        img.thumbnail((max_side, max_side), Image.BICUBIC)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, quality=92)
        return None
    except Exception as e:  # noqa: BLE001 — 깨진 이미지는 건너뜀
        return f"{src}: {type(e).__name__}: {e}"


def prepare(args, cfg):
    from .build_manifest_122 import classification_code2name, load_db

    code2name = classification_code2name(Path(cfg["data"]["label_root_122"]), load_db(Path(cfg["data"]["db_path_122"])))
    rows = []
    for p in Path(args.images_root).rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and re.fullmatch(r"\d{8}", p.parent.name):
            rows.append({"src": str(p), "code": p.parent.name, "stem": p.stem})
    df = pd.DataFrame(rows)
    unknown = sorted(set(df["code"]) - set(code2name))
    if unknown:
        raise SystemExit(f"영양DB 에 없는 코드 {unknown[:5]}")
    df["name"] = df["code"].map(code2name)
    df["group"] = df["stem"].str.replace(r"_\d+$", "", regex=True)

    rng = np.random.default_rng(args.seed)
    split = {}
    for _, g in df.groupby("code"):
        groups = g["group"].unique()
        rng.shuffle(groups)
        n = len(groups)
        n_val, n_test = max(1, round(n * args.val_ratio)), max(1, round(n * args.test_ratio))
        for i, gr in enumerate(groups):
            split[gr] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    df["split"] = df["group"].map(split)

    out = resolve(DATA_DIR)
    df["path"] = [str(out / "img" / c / f"{s}.jpg") for c, s in zip(df["code"], df["stem"])]
    jobs = [(s, d, args.max_side) for s, d in zip(df["src"], df["path"]) if not Path(d).exists()]
    print(f"이미지 {len(df)}장, 음식 {df['code'].nunique()}종, 새로 줄일 것 {len(jobs)}장 → {out}")
    with Pool(args.workers) as pool:
        errors = [e for e in tqdm(pool.imap_unordered(_resize_one, jobs, chunksize=32), total=len(jobs)) if e]
    if errors:
        print(f"실패 {len(errors)}장 (제외): {errors[:3]}")
        bad = {e.split(":")[0] for e in errors}
        df = df[~df["src"].isin(bad)]
    df.drop(columns=["src"]).to_csv(out / "split.csv", index=False)
    print(df.groupby("split").agg(images=("path", "size"), foods=("code", "nunique"), groups=("group", "nunique")))


# ─── 학습 ────────────────────────────────────────────────────────────────────
class FoodDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_idx: dict[str, int], tf):
        self.paths = df["path"].tolist()
        self.labels = [class_idx[c] for c in df["code"]]
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def eval_transform(size: int):
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


def train_transform(size: int):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.2),
    ])


@torch.no_grad()
def predict_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """val NLL 을 최소화하는 temperature (확신도 보정)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def calibration_report(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> dict:
    prob = F.softmax(logits / temperature, dim=1)
    conf, pred = prob.max(1)
    correct = (pred == labels).float()
    top5 = (logits.topk(5, 1).indices == labels[:, None]).any(1).float().mean()
    bins = torch.linspace(0, 1, 16)
    ece = sum(((conf > lo) & (conf <= hi)).float().mean() * (correct[(conf > lo) & (conf <= hi)].mean() - conf[(conf > lo) & (conf <= hi)].mean()).abs()
              for lo, hi in zip(bins[:-1], bins[1:]) if ((conf > lo) & (conf <= hi)).any())
    rep = {"n": len(labels), "top1": float(correct.mean()), "top5": float(top5), "ece": float(ece), "by_threshold": {}}
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        keep = conf >= th
        rep["by_threshold"][str(th)] = {
            "coverage": float(keep.float().mean()),  # 분류기 결과를 그대로 쓰는 비율
            "accuracy_kept": float(correct[keep].mean()) if keep.any() else None,
            "to_gemini": float((~keep).float().mean()),
        }
    return rep


def train(args, cfg):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(resolve(DATA_DIR) / "split.csv", dtype={"code": str})  # 코드 앞자리 0 유지
    classes =df.drop_duplicates("code").sort_values("code")[["code", "name"]].values.tolist()
    class_idx = {c: i for i, (c, _) in enumerate(classes)}
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    print(f"train {len(tr)} / val {len(va)} / test {len(te)} 장, {len(classes)}종")

    mk = lambda d, tf, sh: DataLoader(FoodDataset(d, class_idx, tf), batch_size=args.batch_size, shuffle=sh,
                                      num_workers=args.workers, pin_memory=True, drop_last=sh, persistent_workers=True)
    train_loader = mk(tr, train_transform(args.img_size), True)
    val_loader = mk(va, eval_transform(args.img_size), False)
    test_loader = mk(te, eval_transform(args.img_size), False)

    import timm
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(classes), drop_path_rate=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * len(train_loader)
    warm = len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    out = resolve(RUN_DIR)
    out.mkdir(parents=True, exist_ok=True)
    best, bad, history = -1.0, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"ep{ep}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        logits, labels = predict_logits(model, val_loader, device)
        acc = float((logits.argmax(1) == labels).float().mean())
        history.append({"epoch": ep, "train_loss": tot / n, "val_top1": acc})
        print(f"ep{ep:02d} loss {tot / n:.4f}  val top1 {acc:.4f}", flush=True)
        if acc > best:
            best, bad = acc, 0
            torch.save({"state_dict": model.state_dict(), "model_name": MODEL_NAME, "classes": classes,
                        "img_size": args.img_size, "temperature": 1.0, "epoch": ep}, out / "best.pth")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop (best val top1 {best:.4f})")
                break

    ck = torch.load(out / "best.pth", map_location="cpu")
    model.load_state_dict(ck["state_dict"])
    val_logits, val_labels = predict_logits(model, val_loader, device)
    ck["temperature"] = fit_temperature(val_logits, val_labels)
    torch.save(ck, out / "best.pth")
    test_logits, test_labels = predict_logits(model, test_loader, device)
    report = {
        "best_epoch": ck["epoch"], "temperature": ck["temperature"],
        "val": calibration_report(val_logits, val_labels, ck["temperature"]),
        "test": calibration_report(test_logits, test_labels, ck["temperature"]),
        "test_uncalibrated": calibration_report(test_logits, test_labels, 1.0),
        "history": history,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pred = test_logits.argmax(1)
    per = pd.DataFrame({"code": [classes[i][0] for i in test_labels], "name": [classes[i][1] for i in test_labels],
                        "correct": (pred == test_labels).numpy()})
    per.groupby(["code", "name"])["correct"].agg(["mean", "size"]).rename(columns={"mean": "test_acc", "size": "n"}) \
        .sort_values("test_acc").to_csv(out / "test_per_class.csv", encoding="utf-8-sig")
    print(json.dumps({k: report[k] for k in ("best_epoch", "temperature")} | {"test": report["test"]}, ensure_ascii=False, indent=2))
    print(f"저장: {out}")


# ─── 추론 (파이프라인용) ─────────────────────────────────────────────────────
class FoodClassifier:
    def __init__(self, ckpt_path: str | Path, device: torch.device | str = "cpu"):
        import timm

        ck = torch.load(ckpt_path, map_location="cpu")
        # 초기 체크포인트는 코드를 정수로 저장해 앞자리 0 이 빠졌다 → 8자리로 복원
        self.classes = [(str(c).zfill(8), n) for c, n in ck["classes"]]
        self.temperature = float(ck.get("temperature", 1.0))
        self.device = torch.device(device)
        self.model = timm.create_model(ck["model_name"], pretrained=False, num_classes=len(self.classes))
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()
        self.tf = eval_transform(ck["img_size"])

    @torch.no_grad()
    def predict(self, image: Image.Image, topk: int = 5) -> list[dict]:
        x = self.tf(ImageOps.exif_transpose(image).convert("RGB"))[None].to(self.device)
        prob = F.softmax(self.model(x).float() / self.temperature, dim=1)[0].cpu()
        p, idx = prob.topk(topk)
        return [{"code": self.classes[i][0], "name": self.classes[i][1], "prob": float(v)} for v, i in zip(p, idx)]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--max-side", type=int, default=384)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--patience", type=int, default=6)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--img-size", type=int, default=288)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config(args.config)  # storage_root 설정 (상대 경로 기준)
    prepare(args, cfg) if args.cmd == "prepare" else train(args, cfg)


if __name__ == "__main__":
    main()
