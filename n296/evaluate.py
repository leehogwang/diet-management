"""체크포인트를 manifest(예: test_dataset_296)로 평가.

  python -m n296.evaluate --config configs/stage3_296.yaml \
      --ckpt runs/stage3_296/model_stage3_296_final.pth --manifest data/manifest_test296.csv

원본 model_stage3_296_final.pth 에는 target_stats 가 없어 출력 단위를 모른다.
  --raw-output          : 모델 출력을 그대로 kcal/g 로 간주
  --stats-json <path>   : {"calories": {"mean":..,"std":..}, ...} 로 역정규화
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .common import TARGETS, load_config, resolve
from .dataset import Nutrition296Dataset, load_manifests
from .metrics import compute_metrics
from .train import evaluate
from .model import Stage3NutritionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-mode", choices=["full", "masked"], default="full", help="masked = 텍스트만으로 예측")
    ap.add_argument("--raw-output", action="store_true")
    ap.add_argument("--stats-json", default=None)
    ap.add_argument("--out", default=None, help="샘플별 예측 CSV 저장 경로")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.raw_output:
        stats = {k: {"mean": 0.0, "std": 1.0} for k in TARGETS}
    elif args.stats_json:
        stats = json.loads(open(args.stats_json).read())
    elif "target_stats" in ck:
        stats = ck["target_stats"]
    else:
        raise SystemExit("체크포인트에 target_stats 가 없습니다. --raw-output 또는 --stats-json 을 지정하세요.")

    cfg_eval = dict(cfg)
    cfg_eval["data"] = {**cfg["data"], "enable_text_only": False}
    df = load_manifests(cfg_eval, resolve(args.manifest), None)
    ds = Nutrition296Dataset(df, cfg, resolve(cfg["paths"]["emb_cache"]), stats, train=False,
                             image_cache_dir=resolve(cfg["paths"].get("image_cache_dir")))
    ds.load_images = args.image_mode != "masked"
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
                        num_workers=cfg["training"].get("image_workers", 0))

    model = Stage3NutritionModel(cfg, pretrained=False).to(device)
    model.load_state_dict(ck.get("state_dict", ck))
    metrics, pred, gt = evaluate(model, loader, device, stats, args.image_mode)
    print("전체:", json.dumps({k: round(v, 2) for k, v in metrics.items()}, ensure_ascii=False))

    res = df[["food_code", "food_name", "image_path"]].reset_index(drop=True).copy()
    for k in TARGETS:
        res[f"pred_{k}"], res[f"gt_{k}"] = pred[k], gt[k]
    rows = []
    for name, g in res.groupby("food_name"):
        m = compute_metrics({k: g[f"pred_{k}"].values for k in TARGETS}, {k: g[f"gt_{k}"].values for k in TARGETS})
        rows.append({"food_name": name, "n": len(g), "gt_kcal": g["gt_calories"].iloc[0],
                     "pred_kcal_mean": g["pred_calories"].mean(), **{k: round(v, 1) for k, v in m.items()}})
    print(pd.DataFrame(rows).sort_values("mean_pmae").to_string(index=False))
    if "q_level" in df.columns:  # AIHub 122: 양 등급별 성능
        res["q_level"] = df["q_level"].values
        q_rows = []
        for q, g in res.groupby("q_level"):
            m = compute_metrics({k: g[f"pred_{k}"].values for k in TARGETS}, {k: g[f"gt_{k}"].values for k in TARGETS})
            q_rows.append({"q_level": q, "n": len(g), **{k: round(v, 1) for k, v in m.items()}})
        print("\n양 등급별:")
        print(pd.DataFrame(q_rows).to_string(index=False))
    if args.out:
        resolve(args.out).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(resolve(args.out), index=False)
        print(f"저장: {resolve(args.out)}")


if __name__ == "__main__":
    main()
