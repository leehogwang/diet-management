"""복원한 모델이 model_stage3_296_final.pth 를 strict 로 로드할 수 있는지 확인.

  python -m n296.check_compat --config configs/stage3_296.yaml \
      --ckpt /home/students/cs/202321024/datasets/model_stage3_296_final.pth
"""
from __future__ import annotations

import argparse

import torch

from .common import load_config
from .model import Stage3NutritionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = Stage3NutritionModel(cfg, pretrained=False)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    mismatched = [k for k, v in model.state_dict().items() if k in sd and sd[k].shape != v.shape]
    print(f"missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(mismatched)}")
    for k in missing[:20]:
        print("  missing   ", k)
    for k in unexpected[:20]:
        print("  unexpected", k)
    for k in mismatched[:20]:
        print("  mismatch  ", k, tuple(sd[k].shape), "vs", tuple(model.state_dict()[k].shape))
    if not (missing or unexpected or mismatched):
        print("OK: 모든 파라미터 이름/shape 일치 (strict 로드 가능)")
    for k in ("stage", "epoch", "best_val_pmae", "val_metrics"):
        if k in ckpt:
            print(f"{k}: {ckpt[k]}")


if __name__ == "__main__":
    main()
