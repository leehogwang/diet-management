#!/usr/bin/env python3
"""raw RGBD 5-fold 앙상블 추론 — 497(cutout 부분집합) + 공식 depth_test 507 양쪽 평가.

최종 벤치마크 마감용. 산출: outputs/raw_rgbd_kfold_ensemble_results.json
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from datasets.nutrition5k_rgb_dataset import load_gt_dict, load_split_ids, TARGETS
from datasets.nutrition5k_cutout_dataset import load_cutout_split_ids
from datasets.nutrition5k_raw_rgbd_dataset import Nutrition5kRawRGBDDataset
from datasets.nutrition5k_pred_rgbd_dataset import Nutrition5kPredRGBDDataset
import argparse
_args = argparse.ArgumentParser(); _args.add_argument("--pred", action="store_true")
ARGS = _args.parse_args()
DS = Nutrition5kPredRGBDDataset if ARGS.pred else Nutrition5kRawRGBDDataset
RUN_NAME = "pred_rgbd_kfold" if ARGS.pred else "raw_rgbd_kfold"
OUT_NAME = ("pred" if ARGS.pred else "raw") + "_rgbd_kfold_ensemble_results.json"
from scripts.train_cutout_rgbd import ResNet101SE_RGBD

N_FOLDS = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, loader):
    P, G = [], []
    for x, y in loader:
        P.append(model(x.to(device)).clamp(min=0).cpu().numpy()); G.append(y.numpy())
    return np.concatenate(P), np.concatenate(G)


def metrics(P, G):
    out, paper = {}, []
    for j, t in enumerate(TARGETS):
        p, g = P[:, j], G[:, j]
        mae = float(np.abs(p - g).mean())
        ppm = mae / float(g.mean()) * 100
        out[t] = {"mae": round(mae, 3), "paper_pmae": round(ppm, 2)}
        paper.append(ppm)
    out["mean_paper_pmae"] = round(float(np.mean(paper)), 2)
    return out


def eval_testset(name, ids, gt, models_):
    ds = DS(ids, gt, augment=False)
    loader = DataLoader(ds, batch_size=32, num_workers=4)
    preds, G = [], None
    for m in models_:
        P, G = predict(m, loader)
        preds.append(P)
    ens = np.mean(preds, axis=0)
    singles = [metrics(P, G)["mean_paper_pmae"] for P in preds]
    m = metrics(ens, G)
    print(f"\n=== {name} (n={len(G)}) ===")
    for t in TARGETS:
        print(f"  {t:<10} paperPMAE={m[t]['paper_pmae']:5.1f}%")
    print(f"  {'MEAN':<10} {m['mean_paper_pmae']:5.2f}%   (단일 fold 평균 {np.mean(singles):.2f}±{np.std(singles):.2f})")
    return {"n": int(len(G)), "ensemble": m,
            "single_fold_mean_paper_pmae": [round(s, 2) for s in singles]}


def main():
    gt = load_gt_dict()
    models_ = []
    for k in range(1, N_FOLDS + 1):
        mdl = ResNet101SE_RGBD(len(TARGETS)).to(device)
        ck = torch.load(ROOT / "runs" / RUN_NAME / f"fold{k}" / "best.pt", map_location=device)
        mdl.load_state_dict(ck["model_state"]); mdl.eval()
        models_.append(mdl)
        print(f"fold{k}: best_ep={ck['epoch']} val={ck['val_loss']:.4f}", flush=True)

    _, test_497 = load_cutout_split_ids()            # cutout 부분집합 (기존 비교선)
    test_507 = load_split_ids("test")                # 공식 depth_test (이미지 있는 507)

    out = {
        "test_497_cutout_subset": eval_testset("test 497 (cutout 부분집합)", test_497, gt, models_),
        "test_507_official": eval_testset("공식 depth_test 507", test_507, gt, models_),
    }
    print("\n참고: cutout RGBD 앙상블 19.24(497) / raw 단일 19.85(497)·20.46(507) / DPF 17.8 / Zhao 17.06")
    (ROOT / "outputs" / OUT_NAME).write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print(f"저장: outputs/{OUT_NAME}")


if __name__ == "__main__":
    main()
