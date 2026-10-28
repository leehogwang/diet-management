"""Stronger backbone (ConvNeXt) + higher resolution (384) on the gram-multitask winner.

Reuses the exact gram-multitask setup (RGB-D input + ingredient-gram aux head); the
only changes are the backbone and input resolution.
  --backbone resnet50  : 2015 baseline (2048-d feat)
  --backbone convnext  : torchvision ConvNeXt-Tiny, 2022 (768-d feat), 4ch stem adapted
  --resolution 224|384 : input image size

Control reference = grams_multitask (resnet50@224) = 27.15% test mean PMA.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import (ResNet50_Weights, resnet50,
                                ConvNeXt_Tiny_Weights, convnext_tiny,
                                ConvNeXt_Small_Weights, convnext_small,
                                ConvNeXt_Base_Weights, convnext_base)

ROOT = Path("/home/students/cs/202520906/nutrition5k_pipeline")
sys.path.insert(0, str(ROOT))
from nutrition5k_cbam.utils import save_json, seed_everything  # noqa: E402
from scripts.task_batch.train_depth import OV, NUTR, pma  # noqa: E402
from scripts.task_batch.train_grams import parse_grams, usable, GramsDS, VOCAB_K  # noqa: E402
from scripts.task_batch import train_grams_crop as TGC  # noqa: E402
from scripts.task_batch.train_grams_crop import CropDS  # noqa: E402  (plate-ROI cropped dataset)


def build_backbone(name, pretrained):
    """Return (module, feat_dim) that maps (B,4,H,W) -> (B,feat_dim)."""
    if name == "resnet50":
        bb = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        old = bb.conv1.weight.data
        new = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            new.weight[:, :3] = old
            new.weight[:, 3:] = old.mean(1, keepdim=True)
        bb.conv1 = new
        bb.fc = nn.Identity()
        return bb, 2048
    if name in ("convnext", "convnext_small", "convnext_base"):
        # 2026-08-22: convnext_base 추가(용량 축). 아래 stem 수술은 세 변종에 그대로 통한다
        # — ConvNeXt 계열은 모두 features[0][0] 이 Conv2d(3, C, k=4, s=4) 이고 폭 C 만 다르다.
        if name == "convnext_small":
            bb = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT if pretrained else None)
        elif name == "convnext_base":
            bb = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
        else:
            bb = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        stem = bb.features[0][0]  # Conv2d(3,96,k=4,s=4)
        new = nn.Conv2d(4, stem.out_channels, kernel_size=4, stride=4)
        with torch.no_grad():
            new.weight[:, :3] = stem.weight.data
            new.weight[:, 3:] = stem.weight.data.mean(1, keepdim=True)
            new.bias.copy_(stem.bias.data)
        bb.features[0][0] = new
        in_feat = bb.classifier[2].in_features  # 768
        bb.classifier[2] = nn.Identity()  # keep LayerNorm2d+Flatten, drop final Linear
        return bb, in_feat
    raise ValueError(name)


class StrongNet(nn.Module):
    def __init__(self, backbone, use_aux, vocab_k=VOCAB_K, pretrained=True, dropout=0.2):
        super().__init__()
        self.use_aux = use_aux
        self.backbone, fd = build_backbone(backbone, pretrained)
        self.nutr_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(fd, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(512, 4), nn.Softplus(),
        )
        if use_aux:
            self.aux_head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(fd, 512), nn.ReLU(inplace=True),
                nn.Linear(512, vocab_k),
            )

    def forward(self, x):
        f = self.backbone(x)
        nutr = self.nutr_head(f)
        aux = self.aux_head(f) if self.use_aux else None
        return nutr, aux


def _nrmse(P, T):
    """정규화 RMSE(%) = 100 * RMSE_i / mean(정답_i) 를 4영양소 평균 (TASK5 조기종료 기준).
    PMA(절대오차 기반)와 달리 큰 오차를 제곱으로 벌해 대용량 접시 실수에 민감하다."""
    err = P - T
    rmse = np.sqrt((err ** 2).mean(0))
    denom = np.maximum(T.mean(0), 1e-8)
    out = {f"{n}_RMSE": float(rmse[i]) for i, n in enumerate(NUTR)}
    out["mean_nRMSE"] = float(np.mean(100.0 * rmse / denom))
    return out


def build_side_rehearsal(mode, frac, res, seed, grams, vocab, gmu, gsd, n_overhead):
    """단계1 리허설용 side 데이터셋을 만든다 (2026-09-13).

    ## 왜
    STEP12 체인은 단계0(side 사전학습)에서 overhead 가 없어 못 쓰던 접시 1,060장을 새로
    확보하지만, **단계1·2 는 다시 overhead 2,485접시만 본다.** 단계0 에서 배운 것을 두 단계에
    걸쳐 잊는다(catastrophic forgetting). 여기서는 단계1 학습셋에 side 프레임을 섞어
    그 접시들을 최종 학습까지 끌고 온다.

    ## 모드
      new : overhead 가 없는 접시만 섞는다(신규 정보만 추가). 1,060접시 ≈ 배치의 30%.
      all : train split 의 side 보유 접시 전부(뷰 증강 효과까지). 3,253접시 ≈ 배치의 57%.

    ## 비율
      frac=0  : 위 풀을 그대로 사용(모드가 비율을 정한다).
      frac>0  : side 표본이 전체의 frac 이 되도록 풀을 잘라내거나(부족하면) 반복한다.

    ## 누수 차단
    train split 접시만 쓴다. val/test 접시의 side 프레임은 한 장도 넣지 않는다(아래 assert).
    """
    import json as _json
    import random as _random
    # 지연 임포트 — train_side_pretrain 이 이 모듈의 StrongNet 을 임포트하므로 순환을 피한다.
    from scripts.task_batch.train_side_pretrain import IDX, SideDS  # noqa: E402
    from scripts.task_batch.task2_common import dish_targets5  # noqa: E402

    index = _json.loads(IDX.read_text())
    tgt5 = {}
    for sp in ["train", "val", "test"]:
        tgt5.update(dish_targets5(sp))
    pool = sorted(d for d, m in index["dishes"].items()
                  if m["split"] == "train" and d in tgt5
                  and (mode == "all" or not m["has_overhead"]))
    assert pool, f"리허설 풀이 비었다 (mode={mode})"
    # ★누수 차단 — 이 assert 가 깨지면 학습을 시작하지 않는다.
    assert all(index["dishes"][d]["split"] == "train" for d in pool), "train 외 접시가 섞였다"

    n_pool = len(pool)
    if frac > 0:
        assert 0 < frac < 1, "--rehearse-frac 는 0<frac<1"
        want = int(round(frac * n_overhead / (1.0 - frac)))
        rng = _random.Random(seed)
        if want <= n_pool:
            shuffled = list(pool); rng.shuffle(shuffled)
            pool = sorted(shuffled[:want])
        else:  # 풀보다 많이 필요하면 반복 사용(오버샘플)
            reps, rem = divmod(want, n_pool)
            extra = list(pool); rng.shuffle(extra)
            pool = sorted(pool * reps) + sorted(extra[:rem])
    tgt = {d: tgt5[d][:4] for d in set(pool)}
    ds = SideDS(pool, tgt, grams, vocab, gmu, gsd, index, res=res, train=True)
    n_uniq = len(set(pool))
    n_new = sum(1 for d in set(pool) if not index["dishes"][d]["has_overhead"])
    actual = len(pool) / float(len(pool) + n_overhead)
    print(f"리허설 mode={mode} frac={frac} | side 표본 {len(pool)} "
          f"(고유 접시 {n_uniq}, 그중 overhead 없던 신규 {n_new}) "
          f"| overhead {n_overhead} → 배치 내 side 비율 {actual*100:.1f}%", flush=True)
    return ds, {"rehearse": mode, "rehearse_frac": frac, "n_side_samples": len(pool),
                "n_side_dishes": n_uniq, "n_side_new_dishes": n_new,
                "side_share": round(actual, 4), "n_overhead_dishes": n_overhead}


def evaluate(model, loader, device, log_target=False):
    model.eval(); P, T = [], []
    with torch.no_grad():
        for x, y, gy, _ in loader:
            with torch.autocast("cuda"):
                nutr, _ = model(x.to(device))
            pred = nutr.float().cpu().numpy()
            if log_target:  # 모델이 log1p 공간 예측 → 실단위 복원(상한 클램프)
                pred = np.expm1(np.clip(pred, 0.0, 10.0))
            P.append(pred); T.append(y.numpy())
    P = np.concatenate(P); T = np.concatenate(T)
    return {**pma(P, T), **_nrmse(P, T)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone",
                    choices=["resnet50", "convnext", "convnext_small", "convnext_base"],
                    required=True)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crop", type=int, choices=[0, 1], default=0,
                    help="1 = train/eval on plate-ROI cropped images (background removed)")
    ap.add_argument("--log-target", type=int, default=0,
                    help="1 = 영양 타깃에 log1p 회귀(대용량 음식 편중 완화), 평가시 expm1 복원")
    ap.add_argument("--tgt-reweight", type=float, default=0.0,
                    help=">0 = 칼로리 크기^power 표본별 손실가중(대용량 음식 상향, mean=1 정규화)")
    ap.add_argument("--monitor", choices=["pma", "rmse"], default="pma",
                    help="early stopping 기준. rmse = 정규화 RMSE(큰 오차에 민감) — TASK5")
    ap.add_argument("--init-from", default="", help="파인튜닝 시작 체크포인트 경로")
    # 2026-08-21: 접시 크롭 박스 모드. mixed = 검출 성공하면 검출박스(현행, 88%만 고정박스),
    # fixed = 전 접시 동일 고정박스. --crop 1 일 때만 의미가 있다(원본 프레임 학습은 크롭 자체가 없음).
    ap.add_argument("--crop-box", choices=["mixed", "fixed"], default="mixed")
    # 2026-09-13: 단계1 리허설. none 이면 아래 경로를 만들지도 않으므로 기존 동작과 100% 동일하다.
    ap.add_argument("--rehearse", choices=["none", "new", "all"], default="none",
                    help="단계1 학습셋에 side 프레임을 섞는다. new=overhead 없던 접시만, all=전부")
    ap.add_argument("--rehearse-frac", type=float, default=0.0,
                    help="0 = 모드가 정한 풀 그대로. >0 = side 표본이 전체의 이 비율이 되도록 조정")
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda")
    # DataLoader worker 를 fork 하기 전에 세팅해야 워커에도 전달된다.
    if args.crop_box == "fixed":
        assert args.crop == 1, "--crop-box fixed 는 --crop 1 에서만 의미가 있다"
        TGC.FIXED_BOX = TGC.FALLBACK_BOX
        n_diff = sum(1 for d, b in TGC.BBOX.items() if b and tuple(b) != TGC.FALLBACK_BOX)
        print(f"crop_box=fixed -> 전 접시 {TGC.FALLBACK_BOX} 사용 "
              f"(현행 대비 바뀌는 접시 {n_diff}/{len(TGC.BBOX)})", flush=True)
    print(f"strong run={args.run_name} backbone={args.backbone} res={args.resolution} "
          f"lam={args.lam} crop={args.crop} crop_box={args.crop_box}", flush=True)

    grams = parse_grams()
    tr_d, tr_t = usable("train"); va_d, va_t = usable("val"); te_d, te_t = usable("test")
    df = Counter()
    for d in tr_d:
        for n in grams.get(d, {}):
            df[n] += 1
    vocab = [n for n, _ in df.most_common(VOCAB_K)]
    G = np.array([[np.log1p(grams.get(d, {}).get(n, 0.0)) for n in vocab] for d in tr_d], dtype=np.float64)
    gmu, gsd = G.mean(0), G.std(0) + 1e-6
    print(f"dishes train {len(tr_d)} val {len(va_d)} test {len(te_d)} | vocab {len(vocab)}", flush=True)

    DS = CropDS if args.crop else GramsDS
    mk = lambda dd, tt, tr: DS(dd, tt, grams, vocab, gmu, gsd, res=args.resolution, train=tr)
    train_ds = mk(tr_d, tr_t, True)
    reh_info = {"rehearse": "none"}
    if args.rehearse != "none":
        # vocab/gmu/gsd 는 overhead 학습셋에서 만든 것을 그대로 쓴다(보조 과제 정의가 바뀌면 안 된다).
        side_ds, reh_info = build_side_rehearsal(
            args.rehearse, args.rehearse_frac, args.resolution, args.seed,
            grams, vocab, gmu, gsd, len(tr_d))
        train_ds = torch.utils.data.ConcatDataset([train_ds, side_ds])
    # 검증·평가는 건드리지 않는다 — overhead RGB-D 그대로라 기존 arm 과 직접 비교된다.
    tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va = DataLoader(mk(va_d, va_t, False), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    te = DataLoader(mk(te_d, te_t, False), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    model = StrongNet(args.backbone, use_aux=True, pretrained=True).to(device)
    if args.init_from:  # TASK5: 챔피언 체크포인트에서 이어서 학습
        model.load_state_dict(torch.load(args.init_from, map_location=device,
                                         weights_only=False)["model_state"])
        print(f"init from {args.init_from}", flush=True)
    crit = nn.SmoothL1Loss()
    crit_none = nn.SmoothL1Loss(reduction="none")  # 표본별 재가중용
    cal_med = 1.0
    if args.tgt_reweight > 0:
        cal_arr = np.array([tr_t[d][0] for d in tr_d], dtype=np.float64)
        cal_med = float(np.median(cal_arr))
        print(f"tgt-reweight power={args.tgt_reweight} cal_median={cal_med:.1f}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    run_dir = ROOT / "outputs/task_runs/task4_final/runs" / args.run_name
    ckp = run_dir / "checkpoints"; ckp.mkdir(parents=True, exist_ok=True)
    mkey = "mean_PMA" if args.monitor == "pma" else "mean_nRMSE"
    best = float("inf"); bad = 0; hist = []; stopped_early = False
    for ep in range(1, args.epochs + 1):
        model.train(); ln = 0.0; la = 0.0
        for x, y, gy, _ in tr:
            x, y, gy = x.to(device), y.to(device), gy.to(device)
            with torch.autocast("cuda"):
                nutr, aux = model(x)
                y_t = torch.log1p(y) if args.log_target else y  # 로그타깃
                per = crit_none(nutr, y_t).mean(dim=1)           # (B,) 표본별
                if args.tgt_reweight > 0:  # 대용량(고칼로리) 표본 상향가중
                    w = (y[:, 0] / cal_med).clamp(min=1e-3) ** args.tgt_reweight
                    lo_n = (per * (w / w.mean()).detach()).mean()
                else:
                    lo_n = per.mean()
                lo_a = crit(aux, gy)
                loss = lo_n + args.lam * lo_a
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ln += lo_n.item() * x.size(0); la += lo_a.item() * x.size(0)
        vm = evaluate(model, va, device, log_target=bool(args.log_target))
        hist.append({"epoch": ep, "nutr_loss": ln/len(tr.dataset), "aux_loss": la/len(tr.dataset), "val": vm})
        print(f"ep {ep:3d} | nutr {ln/len(tr.dataset):.2f} aux {la/len(tr.dataset):.3f} | "
              f"val PMA {vm['mean_PMA']:.2f}% nRMSE {vm['mean_nRMSE']:.2f}% "
              f"(cal {vm['calories_PMA']:.1f} calRMSE {vm['calories_RMSE']:.1f})", flush=True)
        if vm[mkey] < best:
            best = vm[mkey]; bad = 0
            torch.save({"model_state": model.state_dict(), "epoch": ep,
                        "monitor": mkey, "monitor_value": best, "val": vm}, ckp / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ {ep}", flush=True); stopped_early = True; break

    state = torch.load(ckp / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    test_m = evaluate(model, te, device, log_target=bool(args.log_target))
    val_m = state.get("val", {})
    save_json(run_dir / "training_history.json", hist)
    save_json(run_dir / "test_metrics.json", {**test_m, "best_val_mean_PMA": val_m.get("mean_PMA", best),
              "monitor": mkey, "monitor_value": best, "val": val_m, "init_from": args.init_from,
              "best_epoch": state["epoch"], "backbone": args.backbone, "resolution": args.resolution,
              "cropped": args.crop, "crop_box": args.crop_box,
              "log_target": args.log_target, "tgt_reweight": args.tgt_reweight,
              # ★2026-08-21 추가: 상한에 잘린 런을 놓치지 않기 위한 기록.
              #   (cs384_crop_seed2 가 상한 80 에 걸렸던 것이 이 필드가 없어 드러나지 않았다)
              "epochs_run": len(hist), "epoch_cap": args.epochs,
              "stopped_early": stopped_early, "hit_epoch_cap": (not stopped_early),
              "lr": args.lr, "patience": args.patience, "seed": args.seed,
              "n_test_dishes": len(te_d), **reh_info})
    flag = "" if stopped_early else "  [!] 학습 미완료(에폭 상한 도달)"
    print(f"DONE {args.run_name} | monitor={mkey} best_val {best:.2f}% "
          f"ep{state['epoch']}/{len(hist)} | "
          f"TEST PMA {test_m['mean_PMA']:.2f}% (cal {test_m['calories_PMA']:.2f}) "
          f"nRMSE {test_m['mean_nRMSE']:.2f}% calRMSE {test_m['calories_RMSE']:.1f}{flag}", flush=True)


if __name__ == "__main__":
    main()
