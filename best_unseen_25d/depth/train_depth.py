"""깊이 브랜치 + 적응형 게이트 모델의 Stage 1~3 학습 (기존 train_curriculum_4stages.py 흐름을 따르되 별도 파일).

기본 설정 = 후속 실험에서 검증된 구성: 텍스트 64D, 중량 피드백 없음, 분량 보조 손실 0.5, 텍스트 드롭아웃 0.3.
Stage 0(kFood 분류 백본)은 기존 체크포인트를 그대로 사용한다.

usage:
  python depth/train_depth.py --stage 1 --out_dir OUT --backbone_path .../checkpoint_stage0_backbone.pth [--no_depth]
  python depth/train_depth.py --stage 2 --out_dir OUT --resume OUT/checkpoint_stage1_omnifood.pth [--no_depth]
  python depth/train_depth.py --stage 3 --fold k --out_dir OUT/foldk --resume OUT/checkpoint_stage2_122_std.pth [--no_depth]
"""
import argparse
import functools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_baseline_the_best"))

import dataset_omnifood8k as omni
import dataset_aihub122_classification_nutrition as std
import dataset_aihub122_multimodal as q122
from train_curriculum_4stages import MultiRehearsalDataset, unified_nutrition_collate_fn, predict_abs_5, pmae
from depth.depth_data import DepthStore, WithDepth, depth_collate
from depth.model_depth import DepthGatedNutritionModel

print = functools.partial(print, flush=True)
NUTS = ["calories", "fat", "protein", "carb"]


def build_model(cfg, use_depth, dev, text_in_dim=3072):
    mc = cfg["model"]
    return DepthGatedNutritionModel(
        use_depth=use_depth, image_dim=mc["image_dim"], text_in_dim=text_in_dim, text_hidden1=mc["text_hidden1"],
        text_hidden2=mc["text_hidden2"], head_hidden=mc["head_hidden"], dropout=mc["dropout"],
        freeze_backbone_layers=mc["freeze_backbone_layers"], modality_dropout_prob=0.0, freq_cfg=cfg.get("freq"),
        mass_feedback=False).to(dev)


def forward(model, b, dev, train):
    return model(b["images"].to(dev), b["has_image"].to(dev), b["text_emb"].to(dev), apply_dropout=train,
                 depth=b["depth"].to(dev), depth_meta=b["depth_meta"].to(dev), has_depth=b["has_depth"].to(dev))


@torch.no_grad()
def evaluate(model, loader, dev, save_preds=None):
    model.eval()
    errs, gates, P, T = [], [], [], []
    gsum = [torch.zeros(5), torch.zeros(5)]
    for b in loader:
        p = predict_abs_5(forward(model, b, dev, False))
        t = b["targets"].to(dev)
        pr = torch.stack([p[k] for k in ["calories", "fat", "protein", "carb", "mass"]], 1)
        errs.append(pmae(pr, t).cpu())
        gsum[0] += (pr - t).abs().sum(0).cpu(); gsum[1] += t.sum(0).cpu()  # G = Σ|p−t| / Σt
        gates.append(model.last_gate.cpu())
        if save_preds is not None:
            P.append(pr.cpu()); T.append(t.cpu())
    e = torch.cat(errs).mean(0) * 100
    g = torch.cat(gates)
    if save_preds is not None:
        np.savez(save_preds, pred=torch.cat(P).numpy(), target=torch.cat(T).numpy(), gate=g.numpy())
    res = {"mean4": float(e[:4].mean()), **{k: float(e[i]) for i, k in enumerate(NUTS + ["mass"])}}
    res["gate"] = [round(float(x), 3) for x in g.mean(0)]
    gv = gsum[0] / gsum[1].clamp(min=1e-6) * 100
    res["G"] = {k: float(gv[i]) for i, k in enumerate(NUTS + ["mass"])}
    res["G_mean4"] = float(gv[:4].mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--backbone_path", default=None)
    ap.add_argument("--no_depth", action="store_true", help="같은 구조에서 깊이 브랜치만 뺀 비교 모델")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--folds_file", default=str(ROOT / "foodlist_folds_122.json"))
    ap.add_argument("--portion_aux", type=float, default=0.5)
    ap.add_argument("--objective", choices=["A", "G"], default="A", help="A: 기존(로그 손실, 검증 A로 선택) / G: 로그 손실 + 절대 L1 항, 검증 G로 선택")
    ap.add_argument("--g_lam", type=float, default=1.0, help="--objective G 의 절대 L1 항 가중치")
    ap.add_argument("--no_mass_loss", action="store_true", help="중량(mass)을 학습 타깃에서 제외 (중량 어블레이션, 기본은 포함)")
    ap.add_argument("--emb_cache", default=None, help="텍스트 임베딩 캐시 교체 (예: CLIP). 기본은 각 데이터셋의 Gemini 캐시")
    ap.add_argument("--text_in_dim", type=int, default=3072, help="텍스트 임베딩 차원 (Gemini 3072, CLIP ViT-L/14 768 등)")
    ap.add_argument("--text_aug", action="store_true", help="Stage 3 학습 샘플에 재료명 변형 텍스트 증강(50%) 적용 (Stage 2와 같은 variants)")
    ap.add_argument("--config", default=str(ROOT / "text_baseline_the_best/config_multimodal_296_text64_ingredients.yaml"))
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(a.config).read_text())
    use_depth = not a.no_depth
    print(f"[depth-train] objective={a.objective} stage={a.stage} use_depth={use_depth} fold={a.fold} portion_aux={a.portion_aux} no_mass_loss={a.no_mass_loss} emb_cache={a.emb_cache} text_in_dim={a.text_in_dim} device={dev}")

    model = build_model(cfg, use_depth, dev, a.text_in_dim)
    if a.stage == 1 and not a.resume and a.backbone_path:
        ck = torch.load(a.backbone_path, map_location=dev, weights_only=False)
        model.image_encoder.load_state_dict(ck["backbone_state_dict"], strict=False)
        print(f"[transfer] Stage 0 backbone <- {a.backbone_path}")
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
        print(f"[transfer] <- {a.resume} (missing={len(missing)}, unexpected={len(unexpected)})")

    feat = {}
    if a.portion_aux > 0:  # 분량(Q1~Q5) 분류 보조 헤드
        model.portion_head = nn.Linear(cfg["model"]["head_hidden"], 5).to(dev)
        model.nutrition_head.dropout.register_forward_hook(lambda m, i, o: feat.__setitem__("h", o))
        if a.resume and "portion_head.weight" in ck["state_dict"]:
            model.portion_head.load_state_dict({k[len("portion_head."):]: v for k, v in ck["state_dict"].items() if k.startswith("portion_head.")})

    store = DepthStore()
    E_omni, E_std, E_q = omni._TRANSFORMS["val"], std._TRANSFORMS["val"], q122._IMAGE_TRANSFORM["eval"]
    def W(ds, tf, train, fv=False):
        if hasattr(ds, "default_emb") and ds.default_emb.numel() != a.text_in_dim:
            ds.default_emb = torch.zeros(a.text_in_dim)  # 임베딩 차원이 3072가 아닐 때(CLIP 등) 결측 대체값 차원 맞춤
        return WithDepth(ds, store, tf, train, flip_v=fv)
    EC = {"emb_cache_path": Path(a.emb_cache)} if a.emb_cache else {}
    so, eo = omni.load_omnifood_dataset(**EC)
    omni_ds = lambda recs, train: W(omni.OmniFoodDataset(recs, eo, train=train), E_omni, train, fv=True)
    test_loader, save = None, None
    if a.stage == 1:
        train_ds, val_ds, test_ds = omni_ds(so["train"], True), omni_ds(so["val"], False), omni_ds(so["test"], False)
        main_recs = so["train"]
        save = out / "checkpoint_stage1_omnifood.pth"
    else:
        s2, e2, v2 = std.load_122_classification_dataset(**EC)
        std_ds = lambda recs, train: W(std.AIHub122ClassificationDataset(recs, e2, v2, train=train), E_std, train)
        if a.stage == 2:
            train_ds = MultiRehearsalDataset(std_ds(s2["train"], True), [omni_ds(so["train"][:500], True)], [0.2])
            val_ds, test_ds = std_ds(s2["val"], False), None
            main_recs = s2["train"]
            save = out / "checkpoint_stage2_122_std.pth"
        else:
            vf = json.loads(Path(a.folds_file).read_text(encoding="utf-8"))["folds"][a.fold] if a.fold is not None else None
            s3, e3 = q122.load_aihub122_dataset(emb_cache_path=Path(a.emb_cache) if a.emb_cache else ROOT / "text_baseline_the_best/emb_cache_with_ingredients.json", val_foods=vf)
            tv = v2 if a.text_aug else None  # food_to_ingredient_texts.json: {음식명: {base, variants}}
            if tv:
                print(f"[text_aug] Stage 3 variants on: {sum(1 for r in s3['train'] if r['food_name'] in tv)}/{len(s3['train'])} train samples covered")
            q_ds = lambda recs, train: W(q122.AIHub122MultimodalDataset(recs, e3, train=train, text_variants=tv if train else None), E_q, train)
            train_ds = MultiRehearsalDataset(q_ds(s3["train"], True), [std_ds(s2["train"][:500], True), omni_ds(so["train"][:500], True)], [0.1, 0.1])
            val_ds, test_ds = q_ds(s3["val"], False), q_ds(s3["test"], False)
            main_recs = s3["train"]
            save = out / "model_stage3_final.pth"

    coll = functools.partial(depth_collate, base_collate=unified_nutrition_collate_fn)
    mk = lambda ds, sh: DataLoader(ds, batch_size=a.batch_size, shuffle=sh, collate_fn=coll, num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    train_loader, val_loader = mk(train_ds, True), mk(val_ds, False)
    if test_ds is not None:
        test_loader = mk(test_ds, False)

    lr = a.lr or {1: 3e-4, 2: 1e-4, 3: 5e-5}[a.stage]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    crit = nn.SmoothL1Loss()
    keys = NUTS + ["mass"]
    mean_t = torch.tensor(np.stack([np.asarray(r["targets"], dtype=np.float32) for r in main_recs]).mean(0), device=dev)
    used = [i for i, k in enumerate(keys) if not (a.no_mass_loss and k == "mass")]
    if a.objective == "G":
        print(f"[objective G] lam={a.g_lam} mean_t={[round(float(x), 2) for x in mean_t]} used={[keys[i] for i in used]}")
    best, bad = float("inf"), 0
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for b in train_loader:
            t = torch.log1p(b["targets"].to(dev).clamp(min=0))
            p = forward(model, b, dev, True)
            loss = sum(crit(p[k].squeeze(-1), t[:, i]) for i, k in enumerate(keys) if not (a.no_mass_loss and k == "mass"))
            if a.objective == "G":  # 큰 음식의 절대 오차에 비례하는 항 (G 지표 = Σ|p−t|/Σt 와 정렬)
                pa = predict_abs_5(p); pa = torch.stack([pa[k] for k in keys], 1)
                raw = b["targets"].to(dev).clamp(min=0)
                loss = loss + a.g_lam * ((pa - raw).abs() / mean_t)[:, used].mean()
            if a.portion_aux > 0:
                q = b["portion"].to(dev)
                if (q >= 0).any():
                    loss = loss + a.portion_aux * nn.functional.cross_entropy(model.portion_head(feat["h"]), q, ignore_index=-1)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(t)
        v = evaluate(model, val_loader, dev)
        score = v["G_mean4"] if a.objective == "G" else v["mean4"]
        sched.step(score)
        print(f"==> [Stage {a.stage}] Epoch {ep:3d} ({time.time() - t0:.1f}s) | Train Loss: {tot / len(train_ds):.4f} | "
              f"Val Mean PMAE: {v['mean4']:.2f}% (G {v['G_mean4']:.2f}%) | cal {v['calories']:.1f} fat {v['fat']:.1f} prot {v['protein']:.1f} carb {v['carb']:.1f} mass {v['mass']:.1f} | gate {v['gate']}")
        if score < best:
            best, bad = score, 0
            torch.save({"epoch": ep, "state_dict": model.state_dict(), "val_pmae": best, "use_depth": use_depth, "val_gate": v["gate"],
                        "no_mass_loss": a.no_mass_loss, "emb_cache": a.emb_cache, "text_in_dim": a.text_in_dim,
                        "objective": a.objective, "g_lam": a.g_lam}, save)
            print(f"  ★ [Saved Best] {save} (Val {best:.2f}%)")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"[EarlyStop] Epoch {ep}. Best Val PMAE: {best:.2f}%")
                break

    if test_loader is not None:
        ck = torch.load(save, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        r = evaluate(model, test_loader, dev, save_preds=out / "test_preds.npz")
        print(f"[TEST] mean4 {r['mean4']:.2f}% (G {r['G_mean4']:.2f}%) | cal {r['calories']:.2f} fat {r['fat']:.2f} prot {r['protein']:.2f} carb {r['carb']:.2f} mass {r['mass']:.2f} | gate {r['gate']}")
        (out / "test_result.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
