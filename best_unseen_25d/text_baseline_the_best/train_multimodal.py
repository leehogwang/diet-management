"""
멀티모달(이미지+음식명) 절대량 영양 회귀 — text_baseline_the_best 확장, AIHub 296 데이터셋.

296 이미지(ConvNeXt-base+GeM, train_aihub122_recipe_rag.py에서 FiLM만 제거한 구성)와
텍스트 전용 영양DB를 modality dropout(learned mask token)으로 하나의 모델에 합친다.
타깃은 100g당 값이 아니라 1인분 절대량.

296 라벨에는 탄수화물이 없다. 칼로리(calories)/지방(fat)/단백질(protein)은 항상 직접
지도학습하고, 탄수화물은 Atwater 항등식(carb = (cal − 9·fat − 4·prot) / 4)으로 역산한다
— 기존 122 파이프라인의 cal_math(칼로리를 역산)와 정반대 방향. NutritionHead(model.py)는
5개 헤드(calories/mass/fat/carb/protein)를 항상 출력하는 공유 클래스라 구조 변경은 불필요하고,
head_carb/head_mass는 그래디언트가 흐르지 않는 미사용 출력으로 남는다.

사용법: python text_baseline_the_best/train_multimodal.py --config text_baseline_the_best/config_multimodal_296.yaml
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import (ConvNeXtImageEncoder, GeM, NutritionHead, FrequencyEncoder, MixerBlock,
                    CONVNEXT_BASE_TAP_CHANNELS, CONVNEXT_BASE_TAP_SPATIAL)  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import dataset_multimodal as dm  # noqa: E402

DIRECT_KEYS = ["calories", "fat", "protein"]  # NutritionHead 출력 키 중 직접 지도학습하는 것들
                                               # (순서는 dm.TARGET_COLS=[cal,fat,prot]과 맞춰야 함)


class MultimodalNutritionModel(nn.Module):
    def __init__(self, image_dim: int, text_in_dim: int, text_hidden1: int, text_hidden2: int,
                 head_hidden: int, dropout: float, freeze_backbone_layers: bool,
                 modality_dropout_prob: float, freq_cfg: dict | None = None,
                 mass_feedback: bool = True):
        super().__init__()
        self.image_encoder = ConvNeXtImageEncoder(pretrained=True)
        if freeze_backbone_layers:
            for p in self.image_encoder.features[:3].parameters():
                p.requires_grad = False
            print("[model] frozen ConvNeXt features[0:3] (stem + stage0)")
        self.pooling = GeM()
        self.mask_token = nn.Parameter(torch.zeros(image_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.text_encoder = nn.Sequential(
            nn.Linear(text_in_dim, text_hidden1), nn.BatchNorm1d(text_hidden1),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(text_hidden1, text_hidden2), nn.BatchNorm1d(text_hidden2),
            nn.ReLU(), nn.Dropout(dropout),
        )

        # --- 주파수(DFT) 브랜치: freq_cfg가 없거나 enabled=False면 기존 파이프라인과 100% 동일 ---
        freq_cfg = freq_cfg or {}
        self.freq_enabled = freq_cfg.get("enabled", False)
        mix_dim = 0
        if self.freq_enabled:
            if freq_cfg.get("mode", "spatial") != "spatial":
                raise NotImplementedError(
                    f"freq.mode={freq_cfg.get('mode')!r} 미구현 — 현재는 'spatial'"
                    "(DFT 밴드 마스킹 -> iFFT -> VGG16)만 지원.")
            cnx_tap = freq_cfg.get("cnx_tap", 5)
            assert CONVNEXT_BASE_TAP_SPATIAL[cnx_tap] == 14, (
                f"cnx_tap={cnx_tap}의 공간 크기는 {CONVNEXT_BASE_TAP_SPATIAL[cnx_tap]}인데, "
                "주파수 브랜치(VGG16 features[:24])는 14x14 고정입니다. "
                "리사이즈 로직 없이는 cnx_tap=5만 지원합니다.")
            self.cnx_tap = cnx_tap
            cnx_tap_channels = CONVNEXT_BASE_TAP_CHANNELS[cnx_tap]
            vgg_out_dim = freq_cfg.get("vgg_out_dim", 512)

            self.freq_encoder = FrequencyEncoder(
                r_low=freq_cfg.get("r_low", 0.15), r_high=freq_cfg.get("r_high", 0.85),
                window_alpha=freq_cfg.get("window_alpha", 0.25), out_dim=vgg_out_dim,
                pretrained=True, freeze_vgg_layers=freq_cfg.get("freeze_vgg_layers", True),
            )

            mixer1_dim = vgg_out_dim + cnx_tap_channels
            self.mixer1 = MixerBlock(
                num_tokens=14 * 14, dim=mixer1_dim,
                token_hidden=freq_cfg.get("mixer1_token_hidden", 256),
                channel_hidden=freq_cfg.get("mixer1_channel_hidden", 2048),
                dropout=dropout,
            )

            conv_dims = freq_cfg.get("conv_dims", [512, 256])
            conv_layers = []
            in_ch = mixer1_dim
            branch_a_size = 14
            for out_ch in conv_dims:
                conv_layers += [
                    nn.MaxPool2d(2),
                    nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
                ]
                in_ch = out_ch
                branch_a_size //= 2
            self.conv_blocks = nn.Sequential(*conv_layers)

            mix_dim = freq_cfg.get("mixer2_dim", 256)
            assert conv_dims[-1] == mix_dim, (
                f"conv_dims[-1]={conv_dims[-1]}는 mixer2_dim={mix_dim}과 같아야 합니다 "
                "(토큰 축으로 concat하므로 채널 차원이 일치해야 함).")

            cnx_final_channels = CONVNEXT_BASE_TAP_CHANNELS[-1]  # ConvNeXt 최종 출력 채널(1024, 고정)
            cnx_final_tokens = CONVNEXT_BASE_TAP_SPATIAL[-1] ** 2  # 7x7=49, 고정
            self.cnx_proj = nn.Linear(cnx_final_channels, mix_dim)
            self.mixer2 = MixerBlock(
                num_tokens=branch_a_size * branch_a_size + cnx_final_tokens, dim=mix_dim,
                token_hidden=freq_cfg.get("mixer2_token_hidden", 64),
                channel_hidden=freq_cfg.get("mixer2_channel_hidden", 512),
                dropout=dropout,
            )
            self.mix_mask_token = nn.Parameter(torch.zeros(mix_dim))
            nn.init.normal_(self.mix_mask_token, std=0.02)

        self.nutrition_head = NutritionHead(
            image_dim + mix_dim + text_hidden2,
            head_hidden,
            dropout=dropout,
            mass_feedback=mass_feedback
        )
        self.modality_dropout_prob = modality_dropout_prob

    def forward(self, images, has_image, text_emb, apply_dropout: bool):
        B = images.shape[0]
        device = images.device
        keep = has_image.clone()
        if apply_dropout and self.modality_dropout_prob > 0:
            drop_roll = torch.rand(B, device=device) < self.modality_dropout_prob
            keep = keep & ~drop_roll

        img_feat = self.mask_token.unsqueeze(0).expand(B, -1).clone()
        mix_feat = self.mix_mask_token.unsqueeze(0).expand(B, -1).clone() if self.freq_enabled else None

        if keep.any():
            kept_images = images[keep]
            n_keep = kept_images.shape[0]
            if self.freq_enabled:
                out_tap, out_final = self.image_encoder(kept_images, tap=self.cnx_tap)
            else:
                out_final = self.image_encoder(kept_images)
            img_feat[keep] = self.pooling(out_final).view(n_keep, -1)

            if self.freq_enabled:
                freq_feat = self.freq_encoder(kept_images)               # [n, vgg_out_dim, 14, 14]
                fused_map = torch.cat([freq_feat, out_tap], dim=1)       # [n, mixer1_dim, 14, 14]
                n, c, h, w = fused_map.shape
                tokens = self.mixer1(fused_map.flatten(2).transpose(1, 2))  # [n, 196, mixer1_dim]
                fused_map = tokens.transpose(1, 2).reshape(n, c, h, w)

                branch_a = self.conv_blocks(fused_map).flatten(2).transpose(1, 2)  # [n, s*s, mix_dim]
                final_tokens = self.cnx_proj(out_final.flatten(2).transpose(1, 2))  # [n, 49, mix_dim]

                mix_tokens = self.mixer2(torch.cat([branch_a, final_tokens], dim=1))
                mix_feat[keep] = mix_tokens.mean(dim=1)

        text_feat = self.text_encoder(text_emb)
        parts = [img_feat, mix_feat, text_feat] if self.freq_enabled else [img_feat, text_feat]
        fused = torch.cat(parts, dim=1)
        return self.nutrition_head(fused)


def pmae(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (pred - gt).abs() / (gt.abs() + 1.0)


def _predict_abs(out: dict) -> torch.Tensor:
    """모델 출력(log1p 공간)을 절대량으로 복원. 순서 [cal, fat, prot] 고정.

    calories/fat/protein은 항상 직접 예측(헤드 지도학습). carb는 여기서 예측하지 않는다 —
    Atwater 역산은 evaluate()에서 GT와 나란히 별도로 계산한다(참고 지표, mean_pmae 제외).
    """
    cal = torch.expm1(out["calories"].clamp(0, 10)).squeeze(-1)
    fat = torch.expm1(out["fat"].clamp(0, 10)).squeeze(-1)
    prot = torch.expm1(out["protein"].clamp(0, 10)).squeeze(-1)
    return torch.stack([cal, fat, prot], dim=1).clamp(min=0)


@torch.no_grad()
def evaluate(model, loader, device, force_mask_image: bool = False) -> dict:
    """절대량 PMAE. force_mask_image=True면 이미지가 있어도 강제로 mask token 사용
    (검증 5: 이미지가 실제로 기여하는지 보는 ablation)."""
    model.eval()
    all_pred, all_gt, all_carb_gt = [], [], []
    for batch in loader:
        images = batch["images"].to(device)
        has_image = batch["has_image"].to(device)
        if force_mask_image:
            has_image = torch.zeros_like(has_image)
        text_emb = batch["text_emb"].to(device)
        out = model(images, has_image, text_emb, apply_dropout=False)
        pred = _predict_abs(out).cpu()
        all_pred.append(pred)
        all_gt.append(batch["targets"])
        all_carb_gt.append(batch["carb_gt"])
    pred = torch.cat(all_pred)
    gt = torch.cat(all_gt)
    carb_gt = torch.cat(all_carb_gt)

    per_nutrient = pmae(pred, gt).mean(0)
    names = ["cal", "fat", "prot"]
    metrics = {n: per_nutrient[i].item() * 100 for i, n in enumerate(names)}
    metrics["mean_pmae"] = per_nutrient.mean().item() * 100  # carb 제외 — GT 자체가 역산값이라 참고용

    # carb는 예측·GT 둘 다 Atwater 역산 — 학습 대상이 아니라 "기록만" 하는 참고 지표.
    carb_pred = ((pred[:, 0] - 9 * pred[:, 1] - 4 * pred[:, 2]) / 4).clamp(min=0)
    metrics["carb_derived"] = pmae(carb_pred, carb_gt).mean().item() * 100
    return metrics


def make_loader(image_records, text_df, nutrition_df, name_to_emb, cfg, train: bool):
    ds = dm.MultimodalNutritionDataset(image_records, text_df, nutrition_df, name_to_emb, train=train)
    return DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=train,
                       collate_fn=dm.collate_fn, num_workers=cfg["training"].get("image_workers", 4))


def train(cfg: dict, tag: str = "") -> dict:
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    splits, nutrition_df = dm.build_dataset(cfg["data"])
    name_to_emb = dm.build_name_to_emb(splits, cfg["embedding"])
    text_in_dim = len(next(iter(name_to_emb.values())))
    print(f"[data] text embedding dim={text_in_dim}")

    train_loader = make_loader(splits["train"]["image"], splits["train"]["text"], nutrition_df,
                                name_to_emb, cfg, train=True)
    val_loader = make_loader(splits["val"]["image"], splits["val"]["text"], nutrition_df,
                              name_to_emb, cfg, train=False)
    test_loader = make_loader(splits["test"]["image"], splits["test"]["text"], nutrition_df,
                               name_to_emb, cfg, train=False)
    # 이미지 보유 test만 따로 — 이미지 ablation 비교용 (test는 원래 text 행이 없지만 방어적으로 재사용)
    test_image_only_loader = make_loader(splits["test"]["image"], nutrition_df.iloc[0:0], nutrition_df,
                                          name_to_emb, cfg, train=False)

    mc = cfg["model"]
    model = MultimodalNutritionModel(
        image_dim=mc["image_dim"], text_in_dim=text_in_dim,
        text_hidden1=mc["text_hidden1"], text_hidden2=mc["text_hidden2"],
        head_hidden=mc["head_hidden"], dropout=mc["dropout"],
        freeze_backbone_layers=mc["freeze_backbone_layers"],
        modality_dropout_prob=mc["modality_dropout_prob"],
        freq_cfg=cfg.get("freq"),
    ).to(device)

    tc = cfg["training"]
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    out_path = Path(cfg["embedding"]["cache_path"]).parent / f"best_model_multimodal{tag}.pth"

    best_val, patience_cnt, best_state = float("inf"), 0, None
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        for batch in train_loader:
            images = batch["images"].to(device)
            has_image = batch["has_image"].to(device)
            text_emb = batch["text_emb"].to(device)
            targets = batch["targets"].to(device)  # [B, 3] = cal, fat, prot (절대량)
            targets_log = torch.log1p(targets.clamp(min=0))

            opt.zero_grad()
            out = model(images, has_image, text_emb, apply_dropout=True)
            pred_log = torch.stack([out[k] for k in DIRECT_KEYS], dim=1).squeeze(-1)  # cal,fat,prot
            loss = ((pred_log - targets_log) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_m = evaluate(model, val_loader, device)
        scheduler.step(val_m["mean_pmae"])
        print(f"epoch {epoch:3d}  val mean_pmae={val_m['mean_pmae']:.2f}  "
              f"cal={val_m['cal']:.1f}  fat={val_m['fat']:.1f}  prot={val_m['prot']:.1f}  "
              f"carb_derived={val_m['carb_derived']:.1f}")

        if val_m["mean_pmae"] < best_val:
            best_val = val_m["mean_pmae"]
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "config": cfg}, out_path)  # 같은 경로 덮어쓰기 — 갱신 시마다 이전 체크포인트 자동 교체, 중단(재부팅 등)에도 최신 best 보존
            print(f"  [checkpoint] saved -> {out_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= tc["patience"]:
                print(f"[early stop] epoch {epoch}, best val mean_pmae={best_val:.2f}")
                break

    model.load_state_dict(best_state)

    print("\n=== Final Test PMAE (전체: 이미지+텍스트 혼합) ===")
    test_m = evaluate(model, test_loader, device)
    for k, v in test_m.items():
        print(f"  {k}: {v:.2f}")

    print("\n=== 이미지 보유 test만: 실제 이미지 사용 ===")
    for k, v in evaluate(model, test_image_only_loader, device, force_mask_image=False).items():
        print(f"  {k}: {v:.2f}")

    print("\n=== 이미지 보유 test만: 이미지를 강제 mask token으로 (검증 5 — 이미지 기여도 ablation) ===")
    for k, v in evaluate(model, test_image_only_loader, device, force_mask_image=True).items():
        print(f"  {k}: {v:.2f}")

    print(f"[saved] {out_path}")  # 이미 매 갱신 시점에 저장돼 있음 — 여기선 최종 확인 로그만

    return test_m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config_multimodal.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    loop = cfg.pop("loop", None)
    apply_overrides = dm.tb_train.apply_overrides  # text_baseline_the_best/train.py의 기존 loop 메커니즘 재사용

    if not loop:
        train(cfg)
    else:
        # pre-flight: 모든 실험의 override 키를 먼저 검증 — 오타 하나로 몇 시간짜리 실험을 날리지 않기 위함.
        for i, overrides in enumerate(loop, 1):
            try:
                apply_overrides(cfg, overrides or {}, quiet=True)
            except KeyError as e:
                raise SystemExit(f"[loop] pre-flight check failed on experiment {i}/{len(loop)}: {e}")
        print(f"[loop] pre-flight check passed — {len(loop)} experiments queued\n")

        results = []
        for i, overrides in enumerate(loop, 1):
            overrides = overrides or {}
            print(f"\n{'=' * 60}\n[loop] experiment {i}/{len(loop)}  overrides={overrides}\n{'=' * 60}")
            exp_cfg = apply_overrides(cfg, overrides)
            test_m = train(exp_cfg, tag=f"_exp{i}")
            results.append((overrides, test_m["mean_pmae"]))

        print(f"\n{'=' * 60}\n[loop] summary ({len(loop)} experiments)\n{'=' * 60}")
        for i, (overrides, mean_pmae) in enumerate(results, 1):
            print(f"  exp{i}: mean_pmae={mean_pmae:.2f}  overrides={overrides}")
