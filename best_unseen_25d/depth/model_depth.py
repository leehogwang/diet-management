"""깊이 브랜치 + 이미지 쪽 샘플별 적응형 게이트 모델. 기존 MultimodalNutritionModel 을 상속(기존 파일 수정 없음).

구조
  이미지 → 비전(ConvNeXt+GeM, 1024) ┐
  이미지 → 주파수(FFT+Mixer, 256)   ├ 샘플별 게이트(이미지쪽 특징만 입력, 브랜치별 최소 비중 floor) → 가중 concat ┐
  깊이   → 깊이 CNN(ResNet18, 256)  ┘                                                                          ├ NutritionHead
  텍스트 → 64D 병목 (게이트 없음, 고정 비중) ─────────────────────────────────────────────────────────────────┘
텍스트 과의존 방지: (1) 게이트는 텍스트 비중을 바꿀 수 없음 (2) 게이트 입력에 텍스트 없음
                    (3) 텍스트 드롭아웃(기본 0.3) — 기존 모델의 드롭아웃은 이미지를 꺼서 오히려 텍스트 의존을 키웠음
use_depth=False 면 같은 구조에서 깊이 브랜치만 뺀 비교 모델(게이트는 비전·주파수 2개).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_baseline_the_best"))
from train_multimodal import MultimodalNutritionModel  # noqa: E402


class DepthEncoder(nn.Module):
    """깊이 맵(112x112, 미터, 패딩 0) → 256D. 입력 채널: log 깊이, 중앙값 대비 상대 깊이. 스칼라: log 중앙값 깊이, log fx, 유효 비율."""

    def __init__(self, out_dim=256):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        w = r.conv1.weight.data.mean(1, keepdim=True)  # RGB 가중치 평균 → 2채널로
        r.conv1 = nn.Conv2d(2, 64, 7, 2, 3, bias=False)
        r.conv1.weight.data = w.repeat(1, 2, 1, 1) / 2
        r.fc = nn.Identity()
        self.cnn = r
        self.proj = nn.Sequential(nn.Linear(512 + 3, out_dim), nn.ReLU(inplace=True))

    def forward(self, depth, meta):
        valid = (depth > 0).float()  # [B,1,H,W]
        n = valid.flatten(1).sum(1).clamp(min=1)
        d = depth.clamp(min=1e-3)
        # 유효 픽셀 중앙값 (패딩 제외)
        med = torch.stack([di[vi > 0].median() if (vi > 0).any() else di.new_tensor(1.0) for di, vi in zip(d, valid)])
        med = med.clamp(min=1e-2)
        x = torch.cat([torch.log(d) * valid, (d - med[:, None, None, None]) / med[:, None, None, None] * valid], 1)
        feat = self.cnn(x)
        scal = torch.stack([torch.log(med), torch.log(meta[:, 0].clamp(min=1.0)), n / valid[0].numel()], 1)
        return self.proj(torch.cat([feat, scal], 1))


class DepthGatedNutritionModel(MultimodalNutritionModel):
    def __init__(self, *, use_depth=True, depth_dim=256, gate_floor=0.1, text_drop=0.3, image_drop=0.1, **kw):
        kw["modality_dropout_prob"] = 0.0  # 부모의 이미지 드롭아웃은 끄고 아래에서 직접 처리
        super().__init__(**kw)
        assert self.freq_enabled, "주파수 브랜치 설정(freq.enabled=true)이 필요합니다"
        self.use_depth, self.text_drop, self.image_drop = use_depth, text_drop, image_drop
        image_dim = self.mask_token.numel()
        mix_dim = self.mix_mask_token.numel()
        text_dim = [m for m in self.text_encoder if isinstance(m, nn.BatchNorm1d)][-1].num_features  # = text_hidden2
        dims = [image_dim, mix_dim] + ([depth_dim] if use_depth else [])
        if use_depth:
            self.depth_encoder = DepthEncoder(depth_dim)
            self.depth_mask_token = nn.Parameter(torch.randn(depth_dim) * 0.02)
        self.text_mask_token = nn.Parameter(torch.randn(text_dim) * 0.02)
        self.gate = nn.Sequential(nn.Linear(sum(dims), 128), nn.ReLU(inplace=True), nn.Linear(128, len(dims)))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)  # 시작은 균등 가중치
        self.gate_floor = gate_floor
        head_cls = type(self.nutrition_head)
        self.nutrition_head = head_cls(sum(dims) + text_dim, self.nutrition_head.fc1.out_features,
                                       dropout=self.nutrition_head.dropout.p, mass_feedback=self.nutrition_head.mass_feedback)
        self.last_gate = None  # [B, K] 최근 게이트 (분석용)

    def _image_branches(self, images):
        out_tap, out_final = self.image_encoder(images, tap=self.cnx_tap)
        img = self.pooling(out_final).view(images.shape[0], -1)
        fm = torch.cat([self.freq_encoder(images), out_tap], 1)
        n, c, h, w = fm.shape
        fm = self.mixer1(fm.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(n, c, h, w)
        a = self.conv_blocks(fm).flatten(2).transpose(1, 2)
        f = self.cnx_proj(out_final.flatten(2).transpose(1, 2))
        mix = self.mixer2(torch.cat([a, f], 1)).mean(1)
        return img, mix

    def forward(self, images, has_image, text_emb, apply_dropout: bool, depth=None, depth_meta=None, has_depth=None):
        B, dev = images.shape[0], images.device
        keep = has_image.clone().bool()
        if apply_dropout and self.image_drop > 0:
            keep &= ~(torch.rand(B, device=dev) < self.image_drop)

        img = self.mask_token.expand(B, -1).clone()
        mix = self.mix_mask_token.expand(B, -1).clone()
        if keep.any():
            img[keep], mix[keep] = self._image_branches(images[keep])
        feats = [img, mix]
        if self.use_depth:
            dfeat = self.depth_mask_token.expand(B, -1).clone()
            dk = keep & (has_depth.bool() if has_depth is not None else torch.ones_like(keep))
            if dk.any():
                dfeat[dk] = self.depth_encoder(depth[dk], depth_meta[dk])
            feats.append(dfeat)

        K = len(feats)
        logits = self.gate(torch.cat(feats, 1))
        g = self.gate_floor + (1 - K * self.gate_floor) * torch.softmax(logits, 1)  # 각 브랜치 최소 floor
        self.last_gate = g.detach()
        gated = [f * (K * g[:, k:k + 1]) for k, f in enumerate(feats)]  # 균등일 때 원래 크기 유지

        text = self.text_encoder(text_emb)
        if apply_dropout and self.text_drop > 0:
            td = torch.rand(B, device=dev) < self.text_drop
            text = torch.where(td[:, None], self.text_mask_token.expand(B, -1), text)
        return self.nutrition_head(torch.cat(gated + [text], 1))
