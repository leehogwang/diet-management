"""model_stage3_296_final.pth 의 state_dict 구조에서 역으로 복원한 모델.

모든 모듈 이름과 텐서 shape 은 체크포인트와 1:1로 맞췄다 (check_compat.py 로 strict 로드 검증).
체크포인트에서 알 수 없는 부분(분기 연결 순서, 활성화 함수, 드롭아웃 위치 등)은 [추정] 으로 표시.

입력  : image [B,3,224,224] (0~1 범위, 정규화 전), text_emb [B,3072], has_image [B] bool
출력  : {"calories","mass","fat","carb","protein"} 각 [B,1]  (train.py 가 저장한 target_stats 로 역정규화)

                 ┌─ ConvNeXt-Base features[0:6] ─► tap [512,14,14] ─┐
 image ──────────┤                                                   ├─concat─► [1024,14,14] ─► 196 tokens ─► mixer1
                 └─ FFT 저/고주파 분해 (6ch) ─► VGG16[:24] ─► proj ─► [512,14,14] ─┘                               │
                                                                                                                 ▼
                 ConvNeXt features[6:8] ─► [1024,7,7] ─┬─► GeM ─► img_feat 1024                  conv_blocks ─► [256,3,3] ─► 9 tokens
                                                       └─► 49 tokens ─► cnx_proj(1024→256) ──────────────┐            │
                                                                                                          └── concat 58 tokens ─► mixer2 ─► mean ─► mix_feat 256
 text_emb 3072 ─► text_encoder ─► 512

 head input = concat[text 512, img 1024, mix 256] = 1792 ─► NutritionHead(2048) ─► 5 heads
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ConvNeXt_Base_Weights, VGG16_Weights, convnext_base, vgg16

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ─── 주파수 분해 ──────────────────────────────────────────────────────────────
class FrequencySplit(nn.Module):
    """이미지를 FFT 로 저주파/고주파 성분으로 나눠 공간 영역 이미지 2장(6ch)으로 돌려준다.

    [추정] 반지름 r 은 중심→가장자리 중앙이 1 이 되도록 정규화.
           저주파: r <= r_low,  고주파: r >= r_high (그 사이 대역은 버림).
           Tukey 창(alpha)을 FFT 전에 곱해 경계 아티팩트를 줄임.
    """

    def __init__(self, size: int, r_low: float, r_high: float, window_alpha: float):
        super().__init__()
        yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        c = (size - 1) / 2
        r = torch.sqrt((yy - c) ** 2 + (xx - c) ** 2) / (size / 2)
        self.register_buffer("low_mask", (r <= r_low).float(), persistent=False)
        self.register_buffer("high_mask", (r >= r_high).float(), persistent=False)
        w = self._tukey(size, window_alpha)
        self.register_buffer("window", torch.outer(w, w), persistent=False)

    @staticmethod
    def _tukey(n: int, alpha: float) -> torch.Tensor:
        if alpha <= 0:
            return torch.ones(n)
        x = torch.linspace(0, 1, n)
        w = torch.ones(n)
        edge = alpha / 2
        lo, hi = x < edge, x > 1 - edge
        w[lo] = 0.5 * (1 + torch.cos(math.pi * (2 * x[lo] / alpha - 1)))
        w[hi] = 0.5 * (1 + torch.cos(math.pi * (2 * x[hi] / alpha - 2 / alpha + 1)))
        return w

    def forward(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01.float() * self.window
        spec = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
        low = torch.fft.ifft2(torch.fft.ifftshift(spec * self.low_mask, dim=(-2, -1)), norm="ortho").real
        high = torch.fft.ifft2(torch.fft.ifftshift(spec * self.high_mask, dim=(-2, -1)), norm="ortho").real
        # [추정] 저주파는 ImageNet 정규화, 고주파는 이미지별 표준화
        mean = low.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = low.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        low = (low - mean) / std
        high = (high - high.mean(dim=(-2, -1), keepdim=True)) / (high.std(dim=(-2, -1), keepdim=True) + 1e-6)
        return torch.cat([low, high], dim=1)


class FreqEncoder(nn.Module):
    """6채널 입력 VGG16 (features[:24], pool4 까지) + 3x3 proj. 출력 [B,512,14,14]."""

    def __init__(self, out_dim: int = 512, pretrained: bool = True):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        feats = vgg.features[:24]
        old = feats[0]
        new = nn.Conv2d(6, old.out_channels, 3, padding=1)
        with torch.no_grad():  # 사전학습 가중치를 두 입력(저/고주파)에 나눠 복사
            new.weight.copy_(torch.cat([old.weight, old.weight], dim=1) / 2)
            new.bias.copy_(old.bias)
        feats[0] = new
        self.features = feats
        self.proj = nn.Sequential(nn.Conv2d(512, out_dim, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.proj(self.features(x))


# ─── Mixer / 기타 ────────────────────────────────────────────────────────────
class MixerBlock(nn.Module):
    def __init__(self, num_tokens: int, dim: int, token_hidden: int, channel_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(token_hidden, num_tokens), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(channel_hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):  # x: [B, N, C]
        x = x + self.token_mlp(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(self.norm2(x))


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p).flatten(1)


class ConvNeXtEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.features = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None).features


class NutritionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head_calories = nn.Linear(hidden_dim, 1)
        self.head_mass = nn.Linear(hidden_dim, 1)
        self.head_fat = nn.Linear(hidden_dim, 1)
        self.head_carb = nn.Linear(hidden_dim, 1)
        self.head_protein = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.dropout(F.relu(self.fc1(x)))
        return {
            "calories": self.head_calories(h), "mass": self.head_mass(h), "fat": self.head_fat(h),
            "carb": self.head_carb(h), "protein": self.head_protein(h),
        }


# ─── 전체 모델 ───────────────────────────────────────────────────────────────
class Stage3NutritionModel(nn.Module):
    def __init__(self, cfg: dict, pretrained: bool = True):
        super().__init__()
        m, fq = cfg["model"], cfg["freq"]
        size = cfg["data"]["image_size"]
        emb_dim = cfg["embedding"]["dim"]
        img_dim = m["image_dim"]
        self.cnx_tap = fq["cnx_tap"]
        self.modality_dropout_prob = m["modality_dropout_prob"]
        tap_ch = 512  # ConvNeXt-Base features[5] 채널
        grid1 = size // 16  # 14
        grid_cnx = size // 32  # 7
        grid_conv = grid1 // 2 // 2  # 14 → 7 → 3
        n_tok2 = grid_cnx * grid_cnx + grid_conv * grid_conv  # 49 + 9 = 58

        self.mask_token = nn.Parameter(torch.zeros(img_dim))
        self.mix_mask_token = nn.Parameter(torch.zeros(fq["mixer2_dim"]))
        self.image_encoder = ConvNeXtEncoder(pretrained)
        self.pooling = GeM()
        self.text_encoder = nn.Sequential(
            nn.Linear(emb_dim, m["text_hidden1"]), nn.BatchNorm1d(m["text_hidden1"]), nn.ReLU(inplace=True), nn.Dropout(m["dropout"]),
            nn.Linear(m["text_hidden1"], m["text_hidden2"]), nn.BatchNorm1d(m["text_hidden2"]), nn.ReLU(inplace=True), nn.Dropout(m["dropout"]),
        )
        self.freq_split = FrequencySplit(size, fq["r_low"], fq["r_high"], fq["window_alpha"])
        self.freq_encoder = FreqEncoder(fq["vgg_out_dim"], pretrained)
        mix1_dim = tap_ch + fq["vgg_out_dim"]  # 1024
        self.mixer1 = MixerBlock(grid1 * grid1, mix1_dim, fq["mixer1_token_hidden"], fq["mixer1_channel_hidden"])
        c1, c2 = fq["conv_dims"]
        self.conv_blocks = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(mix1_dim, c1, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.cnx_proj = nn.Linear(img_dim, fq["mixer2_dim"])
        self.mixer2 = MixerBlock(n_tok2, fq["mixer2_dim"], fq["mixer2_token_hidden"], fq["mixer2_channel_hidden"])
        self.nutrition_head = NutritionHead(m["text_hidden2"] + img_dim + fq["mixer2_dim"], m["head_hidden"], m["dropout"])

        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    # 학습 시 얼릴 사전학습 레이어 [추정]
    def freeze_pretrained_layers(self, cfg: dict) -> list[str]:
        frozen = []
        if cfg["model"].get("freeze_backbone_layers"):
            for i in range(4):  # stem, stage1, downsample, stage2
                for p in self.image_encoder.features[i].parameters():
                    p.requires_grad = False
            frozen.append("image_encoder.features[0:4]")
        if cfg["freq"].get("freeze_vgg_layers"):
            for i in range(1, 17):  # 첫 conv(6ch, 새로 만든 층)는 학습, block1 나머지~block3 고정
                for p in self.freq_encoder.features[i].parameters():
                    p.requires_grad = False
            frozen.append("freq_encoder.features[1:17]")
        return frozen

    def _image_branch(self, img01: torch.Tensor):
        x = (img01 - self.img_mean) / self.img_std
        tap = None
        for i, layer in enumerate(self.image_encoder.features):
            x = layer(x)
            if i == self.cnx_tap:
                tap = x  # [B,512,14,14]
        cnx = x  # [B,1024,7,7]
        img_feat = self.pooling(cnx)

        freq = self.freq_encoder(self.freq_split(img01))  # [B,512,14,14]
        fused = torch.cat([tap, freq], dim=1)  # [추정] 순서: ConvNeXt tap → 주파수
        B, C, H, W = fused.shape
        t1 = self.mixer1(fused.flatten(2).transpose(1, 2))  # [B,196,1024]
        conv = self.conv_blocks(t1.transpose(1, 2).reshape(B, C, H, W))  # [B,256,3,3]
        t_cnx = self.cnx_proj(cnx.flatten(2).transpose(1, 2))  # [B,49,256]
        t2 = self.mixer2(torch.cat([t_cnx, conv.flatten(2).transpose(1, 2)], dim=1))  # [B,58,256]
        return img_feat, t2.mean(dim=1)

    def forward(self, image, text_emb, has_image=None, modality_dropout: bool = False, image_mode: str = "full"):
        B = text_emb.shape[0]
        device = text_emb.device
        has_image = torch.ones(B, dtype=torch.bool, device=device) if has_image is None else has_image.bool()
        use_img = has_image.clone()
        use_txt = torch.ones(B, dtype=torch.bool, device=device)

        if image_mode == "masked":
            use_img[:] = False
        elif modality_dropout and self.training:
            # [추정] 이미지가 있는 샘플에서 p/2 확률로 이미지 제거, p/2 확률로 텍스트 제거
            r = torch.rand(B, device=device)
            p = self.modality_dropout_prob
            drop_img = use_img & (r < p / 2)
            drop_txt = use_img & (r >= p / 2) & (r < p)
            use_img &= ~drop_img
            use_txt &= ~drop_txt

        img_feat = self.mask_token.expand(B, -1).clone()
        mix_feat = self.mix_mask_token.expand(B, -1).clone()
        if use_img.any():
            f_img, f_mix = self._image_branch(image[use_img])
            img_feat[use_img] = f_img.to(img_feat.dtype)
            mix_feat[use_img] = f_mix.to(mix_feat.dtype)

        txt_feat = self.text_encoder(text_emb)
        txt_feat = txt_feat * use_txt.unsqueeze(1).to(txt_feat.dtype)
        return self.nutrition_head(torch.cat([txt_feat, img_feat, mix_feat], dim=1))  # [추정] 순서
