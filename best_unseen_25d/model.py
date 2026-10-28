import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import clip
from google.genai.errors import ServerError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class GeM(nn.Module):
    """Generalized Mean Pooling — learnable pooling exponent p."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)


class IngredientEncoder(nn.Module):
    """CLIP-based ingredient text encoder with in-memory cache.

    재료명은 학습 전체에서 고정된 유한 집합이므로 최초 1회만 encode_text를 호출하고
    이후에는 cache에서 즉시 반환한다.
    """
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self._cache: dict = {}  # ingredient_text -> embedding (cpu tensor)

    def forward(self, ingredient_texts):
        """
        Args:
            ingredient_texts: List of ingredient name strings
        Returns:
            embeddings: Tensor of shape [num_ingredients, clip_dim]
        """
        device = next(self.clip_model.parameters()).device

        uncached = [t for t in ingredient_texts if t not in self._cache]
        if uncached:
            with torch.no_grad():
                text_inputs = clip.tokenize(uncached, truncate=True).to(device)
                embs = self.clip_model.encode_text(text_inputs).float()
            for t, e in zip(uncached, embs):
                self._cache[t] = e.cpu()

        return torch.stack([self._cache[t] for t in ingredient_texts]).to(device)


class GeminiFoodNameEncoder:
    """Gemini Embeddings 2 food-name encoder with disk-persisted cache.

    Drop-in replacement for IngredientEncoder — same __call__ interface.
    Returns 3072-dim embeddings (gemini-embedding-2 default).

    Food names are a fixed set reused across every epoch and every run, so the
    cache is persisted to disk (`cache_path`) — a restart after a crash reuses
    already-embedded names instead of re-calling the API for all of them.
    """

    def __init__(self, api_key: str, device: str, cache_path: str = "gemini_foodname_emb_cache.jsonl"):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._device = device
        self._cache_path = Path(cache_path)
        self._cache: dict = {}
        if self._cache_path.exists():
            with self._cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    self._cache[entry["name"]] = torch.tensor(entry["embedding"], dtype=torch.float32)
            print(f"Loaded {len(self._cache)} cached food-name embeddings from {self._cache_path}")

    # ponytail: transient 5xx from the Gemini API otherwise kills a whole
    # 500-epoch run on an uncached name; retry with backoff before giving up.
    @retry(
        retry=retry_if_exception_type(ServerError),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(8),
    )
    def _embed(self, text: str):
        return self._client.models.embed_content(model="gemini-embedding-2", contents=text)

    def __call__(self, texts: list[str]) -> torch.Tensor:
        vecs = []
        for t in texts:
            if t not in self._cache:
                r = self._embed(t)
                embedding = torch.tensor(r.embeddings[0].values, dtype=torch.float32)
                self._cache[t] = embedding
                with self._cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"name": t, "embedding": embedding.tolist()}) + "\n")
            vecs.append(self._cache[t])
        return torch.stack(vecs).to(self._device)


class TextTransform(nn.Module):
    """Transform text embeddings to match image feature dimensions.

    2-layer MLP: Linear(ReLU(Linear(x)))
    - fc1 + ReLU: 비선형 변환
    - fc2 (no activation): 출력이 음수 포함 → Ximg에 양방향 기여 가능
    """
    def __init__(self, clip_dim=512, resnet_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(clip_dim, resnet_dim)
        self.fc2 = nn.Linear(resnet_dim, resnet_dim)

    def forward(self, text_embedding):
        """
        Args:
            text_embedding: [batch_size, clip_dim] or [clip_dim]
        Returns:
            transformed: [batch_size, resnet_dim, 1, 1] or [resnet_dim, 1, 1]
        """
        transformed = self.fc2(F.relu(self.fc1(text_embedding)))

        if transformed.dim() == 1:
            transformed = transformed.view(-1, 1, 1)
        else:
            transformed = transformed.view(transformed.size(0), -1, 1, 1)

        return transformed

class ResNetImageEncoder(nn.Module):
    """ResNet101 encoder extracting block2 (layer2) features"""
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet101(pretrained=pretrained)

        # Extract layers up to and including layer2 (block2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

    def forward(self, x):
        """
        Args:
            x: [batch_size, 3, H, W]
        Returns:
            features: [batch_size, 512, H', W'] - block2 output
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        return x


# Per-module output channels/spatial size of torchvision convnext_base().features for a
# 224x224 input (8 modules: stem, stage0, down, stage1, down, stage2, down, stage3).
# Used to size the frequency-branch mixer without hardcoding magic numbers at each tap.
CONVNEXT_BASE_TAP_CHANNELS = [128, 128, 256, 256, 512, 512, 1024, 1024]
CONVNEXT_BASE_TAP_SPATIAL = [56, 56, 28, 28, 14, 14, 7, 7]


class ConvNeXtImageEncoder(nn.Module):
    """ConvNeXt-Base encoder — full feature extractor, output [B, 1024, 7, 7] for 224x224 input."""
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import convnext_base, ConvNeXt_Base_Weights
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = convnext_base(weights=weights).features

    def forward(self, x, tap: int | None = None):
        """tap=None (default): unchanged — returns the final feature map, same as before.
        tap=int: returns (features[tap] output, final output) as a tuple, computed in a
        single pass (no double forward)."""
        if tap is None:
            return self.features(x)
        mid = None
        for i, m in enumerate(self.features):
            x = m(x)
            if i == tap:
                mid = x
        return mid, x


_TUKEY_WINDOW_CACHE: dict = {}
_RADIUS_GRID_CACHE: dict = {}


def _tukey_window_1d(n: int, alpha: float, device, dtype) -> torch.Tensor:
    """1D tapered-cosine (Tukey) window. alpha=0 -> rectangular (all ones),
    alpha=1 -> Hann. Middle (1 - alpha) fraction stays flat at 1."""
    if alpha <= 0:
        return torch.ones(n, device=device, dtype=dtype)
    x = torch.linspace(0, 1, n, device=device, dtype=dtype)
    w = torch.ones(n, device=device, dtype=dtype)
    edge = alpha / 2.0
    left = x < edge
    right = x > 1 - edge
    w[left] = 0.5 * (1 + torch.cos(torch.pi * (2 * x[left] / alpha - 1)))
    w[right] = 0.5 * (1 + torch.cos(torch.pi * (2 * x[right] / alpha - 2 / alpha + 1)))
    return w


def _tukey_window_2d(h: int, w: int, alpha: float, device, dtype) -> torch.Tensor:
    """Separable 2D Tukey window [H, W] — flat center, cosine-tapered border.
    Used to suppress the letterbox-padding edge and FFT wrap-around discontinuity
    before taking the DFT, without dimming the (centered) food content."""
    key = (h, w, alpha, device, dtype)
    if key not in _TUKEY_WINDOW_CACHE:
        wy = _tukey_window_1d(h, alpha, device, dtype)
        wx = _tukey_window_1d(w, alpha, device, dtype)
        _TUKEY_WINDOW_CACHE[key] = wy.unsqueeze(1) * wx.unsqueeze(0)
    return _TUKEY_WINDOW_CACHE[key]


def _radius_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    """Normalized distance-from-center grid [H, W] in [0, 1], matching the
    layout after fftshift (DC / lowest frequency at the exact center)."""
    key = (h, w, device, dtype)
    if key not in _RADIUS_GRID_CACHE:
        yy = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        xx = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        r = torch.sqrt(gy ** 2 + gx ** 2)
        _RADIUS_GRID_CACHE[key] = (r / r.max()).clamp(max=1.0)
    return _RADIUS_GRID_CACHE[key]


def band_split(x: torch.Tensor, r_low: float, r_high: float, window_alpha: float = 0.25) -> torch.Tensor:
    """Split an image into a low-frequency band and a high-frequency band via DFT
    masking, then reconstruct each band back to image space (iFFT).

    Args:
        x: [N, C, H, W] real image tensor (e.g. bbox-cropped, letterbox-padded, normalized).
        r_low: keep frequencies with normalized radius <= r_low (low-frequency band).
        r_high: keep frequencies with normalized radius >= r_high (high-frequency band).
        window_alpha: Tukey window taper fraction applied before the DFT, to suppress
            the letterbox-padding edge and FFT wrap-around discontinuity (0=off, 1=Hann).
    Returns:
        [N, 2*C, H, W] = cat([low_band_image, high_band_image], dim=1).
    """
    n, c, h, w = x.shape
    if window_alpha > 0:
        window = _tukey_window_2d(h, w, window_alpha, x.device, x.dtype)
        x = x * window

    freq = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    r = _radius_grid(h, w, x.device, x.dtype)
    low_mask = (r <= r_low).to(freq.dtype)
    high_mask = (r >= r_high).to(freq.dtype)

    low = torch.fft.ifft2(torch.fft.ifftshift(freq * low_mask, dim=(-2, -1))).real
    high = torch.fft.ifft2(torch.fft.ifftshift(freq * high_mask, dim=(-2, -1))).real

    # Natural images concentrate most of their energy in the low band — empirically the
    # high band's per-channel std lands ~70x smaller than the low band's. Left as-is, that
    # scale gap would make the high-frequency input near-invisible to VGG16's conv1 (whose
    # pretrained filters expect roughly unit-std input), regardless of how informative the
    # texture actually is. Rescale each band (divide only, don't re-center) to unit std per
    # (N, C) so both bands sit at the amplitude VGG's pretrained weights were calibrated for
    # — this preserves each band's own DC/mean (the low band's brightness-color cue in
    # particular) while equalizing contrast.
    eps = 1e-6
    low = low / (low.flatten(2).std(dim=-1).view(n, c, 1, 1) + eps)
    high = high / (high.flatten(2).std(dim=-1).view(n, c, 1, 1) + eps)
    return torch.cat([low, high], dim=1)


class FrequencyEncoder(nn.Module):
    """DFT low/high-frequency-band branch: band_split -> VGG16.features[:24] -> Conv+ReLU.

    Input images are expected to already be bbox-cropped (food-only content, no
    background) — the letterbox padding around them still gets partially suppressed
    by the Tukey window inside band_split. Output resolution [14, 14] matches
    ConvNeXt-Base features[5] (stage2) exactly, so no resize/projection is needed
    before the two feature maps are combined.
    """

    def __init__(self, r_low: float = 0.15, r_high: float = 0.85, window_alpha: float = 0.25,
                 out_dim: int = 512, pretrained: bool = True, freeze_vgg_layers: bool = True):
        super().__init__()
        self.r_low = r_low
        self.r_high = r_high
        self.window_alpha = window_alpha

        from torchvision.models import vgg16, VGG16_Weights
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = vgg16(weights=weights)
        self.features = vgg.features[:24]  # up to and including the 4th maxpool -> [B,512,14,14]

        old_conv1 = self.features[0]
        new_conv1 = nn.Conv2d(6, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                               stride=old_conv1.stride, padding=old_conv1.padding)
        if pretrained:
            # Tile the pretrained 3-channel RGB filters onto both the low- and
            # high-band triplets (halved so the initial output scale matches the
            # original 3-channel conv) rather than discarding the ImageNet prior.
            with torch.no_grad():
                new_conv1.weight.copy_(old_conv1.weight.repeat(1, 2, 1, 1) * 0.5)
                new_conv1.bias.copy_(old_conv1.bias)
        self.features[0] = new_conv1

        if freeze_vgg_layers:
            # features[1:17] = relu1_2..pool3, i.e. everything after the reinitialized
            # conv1 up through the 3rd maxpool. Mirrors the ConvNeXt features[:3] freeze policy.
            for p in self.features[1:17].parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Conv2d(512, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        bands = band_split(images, self.r_low, self.r_high, self.window_alpha)
        return self.proj(self.features(bands))


class MixerBlock(nn.Module):
    """MLP-Mixer block: a channel-mixing MLP (applied per-token, i.e. along the
    feature dimension) followed by a token-mixing MLP (applied along the token
    dimension via transpose). Both are residual.

    x: [B, T, D] -> channel MLP (D->channel_hidden->D) -> token MLP (T->token_hidden->T)
    """

    def __init__(self, num_tokens: int, dim: int, token_hidden: int, channel_hidden: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(channel_hidden, dim), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(token_hidden, num_tokens), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.channel_mlp(self.norm1(x))
        x = x + self.token_mlp(self.norm2(x).transpose(1, 2)).transpose(1, 2)
        return x


class NutritionHead(nn.Module):
    """Multi-task nutrition estimation head with sequential mass feedback (Option A)"""
    def __init__(self, input_dim=512, hidden_dim=4096, dropout=0.5, mass_feedback=True, detach_mass=False):
        super().__init__()
        self.mass_feedback = mass_feedback
        self.detach_mass = detach_mass

        # Shared first layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

        # 1. Mass Head (predicts portion mass first)
        self.head_mass = nn.Linear(hidden_dim, 1)

        # 2. Nutrient Heads (conditioned on both hidden features and pred_mass)
        nutri_in_dim = (hidden_dim + 1) if mass_feedback else hidden_dim
        self.head_calories = nn.Linear(nutri_in_dim, 1)
        self.head_fat = nn.Linear(nutri_in_dim, 1)
        self.head_carb = nn.Linear(nutri_in_dim, 1)
        self.head_protein = nn.Linear(nutri_in_dim, 1)

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            predictions: dict with keys ['calories', 'mass', 'fat', 'carb', 'protein']
                        each value is [batch_size, 1]
        """
        # First hidden layer with ReLU and Dropout
        hidden = F.relu(self.fc1(x))
        hidden = self.dropout(hidden)

        # Step 1: Predict mass first
        pred_mass = self.head_mass(hidden)

        # Step 2: Feedback mass to other nutrient heads (Option A: full end-to-end gradient flow)
        if self.mass_feedback:
            m_feat = pred_mass.detach() if self.detach_mass else pred_mass
            nutri_in = torch.cat([hidden, m_feat], dim=-1)  # [batch_size, hidden_dim + 1]
        else:
            nutri_in = hidden

        predictions = {
            'calories': self.head_calories(nutri_in),
            'mass': pred_mass,
            'fat': self.head_fat(nutri_in),
            'carb': self.head_carb(nutri_in),
            'protein': self.head_protein(nutri_in)
        }

        return predictions


def compute_mae_loss(predictions, targets, nutrition_scalers):
    """
    Compute mean absolute error for multi-task learning with RobustScaler normalization

    Args:
        predictions: dict with keys ['calories', 'mass', 'fat', 'carb', 'protein']
        targets: dict with same keys as predictions
        nutrition_scalers: dict of sklearn RobustScaler parameters for each nutrition type
    Returns:
        total_loss: sum of normalized MAE losses (equal weight)
        loss_dict: individual losses for each task (unnormalized for display)
    """
    loss_dict = {}
    total_loss = 0.0

    # Only compute loss for calories, fat, carb, protein (mass removed)
    nutrition_keys = ['calories', 'fat', 'carb', 'protein']

    for key in nutrition_keys:
        # Normalize using sklearn RobustScaler formula: (x - median) / IQR
        scaler_params = nutrition_scalers[key]
        center = scaler_params['center']  # median
        scale = scaler_params['scale']    # IQR

        pred_norm = (predictions[key] - center) / scale
        target_norm = (targets[key] - center) / scale

        # Compute normalized MAE
        mae_norm = F.l1_loss(pred_norm, target_norm)

        # Store unnormalized MAE for display
        mae_unnorm = F.l1_loss(predictions[key], targets[key])
        loss_dict[key] = mae_unnorm.item()

        # Add normalized loss to total (equal weight)
        total_loss += mae_norm

    return total_loss, loss_dict
