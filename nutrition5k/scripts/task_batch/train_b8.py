"""배터리8 학습기 — 챔피언(mx_rmse)에서 **변수 하나만** 바꾸는 통제 실험.

챔피언 재현 조건 (train_strong.py 의 mx_rmse arm 과 100% 동일):
  backbone=convnext_small, res=384, crop=1, lam=30, lr=2e-5, bs=16,
  epochs=60, patience=10, monitor=rmse,
  --init-from outputs/.../cs384_crop_seed{S}/checkpoints/best.pt

여기에 세 가지 실험 축을 추가한다.

1) --freeze : ConvNeXt 파인튜닝 범위
   none  전부 학습 (= 챔피언 현행. 확인 결과 50,286,772 / 100% trainable)
   stem  features[0..1]   동결 (stem + stage1)                    ~0.5% 동결
   early features[0..3]   동결 (stem + stage1 + ds + stage2)       ~2.5% 동결
   late  features[0..5]   동결 (stage3 까지, stage4만 학습)         ~66% 동결
   head  backbone 전체 동결 (헤드만 학습 = 선형 프로빙)             ~98% 동결
   동결 모듈은 requires_grad=False + eval() (ConvNeXt 의 StochasticDepth 를 꺼서
   '고정 특징추출기' 의미를 정확히 구현. LayerNorm 은 running stat 이 없어 영향 없음)

2) --pool : 전역 풀링
   avg     AdaptiveAvgPool2d(1)  (= 챔피언 현행)
   max     AdaptiveMaxPool2d(1)
   avgmax  avg 와 max 를 concat (특징 768 -> 1536)
   attn    어텐션 풀링 — 칸별 점수를 내는 작은 MLP(1x1 conv 768->128->1) 의 softmax 가중합.
           avg(=전부 같은 무게)와 max(=한 칸만) 사이를 학습으로 정하는 방식이며,
           **점수층을 0 으로 초기화하면 균등가중 = avg 와 수식적으로 완전히 동일**하다
           (검증 완료: 초기 상태에서 avg 와 출력 차 0.0). 즉 챔피언에서 출발해
           '가중치를 학습해서 번 순이득'만 측정된다. 추가 파라미터 98,561 개(전체의 0.2%).

3) --crop-box : 접시 크롭 박스 (2026-08-20 추가)
   mixed  검출 성공하면 검출박스, 실패하면 고정박스 (= 챔피언 현행. 88.0%가 고정박스)
   fixed  모든 접시에 동일한 고정박스 (110,15,575,470) — 박스 통일
   학습·검증·테스트 전부 같은 모드로 바뀐다(전처리 통일이므로 test-time 만 바꾸는 게 아니다).
   test 검출박스 60장이 고정박스 447장보다 PMA 5.73%p 나빴다는 관찰의 인과 검증용.

4) --spatial : 공간 프레임 벡터 주입
   none    주입 안함 (= 챔피언 현행)
   bbox    crop bbox 기하 5차원만 (cx,cy,bw,bh,barea)
   full    bbox 5 + 마스크 기하 4 + 깊이 통계 3 = 12차원
   senior5 ★선배님이 지목하신 정확한 5차원 (2026-08-20 추가)
           [barea(면적비), cx, cy, aspect(=bw/bh 가로세로비), depth_mean(마스크내 평균깊이)]
           full 12차원과 성분이 겹치지만 **aspect 는 12차원에 없던 파생 성분**이라
           엄밀히 말해 부분집합이 아니다. 그래서 별도 arm 으로 직접 검증한다.
   벡터는 train 분포로 표준화 -> 작은 MLP(→64d) -> 이미지 특징에 concat.

체크포인트 이어받기: 특징차원이 바뀌는 arm(avgmax/spatial)에서도 헤드 1층의
**앞쪽 768열에 챔피언 가중치를 그대로 복사하고 신규 열은 0으로 초기화**한다.
=> 학습 시작 시점의 출력이 챔피언과 정확히 동일해, '추가분의 순효과'만 측정된다.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (ConvNeXt_Small_Weights, convnext_small,
                                ConvNeXt_Base_Weights, convnext_base)

ROOT = Path("/home/students/cs/202520906/nutrition5k_pipeline")
sys.path.insert(0, str(ROOT))
from nutrition5k_cbam.utils import save_json, seed_everything  # noqa: E402
from scripts.task_batch.train_depth import NUTR, OV, pma, MANIFEST_DIR  # noqa: E402
from scripts.task_batch.train_grams import parse_grams, usable, VOCAB_K, DSMETA  # noqa: E402
from scripts.task_batch import train_grams_crop as TGC  # noqa: E402
from scripts.task_batch.train_grams_crop import CropDS, crop_depth, crop_rgb  # noqa: E402
from scripts.task_batch.train_strong import _nrmse  # noqa: E402
from scripts.task_batch.precompute_spatial import (  # noqa: E402
    BBOX_ONLY_IDX, DIM as SPDIM, FLIP_X_IDX, NAMES as SPNAMES)

RUNS = ROOT / "outputs/task_runs/task_b8/runs"
SPATIAL_NPZ = ROOT / "outputs/task_runs/task_b8/spatial_feats.npz"
# 동결 범위 -> 동결할 features 인덱스
FREEZE_SETS = {
    "none": [],
    "stem": [0, 1],
    "early": [0, 1, 2, 3],
    "late": [0, 1, 2, 3, 4, 5],
    "head": [0, 1, 2, 3, 4, 5, 6, 7],   # + classifier
}


# --------------------------------------------------------------------------- data
class SpatialCropDS(CropDS):
    """CropDS + 공간 프레임 벡터.

    주의 두 가지
      1) flip 증강 시 x 좌표 성분은 미러링해야 한다. 미러링은 **표준화 전 원값**(0~1 좌표)
         에서 1-v 로 해야 의미가 맞으므로, 원값을 들고 있다가 __getitem__ 안에서
         미러링 -> 슬라이스 -> 표준화 순서로 처리한다.
      2) 표준화 통계(smu/ssd)는 train 접시로만 계산한다(val/test 누수 방지).
    """

    def __init__(self, *a, spat_raw=None, sp_idx=None, smu=None, ssd=None, **kw):
        super().__init__(*a, **kw)
        self.spat_raw = spat_raw   # dish -> np.float32 (SPDIM,) 원값
        self.sp_idx = sp_idx       # 사용할 성분 인덱스(원 12차원 기준). None = 미사용
        self.smu = smu; self.ssd = ssd

    def __getitem__(self, i):
        if self.sp_idx is None:
            x, y, gy, dd = super().__getitem__(i)
            return x, y, gy, dd, torch.zeros(1)
        d = self.d[i]
        flip = self.train and bool(torch.rand(1).item() < 0.5)
        rgb = crop_rgb(OV / d / "rgb.png", d, self.res, flip)
        dep = crop_depth(OV / d / "depth_raw.png", d, self.res, flip)
        x = torch.from_numpy(np.concatenate([rgb, dep], axis=0))
        y = torch.tensor(self.t[d], dtype=torch.float32)
        gy = torch.from_numpy(self.gvec(d))
        v = self.spat_raw[d].copy()
        if flip:
            for k in FLIP_X_IDX:
                v[k] = 1.0 - v[k]
        s = ((v[self.sp_idx] - self.smu) / self.ssd).astype(np.float32)
        return x, y, gy, d, torch.from_numpy(s)


# 원 12차원 뒤에 파생 성분 aspect(=bw/bh) 를 13번째로 붙인다(선배님 5차원 전용).
# flip 미러링은 원값 인덱스 기준(FLIP_X_IDX=[0,6])이고 aspect 는 좌우반전에 불변이라
# 기존 SpatialCropDS 를 한 줄도 고치지 않고 그대로 쓸 수 있다.
SPNAMES13 = list(SPNAMES) + ["aspect"]
ASPECT_IDX = SPDIM                       # 12
SENIOR5_IDX = [4, 0, 1, ASPECT_IDX, 10]  # barea, cx, cy, aspect, depth_mean


def load_spatial(mode):
    """mode -> (dish->원값 dict, 사용 인덱스, 차원). 'none' 이면 (None,None,0)."""
    if mode == "none":
        return None, None, 0
    z = np.load(SPATIAL_NPZ, allow_pickle=True)
    dishes = [str(x) for x in z["dishes"]]
    feats = z["feats"].astype(np.float32)
    # 13번째 열 = 가로세로비. bh=0 인 병리 케이스는 1.0(정사각)으로 둔다.
    bw, bh = feats[:, 2], feats[:, 3]
    aspect = np.where(bh > 1e-6, bw / np.maximum(bh, 1e-6), 1.0).astype(np.float32)
    feats = np.concatenate([feats, aspect[:, None]], axis=1)
    if mode == "bbox":
        idx = np.array(BBOX_ONLY_IDX)
    elif mode == "senior5":
        idx = np.array(SENIOR5_IDX)
    else:
        idx = np.array(list(range(SPDIM)))
    return {d: feats[i] for i, d in enumerate(dishes)}, idx, len(idx)


# -------------------------------------------------------------------------- model
def build_convnext4ch(pretrained=True, arch="convnext_small"):
    """4채널 stem 으로 바꾼 ConvNeXt. train_strong.build_backbone 과 동일한 수술.

    arch 로 small(768d) / base(1024d) 를 고른다. stem 수술은 둘 다 동일하고 폭만 다르다.
    """
    if arch == "convnext_base":
        bb = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
    else:
        bb = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT if pretrained else None)
    stem = bb.features[0][0]
    new = nn.Conv2d(4, stem.out_channels, kernel_size=4, stride=4)
    with torch.no_grad():
        new.weight[:, :3] = stem.weight.data
        new.weight[:, 3:] = stem.weight.data.mean(1, keepdim=True)
        new.bias.copy_(stem.bias.data)
    bb.features[0][0] = new
    in_feat = bb.classifier[2].in_features           # 768
    bb.classifier[2] = nn.Identity()                 # LayerNorm2d + Flatten 만 남김
    return bb, in_feat


def gram_nutrition(vocab, dishes):
    """재료 이름 -> 1g 당 [cal, fat, carb, protein]. 주어진 접시들에서만 평균낸다(누수 방지)."""
    acc = {n: [] for n in vocab}
    keep = set(dishes)
    for cafe in ["dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"]:
        for line in (DSMETA / cafe).open():
            p = line.rstrip("\n").split(",")
            if not p or not p[0].startswith("dish_") or p[0] not in keep:
                continue
            for i in range(6, len(p) - 6, 7):
                name = p[i + 1].strip()
                if name not in acc:
                    continue
                try:
                    g = float(p[i + 2])
                    if g > 0:
                        acc[name].append([float(p[i + 3 + j]) / g for j in range(4)])
                except ValueError:
                    pass
    M = np.zeros((len(vocab), 4), np.float32)
    for k, n in enumerate(vocab):
        if acc[n]:
            M[k] = np.mean(acc[n], 0)
    return M


class B8Net(nn.Module):
    def __init__(self, pool="avg", spatial_dim=0, sp_embed=64, vocab_k=VOCAB_K,
                 pretrained=True, dropout=0.2, arch="convnext_small", decode="head",
                 fat_head="dedicated"):
        super().__init__()
        self.pool = pool
        self.arch = arch
        self.backbone, self.base_dim = build_convnext4ch(pretrained, arch)
        fd = self.base_dim * (2 if pool == "avgmax" else 1)
        if pool == "attn":
            # 칸별 점수를 내는 작은 MLP(1x1 conv 2층). softmax 가중합 = '어텐션 풀링'.
            # ★마지막 층을 0 으로 초기화하면 점수가 전부 0 -> softmax 가 균등가중 -> 평균 풀링과
            #   수식적으로 완전히 동일하다. 즉 학습 시작 시점이 챔피언과 같아서 순효과만 측정된다.
            self.attn = nn.Sequential(
                nn.Conv2d(self.base_dim, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))
            nn.init.zeros_(self.attn[2].weight); nn.init.zeros_(self.attn[2].bias)
        self.sp_embed = sp_embed if spatial_dim > 0 else 0
        if spatial_dim > 0:
            self.sp_mlp = nn.Sequential(
                nn.Linear(spatial_dim, sp_embed), nn.GELU(), nn.Linear(sp_embed, sp_embed))
        self.feat_dim = fd + self.sp_embed
        self.nutr_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.feat_dim, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(512, 4), nn.Softplus(),
        )
        self.aux_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.feat_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, vocab_k),
        )
        # ── 조성 축 (2026-09-10): 영양소를 '예측 그램 × 재료별 영양표' 로 **직접 계산**한다.
        #   현행(decode="head")은 영양소 헤드와 그램 보조헤드가 **독립**이라 서로 일치할 이유가 없다.
        #   decode="grams" 는 aux_head 가 내는 표준화 log1p(그램)을 실제 그램으로 되돌린 뒤
        #   고정 영양표(K,4)를 곱한다. 표는 데이터셋 라벨에서 계산한 상수라 **학습 파라미터가 아니다.**
        #   어휘 밖 재료(test 접시 평균 19%)는 잔차 헤드가 흡수한다.
        #   decode="head" 면 아래 경로를 만들지도 않으므로 기존 동작과 100% 동일하다.
        # ── 절충안 (2026-09-20, STEP 2): decode="hybrid"
        #   경로 A (칼로리·탄수·단백) = 예측그램 × 고정 영양표 + 잔차헤드   ← grams 와 동일
        #   경로 B (지방)            = 사진 특징에서 FC 로 **직접** 예측
        #   근거(STEP 1): 경유 모델에서 지방은 잔차헤드 기여가 0.0 이라 그램 오차를 100% 떠안았다
        #   (부호 있는 상관 r=+0.934). 칼로리·탄수·단백은 잔차가 30~62% 를 흡수해 멀쩡했다.
        #   지방만 잔차 완충이 없으므로, 지방만 떼어내 직접 예측으로 되돌린다.
        self.decode = decode
        self.fat_head_mode = fat_head
        if decode in ("grams", "hybrid"):
            self.register_buffer("nutr_matrix", torch.zeros(vocab_k, 4))
            self.register_buffer("g_mu", torch.zeros(vocab_k))
            self.register_buffer("g_sd", torch.ones(vocab_k))
            self.res_head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(self.feat_dim, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 4), nn.Softplus(),
            )
        if decode == "hybrid":
            if fat_head == "deep":
                # 대안 B (명세 2-C): 지방 전용 FC 를 3층으로 확장. dedicated 와 초기화·출발이
                # 같으므로 hyb196 과의 차이는 **용량 한 가지**다(순수 용량 절제실험).
                self.fat_fc = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(self.feat_dim, 512), nn.ReLU(inplace=True),
                    nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1),
                )
            else:
                # 명세 그대로: Linear(feat,256) -> ReLU -> Linear(256,1)
                self.fat_fc = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(self.feat_dim, 256), nn.ReLU(inplace=True),
                    nn.Linear(256, 1),
                )
            if fat_head in ("dedicated", "deep"):
                self.fat_out = nn.Softplus()      # 독립 헤드 (0 이상 보장)
            else:
                # "resid": 챔피언 영양소 헤드의 지방 출력 + 보정. 마지막 층을 0 으로 초기화하면
                # 학습 시작 시점의 지방 예측이 챔피언과 **정확히 같다**(이 저장소의 표준 수법).
                nn.init.zeros_(self.fat_fc[3].weight); nn.init.zeros_(self.fat_fc[3].bias)
        self._frozen = []

    # ---- 풀링 교체는 forward 에서 처리 (classifier 의 LayerNorm2d 를 공유해야 하므로)
    def features(self, x):
        fm = self.backbone.features(x)                       # (B,768,h,w)
        if self.pool == "avg":
            f = self.backbone.classifier(torch.nn.functional.adaptive_avg_pool2d(fm, 1))
        elif self.pool == "max":
            f = self.backbone.classifier(torch.nn.functional.adaptive_max_pool2d(fm, 1))
        elif self.pool == "avgmax":
            a = self.backbone.classifier(torch.nn.functional.adaptive_avg_pool2d(fm, 1))
            m = self.backbone.classifier(torch.nn.functional.adaptive_max_pool2d(fm, 1))
            f = torch.cat([a, m], dim=1)                     # 앞 768 = avg = 챔피언 자리
        elif self.pool == "attn":
            b, c, h, w = fm.shape
            score = self.attn(fm).flatten(2)                 # (B,1,h*w)
            wgt = torch.softmax(score, dim=-1).view(b, 1, h, w)
            f = self.backbone.classifier((fm * wgt).sum((2, 3), keepdim=True))
        else:
            raise ValueError(self.pool)
        return f

    def forward(self, x, s=None):
        f = self.features(x)
        if self.sp_embed:
            f = torch.cat([f, self.sp_mlp(s)], dim=1)
        aux = self.aux_head(f)
        if self.decode in ("grams", "hybrid"):
            # 표준화 log1p -> 그램. clamp 상한 7.6 = log1p(2000g) (수치 폭주 방지)
            g = torch.expm1(torch.clamp(aux.float() * self.g_sd + self.g_mu, 0.0, 7.6))
            y = g @ self.nutr_matrix + self.res_head(f).float()
            if self.decode == "hybrid":
                d = self.fat_fc(f).float().squeeze(-1)
                fat = (self.fat_out(d) if self.fat_head_mode in ("dedicated", "deep")
                       else torch.clamp_min(self.nutr_head(f).float()[:, 1] + d, 0.0))
                y = torch.cat([y[:, :1], fat.unsqueeze(1), y[:, 2:]], dim=1)
            return y, aux
        return self.nutr_head(f), aux

    # ---- 동결
    def apply_freeze(self, mode):
        self._frozen = []
        if mode == "none":
            return
        for i in FREEZE_SETS[mode]:
            self._frozen.append(self.backbone.features[i])
        if mode == "head":
            self._frozen.append(self.backbone.classifier)
        for mod in self._frozen:
            for p in mod.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        for m in self._frozen:      # 동결 모듈은 항상 eval (StochasticDepth off)
            m.eval()
        return self


def init_from_champion(model, ckpt, verbose=True):
    """챔피언 state_dict 를 새 구조에 옮긴다.

    - shape 이 같은 텐서는 그대로 복사
    - 헤드 1층 Linear(feat_dim,512) 는 앞쪽 768열에 챔피언 가중치, 나머지 열은 0
      (=> 학습 시작 시점 출력이 챔피언과 동일)
    - sp_mlp 는 챔피언에 없으므로 랜덤 초기화 유지
    """
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state"]
    own = model.state_dict()
    copied, expanded, skipped = 0, [], []
    for k, v in sd.items():
        if k not in own:
            skipped.append(k); continue
        if own[k].shape == v.shape:
            own[k].copy_(v); copied += 1
        elif own[k].dim() == 2 and v.dim() == 2 and own[k].shape[0] == v.shape[0] \
                and own[k].shape[1] > v.shape[1]:
            own[k].zero_()
            own[k][:, :v.shape[1]].copy_(v)
            expanded.append(f"{k} {tuple(v.shape)}->{tuple(own[k].shape)}")
        else:
            skipped.append(f"{k} {tuple(v.shape)} vs {tuple(own[k].shape)}")
    model.load_state_dict(own)
    if verbose:
        print(f"init from {ckpt}", flush=True)
        print(f"  copied={copied} expanded={len(expanded)} skipped={len(skipped)}", flush=True)
        for e in expanded:
            print(f"  expand(zero-pad): {e}", flush=True)
        for s in skipped:
            print(f"  SKIP: {s}", flush=True)


# ----------------------------------------------------------------------- train/eval
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); P, T = [], []
    for x, y, _, _, s in loader:
        with torch.autocast("cuda"):
            nutr, _ = model(x.to(device), s.to(device))
        P.append(nutr.float().cpu().numpy()); T.append(y.numpy())
    P = np.concatenate(P); T = np.concatenate(T)
    return {**pma(P, T), **_nrmse(P, T)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--pool", choices=["avg", "max", "avgmax", "attn"], default="avg")
    ap.add_argument("--freeze", choices=list(FREEZE_SETS), default="none")
    ap.add_argument("--spatial", choices=["none", "bbox", "full", "senior5"], default="none")
    ap.add_argument("--crop-box", choices=["mixed", "fixed"], default="mixed")
    # 어텐션 점수층 전용 학습률 배수. 2026-08-18 어텐션 arm 이 균등에서 0.1%도 안 움직인 것이
    # "0 초기화 + lr 2e-5 라 움직일 동기가 약해서"였다는 대안 설명을 배제하기 위한 축.
    ap.add_argument("--attn-lr-mult", type=float, default=1.0)
    # 2026-08-21: 에폭마다 test 도 함께 재서 기록만 한다(체크포인트 선택에는 절대 쓰지 않는다).
    # 목적 = "val 로 에폭을 고르면 test 가 좋아지는가"를 직접 재기 위함. val 신뢰도 검증 전용.
    ap.add_argument("--track-test", action="store_true")
    ap.add_argument("--sp-embed", type=int, default=64)
    ap.add_argument("--arch", choices=["convnext_small", "convnext_base"],
                    default="convnext_small")
    ap.add_argument("--resolution", type=int, default=384)
    ap.add_argument("--lam", type=float, default=30.0)
    # 조성 축 (2026-09-10). 기본값은 현행과 동일하다.
    ap.add_argument("--vocab-k", type=int, default=VOCAB_K)   # 학습셋 재료는 총 196종
    ap.add_argument("--decode", choices=["head", "grams", "hybrid"], default="head")
    # 지방 전용 경로의 초기화 방식. dedicated = 명세 그대로 새 헤드(랜덤 출발),
    # resid = 챔피언 지방 출력 + 0초기화 보정(출발 시점이 챔피언과 동일 → 순효과만 측정)
    ap.add_argument("--fat-head", choices=["dedicated", "resid", "deep"], default="dedicated")
    # 손실. smoothl1 = 챔피언과 100% 동일(기본). mae_norm = 영양소별 train 평균으로 나눈 뒤 MAE
    # (원 단위 SmoothL1 은 선형구간에서 네 영양소에 같은 기울기를 주므로, 평균 12.8g 인 지방이
    #  평균 255kcal 인 칼로리보다 상대적으로 과소가중된다. 그 축을 따로 열어 둔다.)
    #  mae_norm     명세 그대로. 영양소별 train 평균으로 나눈 MAE. 손실의 **절대 크기도** 100배
    #               가까이 줄어드므로 lam=30 인 보조(그램) 손실의 상대 비중이 같이 바뀐다.
    #  mae_norm_bal 같은 재가중이지만 sc.mean() 을 되곱해 **전체 크기를 원 단위에 맞춘다**
    #               → 바뀌는 변수가 '네 영양소 사이의 상대 가중' 하나뿐이 된다(영양소↔보조 비중 유지).
    ap.add_argument("--loss", choices=["smoothl1", "mae_norm", "mae_norm_bal"],
                    default="smoothl1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    # 학습률 스케줄. none = 챔피언과 100% 동일(스케줄러 객체를 아예 만들지 않는다).
    #   cosine  : CosineAnnealingLR(T_max=epochs, eta_min=lr*eta-min-mult), 에폭마다 step
    #   plateau : ReduceLROnPlateau(factor, patience=sched-patience), 검증 monitor 로 step
    ap.add_argument("--sched", choices=["none", "cosine", "plateau"], default="none")
    ap.add_argument("--sched-factor", type=float, default=0.5)
    ap.add_argument("--sched-patience", type=int, default=5)
    ap.add_argument("--eta-min-mult", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--monitor", choices=["pma", "rmse"], default="rmse")
    ap.add_argument("--init-from", default="")
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda")
    # 크롭 박스 모드는 DataLoader worker 를 fork 하기 전에 세팅해야 워커에도 전달된다.
    if args.crop_box == "fixed":
        TGC.FIXED_BOX = TGC.FALLBACK_BOX
        n_diff = sum(1 for d, b in TGC.BBOX.items() if b and tuple(b) != TGC.FALLBACK_BOX)
        print(f"crop_box=fixed -> 전 접시 {TGC.FALLBACK_BOX} 사용 "
              f"(현행 대비 실제로 바뀌는 접시 {n_diff}/{len(TGC.BBOX)})", flush=True)
    print(f"manifest_dir={MANIFEST_DIR}  track_test={args.track_test}", flush=True)
    print(f"b8 run={args.run_name} pool={args.pool} freeze={args.freeze} "
          f"spatial={args.spatial} crop_box={args.crop_box} seed={args.seed} lr={args.lr}",
          flush=True)

    grams = parse_grams()
    tr_d, tr_t = usable("train"); va_d, va_t = usable("val"); te_d, te_t = usable("test")
    df = Counter()
    for d in tr_d:
        for n in grams.get(d, {}):
            df[n] += 1
    vocab = [n for n, _ in df.most_common(args.vocab_k)]
    G = np.array([[np.log1p(grams.get(d, {}).get(n, 0.0)) for n in vocab] for d in tr_d], np.float64)
    gmu, gsd = G.mean(0), G.std(0) + 1e-6
    print(f"dishes train {len(tr_d)} val {len(va_d)} test {len(te_d)} | vocab {len(vocab)}", flush=True)

    # 공간 벡터: train 분포로 표준화 (val/test 누수 없음)
    spat_raw, sp_idx, sp_dim = load_spatial(args.spatial)
    smu = ssd = None
    if sp_dim:
        A = np.stack([spat_raw[d] for d in tr_d])[:, sp_idx]
        smu, ssd = A.mean(0), A.std(0) + 1e-6
        print(f"spatial={args.spatial} dim={sp_dim} cols={[SPNAMES13[i] for i in sp_idx]}",
              flush=True)
        print(f"  train std={np.round(A.std(0), 5).tolist()}", flush=True)

    mk = lambda dd, tt, tr: SpatialCropDS(
        dd, tt, grams, vocab, gmu, gsd, res=args.resolution, train=tr,
        spat_raw=spat_raw, sp_idx=(sp_idx if sp_dim else None), smu=smu, ssd=ssd)
    tr = DataLoader(mk(tr_d, tr_t, True), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va = DataLoader(mk(va_d, va_t, False), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    te = DataLoader(mk(te_d, te_t, False), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    model = B8Net(pool=args.pool, spatial_dim=sp_dim, sp_embed=args.sp_embed,
                  vocab_k=len(vocab), pretrained=True, arch=args.arch, decode=args.decode,
                  fat_head=args.fat_head)
    if args.decode in ("grams", "hybrid"):
        M = gram_nutrition(vocab, tr_d)          # 학습 접시에서만 계산
        model.nutr_matrix.copy_(torch.from_numpy(M))
        model.g_mu.copy_(torch.from_numpy(gmu.astype(np.float32)))
        model.g_sd.copy_(torch.from_numpy(gsd.astype(np.float32)))
        cov = float((M.sum(1) > 0).mean())
        print(f"decode={args.decode} fat_head={args.fat_head} | 영양표 {M.shape} · 값이 있는 재료 {cov*100:.0f}% "
              f"| 1g당 칼로리 중앙값 {np.median(M[M[:,0]>0,0]):.3f}", flush=True)
    model = model.to(device)
    if args.init_from:
        init_from_champion(model, args.init_from)
    model.apply_freeze(args.freeze)
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params total {tot:,} trainable {trn:,} ({100*trn/tot:.2f}%) "
          f"frozen {tot-trn:,} ({100*(tot-trn)/tot:.2f}%)", flush=True)

    crit = nn.SmoothL1Loss()
    if args.loss in ("mae_norm", "mae_norm_bal"):
        sc = torch.tensor(np.array([tr_t[d] for d in tr_d], np.float32).mean(0),
                          device=device).clamp_min(1e-6)
        # bal = 스케일의 평균을 되곱한다. 네 영양소의 **상대** 가중만 바뀌고 손실의 절대 크기는
        # 원 단위 SmoothL1 과 같은 자리에 머무르므로, lam=30 보조손실과의 비중이 유지된다.
        k = float(sc.mean().item()) if args.loss == "mae_norm_bal" else 1.0
        print(f"loss={args.loss} | 영양소 스케일(train 평균) "
              f"{np.round(sc.cpu().numpy(),1).tolist()} | 되곱 계수 {k:.2f}", flush=True)
        crit_nutr = lambda p, y: k * ((p - y).abs() / sc).mean()
    else:
        crit_nutr = crit
    if args.attn_lr_mult != 1.0 and args.pool == "attn":
        att_ids = {id(p) for p in model.attn.parameters()}
        att = [p for p in model.attn.parameters() if p.requires_grad]
        rest = [p for p in model.parameters() if p.requires_grad and id(p) not in att_ids]
        opt = torch.optim.AdamW(
            [{"params": rest, "lr": args.lr},
             {"params": att, "lr": args.lr * args.attn_lr_mult}], weight_decay=1e-4)
        print(f"attn 점수층 전용 lr = {args.lr * args.attn_lr_mult:.2e} "
              f"({args.attn_lr_mult:g}x, {sum(p.numel() for p in att):,} params) / "
              f"나머지 {args.lr:.2e}", flush=True)
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    sched = None
    if args.sched == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr * args.eta_min_mult)
        print(f"sched=cosine | T_max={args.epochs} lr {args.lr:.2e} -> "
              f"{args.lr*args.eta_min_mult:.2e}", flush=True)
    elif args.sched == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=args.sched_factor, patience=args.sched_patience)
        print(f"sched=plateau | factor={args.sched_factor} patience={args.sched_patience} "
              f"(검증 monitor 기준)", flush=True)

    run_dir = RUNS / args.run_name
    ckp = run_dir / "checkpoints"; ckp.mkdir(parents=True, exist_ok=True)
    mkey = "mean_PMA" if args.monitor == "pma" else "mean_nRMSE"
    best = float("inf"); bad = 0; hist = []; stopped_early = False
    for ep in range(1, args.epochs + 1):
        model.train(); ln = la = 0.0
        for x, y, gy, _, s in tr:
            x, y, gy, s = x.to(device), y.to(device), gy.to(device), s.to(device)
            with torch.autocast("cuda"):
                nutr, aux = model(x, s)
                lo_n = crit_nutr(nutr, y); lo_a = crit(aux, gy)
                loss = lo_n + args.lam * lo_a
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ln += lo_n.item() * x.size(0); la += lo_a.item() * x.size(0)
        vm = evaluate(model, va, device)
        cur_lr = float(opt.param_groups[0]["lr"])
        rec = {"epoch": ep, "nutr_loss": ln/len(tr.dataset),
               "aux_loss": la/len(tr.dataset), "val": vm, "lr": cur_lr}
        tmsg = ""
        if args.track_test:
            # ★기록 전용. 아래 체크포인트 선택 분기는 vm(=val)만 본다.
            tm = evaluate(model, te, device)
            rec["test_tracked"] = tm
            tmsg = (f" || test PMA {tm['mean_PMA']:.2f}% calRMSE {tm['calories_RMSE']:.1f}")
        hist.append(rec)
        print(f"ep {ep:3d} | nutr {ln/len(tr.dataset):.2f} aux {la/len(tr.dataset):.3f} | "
              f"val PMA {vm['mean_PMA']:.2f}% nRMSE {vm['mean_nRMSE']:.2f}% "
              f"(cal {vm['calories_PMA']:.1f} calRMSE {vm['calories_RMSE']:.1f}) "
              f"lr {cur_lr:.2e}{tmsg}", flush=True)
        if sched is not None:                      # ★체크포인트 선택 분기보다 앞이든 뒤든
            if args.sched == "plateau":            #   무관하지만, 기록된 lr 은 '이 에폭에 쓴 lr'이다
                sched.step(vm[mkey])
            else:
                sched.step()
        if vm[mkey] < best:
            best = vm[mkey]; bad = 0
            torch.save({"model_state": model.state_dict(), "epoch": ep, "monitor": mkey,
                        "monitor_value": best, "val": vm,
                        "cfg": {"pool": args.pool, "freeze": args.freeze,
                                "spatial": args.spatial, "sp_dim": sp_dim,
                                "crop_box": args.crop_box, "arch": args.arch,
                                "sp_embed": args.sp_embed, "vocab_k": len(vocab),
                                "decode": args.decode,
                                "fat_head": args.fat_head}}, ckp / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ {ep}", flush=True); stopped_early = True; break

    state = torch.load(ckp / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    test_m = evaluate(model, te, device)
    save_json(run_dir / "training_history.json", hist)
    save_json(run_dir / "test_metrics.json", {
        **test_m, "best_val_mean_PMA": state.get("val", {}).get("mean_PMA"),
        "monitor": mkey, "monitor_value": best, "val": state.get("val", {}),
        "best_epoch": state["epoch"], "epochs_run": len(hist),
        "epoch_cap": args.epochs, "stopped_early": stopped_early,
        "hit_epoch_cap": (not stopped_early),
        "pool": args.pool, "freeze": args.freeze, "spatial": args.spatial,
        "crop_box": args.crop_box, "attn_lr_mult": args.attn_lr_mult,
        "manifest_dir": MANIFEST_DIR, "track_test": args.track_test,
        "arch": args.arch, "resolution": args.resolution,
        "decode": args.decode, "fat_head": args.fat_head, "loss": args.loss,
        "sched": args.sched, "sched_factor": args.sched_factor,
        "sched_patience": args.sched_patience, "eta_min_mult": args.eta_min_mult,
        "patience": args.patience,
        "lr_last": float(opt.param_groups[0]["lr"]),
        "lr_at_best": next((h.get("lr") for h in hist if h["epoch"] == state["epoch"]), None),
        "vocab_k": len(vocab), "lam": args.lam,
        "sp_dim": sp_dim, "sp_embed": args.sp_embed, "seed": args.seed,
        "lr": args.lr, "init_from": args.init_from,
        "trainable_params": trn, "total_params": tot, "n_test_dishes": len(te_d)})
    flag = "" if stopped_early else "  [!] 학습 미완료(에폭 상한 도달)"
    print(f"DONE {args.run_name} | monitor={mkey} best_val {best:.2f}% ep{state['epoch']}/{len(hist)} | "
          f"TEST PMA {test_m['mean_PMA']:.2f}% (cal {test_m['calories_PMA']:.2f}) "
          f"nRMSE {test_m['mean_nRMSE']:.2f}% calRMSE {test_m['calories_RMSE']:.1f}{flag}", flush=True)


if __name__ == "__main__":
    main()
