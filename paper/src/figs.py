"""논문 초안 파이프라인 구조도 4장 (PNG). usage: /usr/bin/python3 figs.py <out_dir>"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
for f in ("/home/leehogwang/.fonts/NanumGothic-Regular.ttf", "/home/leehogwang/.fonts/NanumGothic-Bold.ttf"):
    fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "NanumGothic"; plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, LINE = "#1b2430", "#5a6577", "#9aa5b4"
C = {"n": ("#ffffff", "#9aa5b4"), "known": ("#e2f1ee", "#1d776a"), "unk": ("#f7ebde", "#a85b17"),
     "qty": ("#e7ebfb", "#3b56cf"), "out": ("#eef1f5", "#1b2430"), "grl": ("#f8e2e7", "#bf2f4a"), "txt": ("#f3eefa", "#6a4bb0")}


def fig(w, h):
    f, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(-0.05, w); ax.set_ylim(-0.2, h); ax.axis("off"); return f, ax


def box(ax, x, y, w, h, title, sub="", kind="n", fs=11, dashed=False):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=1.8 if kind != "n" else 1.3,
                                ls=(0, (4, 3)) if dashed else "-"))
    if sub:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=fs - 2.2, color=MUTED, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK, linespacing=1.25)


def arr(ax, pts, color=INK, dashed=False, lw=1.5):
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=(0, (5, 4)) if dashed else "-", solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=(0, (5, 4)) if dashed else "-",
                                                                 shrinkA=0, shrinkB=0, mutation_scale=14))


def label(ax, x, y, t, color=MUTED, fs=9, ha="left", bold=False):
    ax.text(x, y, t, ha=ha, va="center", fontsize=fs, color=color, fontweight="bold" if bold else "normal")


def save(f, name):
    f.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.08, facecolor="white"); plt.close(f)


# ── 그림 1. P1·P2 직접 회귀 ─────────────────────────────────────
f, ax = fig(12, 2.9)
box(ax, 0.2, 1.65, 1.7, 0.95, "사진", "RGB")
box(ax, 0.2, 0.25, 1.7, 0.95, "깊이", "N5k 실측 RealSense\n한식 추정 UniDepth·DA")
box(ax, 2.6, 0.85, 1.9, 1.15, "4채널 입력", "RGB + 깊이")
arr(ax, [(1.9, 2.12), (2.25, 2.12), (2.25, 1.6), (2.6, 1.6)]); arr(ax, [(1.9, 0.72), (2.25, 0.72), (2.25, 1.25), (2.6, 1.25)])
box(ax, 5.2, 0.75, 2.6, 1.35, "백본", "P1 ResNet101-SE\nP2 ConvNeXt-S")
arr(ax, [(4.5, 1.42), (5.2, 1.42)])
box(ax, 8.5, 0.85, 1.6, 1.15, "회귀 헤드")
arr(ax, [(7.8, 1.42), (8.5, 1.42)])
box(ax, 10.6, 0.75, 1.3, 1.35, "영양값", "칼로리·지방\n단백질·탄수", kind="out")
arr(ax, [(10.1, 1.42), (10.6, 1.42)])
label(ax, 5.2, 2.5, "음식명·1인분 개념 없이 사진에서 바로 회귀", color=MUTED, fs=9.5)
save(f, "fig1_direct.png")

# ── 그림 2. P3 라우팅 (3번 최고 구성)과 P6 변경점 ─────────────────
f, ax = fig(16.2, 7.4)
box(ax, 0.2, 3.25, 1.4, 1.0, "사진")
box(ax, 2.1, 2.95, 2.3, 1.6, "1. 음식 분류기", "ConvNeXt-Tiny top1\n+ open-set 판정\n(Mahalanobis 거리)")
arr(ax, [(1.6, 3.75), (2.1, 3.75)])
label(ax, 4.95, 6.85, "아는 음식", color=C["known"][1], fs=10.5, bold=True)
label(ax, 4.95, 4.55, "모르는 음식", color=C["unk"][1], fs=10.5, bold=True)
arr(ax, [(4.4, 4.15), (4.65, 4.15), (4.65, 6.05), (4.95, 6.05)])
arr(ax, [(4.4, 3.75), (4.95, 3.75)])
# 아는 음식
box(ax, 4.95, 5.5, 2.4, 1.1, "1-a. 음식명", "분류기 top1", "known")
box(ax, 7.85, 5.5, 2.4, 1.1, "2. 재료·대분류", "Gemini · 사진+음식명", "known")
box(ax, 10.75, 5.5, 2.7, 1.1, "4-1. 1인분 값", "영양 모델 (텍스트+이미지+주파수)", "known")
arr(ax, [(7.35, 6.05), (7.85, 6.05)]); arr(ax, [(10.25, 6.05), (10.75, 6.05)])
# 모르는 음식
box(ax, 4.95, 3.2, 2.4, 1.1, "1-b. 음식명", "Gemini · 사진", "unk")
box(ax, 7.85, 3.2, 2.4, 1.1, "2. 재료·대분류", "Gemini · 사진+음식명", "unk")
box(ax, 10.75, 3.2, 2.7, 1.1, "3. 1인분 값", "Gemini · 음식명+재료", "unk")
arr(ax, [(7.35, 3.75), (7.85, 3.75)]); arr(ax, [(10.25, 3.75), (10.75, 3.75)])
# P6 변경: 아는 음식 표 조회
ax.text(12.1, 7.0, "P6: 학습 라벨 1인분 표 조회", ha="center", va="center", fontsize=9.5, color=C["qty"][1], fontweight="bold")
ax.add_patch(FancyBboxPatch((10.65, 5.4), 2.9, 1.3, boxstyle="round,pad=0,rounding_size=0.15", fc="none", ec=C["qty"][1], lw=1.4, ls=(0, (4, 3))))
# 양 추정
box(ax, 4.95, 0.35, 8.5, 1.6, "", kind="qty")
ax.text(5.2, 1.65, "4-2. 양 추정 모델 → Q1~Q5", ha="left", va="center", fontsize=11, fontweight="bold", color=C["qty"][1])
ax.text(5.2, 1.1, "ConvNeXt-Tiny 384 특징 ⊕ 대분류 임베딩 64 → 양 헤드 → argmax", ha="left", va="center", fontsize=9.5, color=INK)
ax.text(5.2, 0.62, "P6: Faster R-CNN 음식 박스(그릇 제외)를 잘라 입력, 깊이·주파수 분기 추가 (실험 42·45)", ha="left", va="center", fontsize=9.2, color=C["qty"][1])
arr(ax, [(0.9, 3.25), (0.9, 1.15), (4.95, 1.15)])
label(ax, 1.05, 2.0, "같은 사진", fs=9)
arr(ax, [(9.05, 3.2), (9.05, 1.95)], color=C["qty"][1], dashed=True)
label(ax, 9.2, 2.6, "대분류", color=C["qty"][1], fs=9)
# 보정, 최종
box(ax, 14.1, 3.2, 1.9, 1.1, "5. 보정", "1인분 × Q 비율")
ax.plot([13.45, 13.75, 13.75], [6.05, 6.05, 3.75], color=INK, lw=1.5); arr(ax, [(13.45, 3.75), (14.1, 3.75)])
arr(ax, [(13.45, 1.15), (15.05, 1.15), (15.05, 3.2)])
ax.text(15.15, 2.2, "Q 비율\n0.25 · 0.5 · 0.75\n1.0 · 1.25", ha="left", va="center", fontsize=8.6, color=MUTED, linespacing=1.3)
box(ax, 14.1, 5.5, 1.9, 1.1, "최종 영양값", "칼·지·단·탄·중량", "out")
arr(ax, [(15.05, 4.3), (15.05, 5.5)])
save(f, "fig2_routing.png")

# ── 그림 3. P4 분해형 (25-d) ───────────────────────────────────
f, ax = fig(14, 5.4)
box(ax, 0.2, 3.0, 1.5, 1.0, "사진")
box(ax, 0.2, 0.55, 1.5, 1.0, "음식명", "정답 또는 Gemini")
box(ax, 2.4, 4.15, 2.6, 0.85, "ConvNeXt-B", "")
box(ax, 2.4, 3.1, 2.6, 0.85, "주파수 분기", "FFT → VGG16")
box(ax, 2.4, 2.05, 2.6, 0.85, "깊이 분기", "UniDepth V2 → ResNet18")
for yy in (4.57, 3.52, 2.47):
    arr(ax, [(1.7, 3.5), (2.05, 3.5), (2.05, yy), (2.4, yy)])
box(ax, 5.7, 2.9, 1.9, 1.3, "사진별 게이트", "세 분기 가중합")
for yy in (4.57, 3.52, 2.47):
    arr(ax, [(5.0, yy), (5.35, yy), (5.35, 3.55), (5.7, 3.55)])
box(ax, 8.3, 2.95, 2.2, 1.2, "배율 r", "양 (≈ 중량 비율)", "qty")
arr(ax, [(7.6, 3.55), (8.3, 3.55)])
box(ax, 2.4, 0.95, 2.6, 0.85, "음식명 임베딩 64", "", "txt", fs=10)
box(ax, 2.4, 0.0, 2.6, 0.8, "LLM 1인분 사전값", "Gemini 3.8-flash", "txt", fs=10)
arr(ax, [(1.7, 1.05), (2.05, 1.05), (2.05, 1.37), (2.4, 1.37)]); arr(ax, [(1.7, 0.85), (2.05, 0.85), (2.05, 0.4), (2.4, 0.4)])
box(ax, 5.7, 0.35, 1.9, 1.2, "보정 $\\delta$", "사전값 + $\\delta$")
arr(ax, [(5.0, 1.37), (5.35, 1.37), (5.35, 1.0), (5.7, 1.0)]); arr(ax, [(5.0, 0.4), (5.35, 0.4), (5.35, 0.85), (5.7, 0.85)])
box(ax, 8.3, 0.35, 2.2, 1.2, "1인분 s", "음식별 영양값", "txt")
arr(ax, [(7.6, 0.95), (8.3, 0.95)])
box(ax, 11.3, 1.7, 2.4, 1.4, "최종 영양값", "s × r", "out")
arr(ax, [(10.5, 3.55), (12.5, 3.55), (12.5, 3.1)]); arr(ax, [(10.5, 0.95), (12.5, 0.95), (12.5, 1.7)])
label(ax, 8.3, 4.75, "한식 정답(DB 1인분 × 양 비율)과 같은 구조", color=C["qty"][1], fs=9.5, bold=True)
save(f, "fig3_decomp.png")

# ── 그림 4. P5 통합형 (실험 42 pipe_best) ───────────────────────
f, ax = fig(14.6, 5.6)
box(ax, 0.2, 2.6, 1.7, 1.1, "사진", "RGB + z 깊이")
box(ax, 2.6, 3.9, 2.5, 0.85, "ConvNeXt-S + SE", "768")
box(ax, 2.6, 2.85, 2.5, 0.85, "주파수 믹스", "256")
box(ax, 2.6, 1.8, 2.5, 0.85, "절대 깊이 스칼라", "128")
for yy in (4.32, 3.27, 2.22):
    arr(ax, [(1.9, 3.15), (2.25, 3.15), (2.25, yy), (2.6, yy)])
box(ax, 5.8, 2.5, 1.8, 1.4, "특징 게이트", "1152")
for yy in (4.32, 3.27, 2.22):
    arr(ax, [(5.1, yy), (5.45, yy), (5.45, 3.2), (5.8, 3.2)])
box(ax, 8.4, 4.2, 2.5, 0.9, "직접 회귀 헤드", "→ 영양값 4개", "known")
box(ax, 8.4, 2.95, 2.5, 0.9, "log 중량 헤드", "→ 중량 (g)", "unk")
box(ax, 8.4, 1.7, 2.5, 0.9, "100 g당 밀도", "사전값 + $\\delta$", "unk")
for yy in (4.65, 3.4, 2.15):
    arr(ax, [(7.6, 3.2), (8.0, 3.2), (8.0, yy), (8.4, yy)])
box(ax, 11.6, 2.3, 2.3, 1.0, "분해", "중량/100 × 밀도", "unk")
arr(ax, [(10.9, 3.4), (11.25, 3.4), (11.25, 2.95), (11.6, 2.95)]); arr(ax, [(10.9, 2.15), (11.25, 2.15), (11.25, 2.6), (11.6, 2.6)])
box(ax, 11.6, 4.1, 2.3, 1.1, "최종 영양값", "$\\alpha$·직접 + (1-$\\alpha$)·분해", "out", fs=10.5)
arr(ax, [(10.9, 4.65), (11.6, 4.65)]); arr(ax, [(12.75, 3.3), (12.75, 4.1)])
box(ax, 5.8, 0.3, 1.8, 0.9, "재료 텍스트 64", "", "txt", fs=10)
box(ax, 8.4, 0.3, 2.5, 0.9, "게이트 $\\alpha$", "[특징, 텍스트 64]", "grl")
arr(ax, [(7.6, 0.75), (8.4, 0.75)]); arr(ax, [(6.7, 2.5), (6.7, 1.75), (8.15, 1.75), (8.15, 1.05), (8.4, 1.05)], color=LINE)
arr(ax, [(10.9, 0.75), (14.25, 0.75), (14.25, 4.65), (13.9, 4.65)], color=C["grl"][1])
ax.text(14.35, 2.6, "$\\alpha$", ha="left", va="center", fontsize=11, color=C["grl"][1], fontweight="bold")
label(ax, 8.4, 5.45, "N5k에서 학습된 $\\alpha$ ≈ 0.90 (직접 회귀 비중)", color=C["grl"][1], fs=9.5, bold=True)
save(f, "fig4_fused.png")
print("done", sorted(p.name for p in OUT.glob("*.png")))
