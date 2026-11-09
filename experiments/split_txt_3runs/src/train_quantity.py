"""n296_stage3_3 양추정 모델(QuantityNet: ConvNeXt-Tiny 384 + 음식 대분류 임베딩 64 → Q1~Q5) 을 food_split_txt 분할로 학습.
학습 설정은 n296/quantity_estimator.train 과 같다: AdamW lr 1e-4 (헤드·임베딩 ×10), wd 0.05, 1 epoch warmup + cosine, batch 48,
label smoothing 0.1, drop_path 0.1, 384px (Resize + flip + affine + ColorJitter), cat-dropout 0.2, fp16 AMP, 검증 정확도 최고 체크포인트,
patience 6, 최대 30 epoch, 끝나고 val 로 temperature 보정. 대분류 = food_category.FOOD_CATEGORY (사람이 정한 정답표, 없으면 기타).
test 는 대분류 3조건으로 예측: gt(정답표) / gemini(Gemini 추정 음식명을 정답표로 매핑, 없으면 기타) / masked(모두 기타).
출력 EXP/runs/<split>_quantity/{best.pth, history.csv, report.json, test_logits.npz, train.log}
usage: python train_quantity.py --split 122_1_forseen --device cuda:0
음식 정체성 제거 (처음 보는 음식 일반화): --grl L → 백본 특징에 gradient reversal + 음식 분류 헤드(train 음식). 역전 세기 = L × (2/(1+exp(-10p)) - 1), p = 학습 진행률.
  음식 헤드는 학습에만 쓰고 best.pth 에는 QuantityNet 만 저장 (추론 구조 같음).
--aug strong → 색·질감 증강 강화 (RandomGrayscale 0.5, ColorJitter 0.4/0.4/0.4/0.1, GaussianBlur 0.3). 크기·위치 증강은 기존과 같음 (양 단서 보존).
soft label (거리 기반, ordinal): --soft_sigma S --tag _softS → 목표 분포 ∝ exp(-(k-q)²/(2S²)), 출력 runs/<split>_quantity<tag>/. 기본(0)은 CE + label smoothing 0.1"""
import argparse, csv, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

EXP = Path(__file__).resolve().parents[1]
REPO = Path("/home/leehogwang/food_b")
sys.path.insert(0, str(REPO))
from n296.common import seed_everything  # noqa: E402
from n296.food_category import CATEGORY_NAMES, FOOD_CATEGORY, UNKNOWN  # noqa: E402
from n296.food_classifier import fit_temperature  # noqa: E402
from n296.quantity_estimator import (MODEL_NAME, Q_LEVELS, Q_RATIO, QuantityNet, eval_transform, quantity_report,  # noqa: E402
                                     train_transform)

UNK = CATEGORY_NAMES.index(UNKNOWN)


class _GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None


def strong_train_transform(size):
    from n296.quantity_estimator import MEAN, STD
    return transforms.Compose([
        transforms.Resize((size, size)), transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.RandomGrayscale(0.5), transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomApply([transforms.GaussianBlur(9, (0.5, 2.0))], p=0.3),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


FOODS = {}  # train 음식명 → 번호 (GRL 음식 헤드용)


def category(r):
    """정답 대분류: 정규화 음식명 → 원래 음식명 순서로 정답표에서 찾고, 없으면 기타."""
    return FOOD_CATEGORY.get(r["food"]) or FOOD_CATEGORY.get(r["raw_food"]) or UNKNOWN


class DS(Dataset):
    def __init__(s, recs, ids, tf, cats, cat_dropout=0.0):
        s.r = [recs[i] for i in ids]; s.tf, s.cd = tf, cat_dropout; s.c = [CATEGORY_NAMES.index(cats[i]) for i in ids]
        s.f = [FOODS.get(r["food"], -1) for r in s.r]
    def __len__(s): return len(s.r)
    def __getitem__(s, i):
        c = s.c[i]
        if s.cd and torch.rand(1).item() < s.cd: c = UNK
        return s.tf(Image.open(s.r[i]["img"]).convert("RGB")), c, s.r[i]["q"], s.f[i]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=48); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=384); ap.add_argument("--cat_dropout", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--soft_sigma", type=float, default=0.0); ap.add_argument("--tag", default="")
    ap.add_argument("--grl", type=float, default=0.0); ap.add_argument("--aug", choices=["base", "strong"], default="base")
    a = ap.parse_args(); seed_everything(42); dev = torch.device(a.device)
    recs = json.loads((EXP / "records.json").read_text()); sp = json.loads((EXP / "splits.json").read_text())[a.split]
    if a.smoke: sp = {k: v[:96] for k, v in sp.items()}; a.epochs = 1
    rd = EXP / ("runs_smoke" if a.smoke else "runs") / f"{a.split}_quantity{a.tag}"; rd.mkdir(parents=True, exist_ok=True)
    log = open(rd / "train.log", "a")
    def P(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
    FOODS.update({f: k for k, f in enumerate(sorted({recs[i]["food"] for i in sp["train"]}))})
    gt_cat = {r["id"]: category(r) for r in recs if r["ds"] == "122"}
    gem = {}
    for l in open(EXP / "gemini_pred.jsonl"):
        j = json.loads(l)
        if "food" in j: gem[j["id"]] = FOOD_CATEGORY.get(j["food"].replace(" ", ""), FOOD_CATEGORY.get(j["food"], UNKNOWN))
    mask = {i: UNKNOWN for i in gt_cat}
    mk = lambda ids, tf, sh, cats, cd=0.0: DataLoader(DS(recs, ids, tf, cats, cd), batch_size=a.batch_size, shuffle=sh, num_workers=a.workers,
                                                     pin_memory=True, drop_last=sh, persistent_workers=sh)
    trL = mk(sp["train"], (strong_train_transform if a.aug == "strong" else train_transform)(a.img_size), True, gt_cat, a.cat_dropout)
    vaL = mk(sp["val"], eval_transform(a.img_size), False, gt_cat)
    model = QuantityNet(MODEL_NAME, len(Q_LEVELS), len(CATEGORY_NAMES), pretrained=True, drop_path_rate=0.1).to(dev)
    new = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    food_head = None
    if a.grl > 0:
        d = model.backbone.num_features
        food_head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, len(FOODS))).to(dev); new += list(food_head.parameters())
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": a.lr}, {"params": new, "lr": a.lr * 10}], weight_decay=0.05)
    steps, warm = a.epochs * len(trL), len(trL)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda"); crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    if a.soft_sigma > 0:
        lv = torch.arange(len(Q_LEVELS), dtype=torch.float32, device=dev)
        SOFT = torch.softmax(-(lv[None] - lv[:, None]) ** 2 / (2 * a.soft_sigma ** 2), dim=1)  # [정답 q, k]
        crit = lambda lg, y: -(SOFT[y] * F.log_softmax(lg.float(), 1)).sum(1).mean()

    @torch.no_grad()
    def logits_of(L):
        model.eval(); o, y = [], []
        for x, c, t, _ in L:
            with torch.autocast("cuda", dtype=torch.float16): o.append(model(x.to(dev, non_blocking=True), c.to(dev)).float().cpu())
            y.append(t)
        return torch.cat(o), torch.cat(y)

    cats_tr = sorted({gt_cat[i] for i in sp["train"]})
    P(f"[{a.split}_quantity{a.tag}] soft_sigma {a.soft_sigma} grl {a.grl} aug {a.aug} 음식 {len(FOODS)} | train {len(sp['train'])} val {len(sp['val'])} test {len(sp['test'])} | 대분류 {len(cats_tr)}개 학습 | {a.device}")
    hist = open(rd / "history.csv", "w", newline=""); hw = csv.writer(hist); hw.writerow(["epoch", "sec", "train_loss", "val_acc", "val_within1"])
    best, bad, bep, t0 = -1.0, 0, 0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); te = time.time(); tot = n = 0; ftot = fok = 0.0
        if food_head is not None: food_head.train()
        for it, (x, c, y, fy) in enumerate(trL):
            x, c, y, fy = x.to(dev, non_blocking=True), c.to(dev), y.to(dev, non_blocking=True), fy.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                if food_head is None: loss = crit(model(x, c), y)
                else:
                    f = model.backbone(x); h = torch.cat([f, model.cat_emb(c)], 1) if model.cat_emb is not None else f
                    pr_ = ((ep - 1) * len(trL) + it) / steps; lam = a.grl * (2 / (1 + math.exp(-10 * pr_)) - 1)
                    fl = food_head(_GradRev.apply(f, lam)); lf = F.cross_entropy(fl.float(), fy)
                    loss = crit(model.head(h), y) + lf; ftot += lf.item() * len(y); fok += (fl.argmax(1) == fy).sum().item()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item() * len(y); n += len(y)
        lg, lb = logits_of(vaL); pr = lg.argmax(1)
        acc = float((pr == lb).float().mean()); w1 = float(((pr - lb).abs() <= 1).float().mean())
        hw.writerow([ep, round(time.time() - te), round(tot / n, 5), round(acc, 5), round(w1, 5)]); hist.flush()
        P(f"ep {ep:2d} ({time.time() - te:.0f}s) loss {tot / n:.4f} | val acc {acc:.4f} ±1 {w1:.4f}"
          + (f" | 음식헤드 loss {ftot / n:.3f} acc {fok / n:.3f}" if food_head is not None else ""))
        if acc > best:
            best, bad, bep = acc, 0, ep; torch.save({"state_dict": model.state_dict(), "epoch": ep}, rd / "best.pth")
        else:
            bad += 1
            if bad >= a.patience: P(f"[EarlyStop] ep {ep}, best ep {bep}"); break
    ck = torch.load(rd / "best.pth", map_location="cpu"); model.load_state_dict(ck["state_dict"])
    vl, vy = logits_of(vaL); T = fit_temperature(vl, vy)
    torch.save({"model_name": MODEL_NAME, "q_levels": Q_LEVELS, "q_ratio": Q_RATIO, "img_size": a.img_size, "categories": CATEGORY_NAMES,
                "cat_dim": 64, "state_dict": ck["state_dict"], "temperature": T, "epoch": bep}, rd / "best.pth")
    rep = {"split": a.split, "best_epoch": bep, "epochs_run": ep, "temperature": T, "minutes": round((time.time() - t0) / 60, 1),
           "val": quantity_report(vl, vy, T)}
    out = {"id": np.array(sp["test"])}
    for name, cats in (("gt", gt_cat), ("gemini", {**mask, **gem}), ("masked", mask)):
        lg, lb = logits_of(mk(sp["test"], eval_transform(a.img_size), False, cats))
        rep[f"test_cat_{name}"] = quantity_report(lg, lb, T); out[f"logits_{name}"] = lg.numpy(); out["label"] = lb.numpy()
        r_ = rep[f"test_cat_{name}"]
        P(f"[TEST 대분류 {name}] acc {r_['acc']:.4f} ±1 {r_['acc_within1']:.4f} ratio MAE {r_['ratio_mae_argmax']:.4f}")
    rep["test_gemini_category_known"] = float(np.mean([gem.get(i, UNKNOWN) != UNKNOWN for i in sp["test"]]))
    np.savez(rd / "test_logits.npz", **out)
    (rd / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2)); P("DONE")


if __name__ == "__main__":
    main()
