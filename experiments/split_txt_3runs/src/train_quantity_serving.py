"""양추정에 Gemini 1인분 중량을 기준점으로 주는 실험 (122_2_forunseen). 백본·학습 설정은 train_quantity.py 와 같고 대분류는 쓰지 않는다.
  --mode in  : ConvNeXt 특징 ⊕ Linear(표준화 log 1인분 중량 → 16) → Q1~Q5 CE (label smoothing 0.1)
               학습 1인분 중량 = Gemini 1인분(정답 음식명, gemini_serving.jsonl) — test 와 같은 출처. Q 정답은 DB 1인분 기준 그대로.
  --mode reg : 사진 → log 실제 중량 회귀 (smooth L1). 학습에 1인분 값을 쓰지 않는다.
               Q = argmin_k |예측 log 중량 - log 1인분 중량 - log 비율_k|   (1인분 중량은 추론 때만 사용)
1인분 중량 출처 (test): pipe  = 파이프라인과 같음. Gemini 로 넘긴 사진 → gemini_serving_ing (음식명 + 재료), 분류기 통과 → gemini_serving (음식명)
                        db    = DB 정답 1인분 중량 (상한 확인용)
val 의 1인분 중량은 정답 음식명의 Gemini 값. 체크포인트 선택은 val 정확도 (pipe 와 같은 Gemini 출처).
출력 runs/122_2_forunseen_quantity_serv_<mode>/{best.pth, history.csv, report.json, test_pred.npz, train.log}
usage: python train_quantity_serving.py --mode in --device cuda:0"""
import argparse, csv, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts_ing  # noqa: E402
from train_quantity import EXP, MODEL_NAME, eval_transform, train_transform  # noqa: E402
from n296.common import seed_everything  # noqa: E402
from n296.food_classifier import fit_temperature  # noqa: E402

SPLIT = "122_2_forunseen"
RATIO = np.array([0.25, 0.5, 0.75, 1.0, 1.25]); LOGR = torch.tensor(np.log(RATIO), dtype=torch.float32)


class DS(Dataset):
    def __init__(s, recs, ids, tf, serv):
        s.r = [recs[i] for i in ids]; s.tf = tf; s.s = [float(np.log(serv[i])) for i in ids]
    def __len__(s): return len(s.r)
    def __getitem__(s, i):
        r = s.r[i]
        return s.tf(Image.open(r["img"]).convert("RGB")), torch.tensor(s.s[i]), r["q"], torch.tensor(float(np.log(r["y"][4])))


class Net(nn.Module):
    def __init__(s, mode, mu, sd, pretrained=True):
        super().__init__(); import timm
        s.mode, s.mu, s.sd = mode, mu, sd
        s.backbone = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0, drop_path_rate=0.1); d = s.backbone.num_features
        if mode == "in":
            s.semb = nn.Linear(1, 16); d += 16
        s.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 5 if mode == "in" else 1))
    def forward(s, x, logs):
        f = s.backbone(x)
        if s.mode == "in": f = torch.cat([f, s.semb(((logs - s.mu) / s.sd).unsqueeze(1).to(f.dtype))], 1)
        o = s.head(f)
        return o if s.mode == "in" else o[:, 0] * s.sd + s.mu   # reg: log 중량 (1인분 분포 기준으로 스케일)


def decode(mode, out, logs):
    """→ (Q 예측 [N], 로짓/점수 [N,5])"""
    if mode == "in": return out.argmax(1), out
    sc = -(out[:, None] - logs[:, None] - LOGR[None]).abs(); return sc.argmax(1), sc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["in", "reg"], required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=48); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=384); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); seed_everything(42); dev = torch.device(a.device)
    recs = json.loads((EXP / "records.json").read_text()); sp = json.loads((EXP / "splits.json").read_text())[SPLIT]
    if a.smoke: sp = {k: v[:96] for k, v in sp.items()}; a.epochs = 1
    rd = EXP / ("runs_smoke" if a.smoke else "runs") / f"{SPLIT}_quantity_serv_{a.mode}"; rd.mkdir(parents=True, exist_ok=True)
    log = open(rd / "train.log", "a")
    def P(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()

    G = {j["food"]: j["mass"] for j in map(json.loads, open(EXP / "gemini_serving.jsonl"))}
    GI = {j["food"]: j["mass"] for j in map(json.loads, open(EXP / "gemini_serving_ing.jsonl"))}
    ING = texts_ing.ing_records(); pn = texts_ing.pipe_names(SPLIT, osr=True)
    s_gt = {i: G[recs[i]["food"]] for i in sp["train"] + sp["val"]}                     # 학습·val: 정답 음식명의 Gemini 1인분
    s_pipe = {}
    for i in sp["test"]:
        n, src = pn[i]
        s_pipe[i] = GI[texts_ing.render(n, ING[(i, n)]["ingredients"])] if src == "gemini" else G[n]
    s_db = {i: recs[i]["y"][4] / RATIO[recs[i]["q"]] for i in sp["test"]}
    ls = np.log([s_gt[i] for i in sp["train"]]); mu, sd = float(ls.mean()), max(float(ls.std()), 0.1)

    mk = lambda ids, tf, sh, serv: DataLoader(DS(recs, ids, tf, serv), batch_size=a.batch_size, shuffle=sh, num_workers=a.workers,
                                              pin_memory=True, drop_last=sh, persistent_workers=sh)
    trL = mk(sp["train"], train_transform(a.img_size), True, s_gt); vaL = mk(sp["val"], eval_transform(a.img_size), False, s_gt)
    model = Net(a.mode, mu, sd).to(dev)
    new = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": a.lr}, {"params": new, "lr": a.lr * 10}], weight_decay=0.05)
    steps, warm = a.epochs * len(trL), len(trL)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda"); ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    crit = (lambda o, q, lm: ce(o.float(), q)) if a.mode == "in" else (lambda o, q, lm: F.smooth_l1_loss(o.float(), lm, beta=0.1))

    @torch.no_grad()
    def run(L):
        model.eval(); O, S, Y, M = [], [], [], []
        for x, s, q, lm in L:
            with torch.autocast("cuda", dtype=torch.float16): O.append(model(x.to(dev, non_blocking=True), s.to(dev)).float().cpu())
            S.append(s); Y.append(q); M.append(lm)
        return torch.cat(O), torch.cat(S), torch.cat(Y), torch.cat(M)

    P(f"[{SPLIT}_quantity_serv_{a.mode}] train {len(sp['train'])} val {len(sp['val'])} test {len(sp['test'])} | log 1인분 평균 {mu:.3f} 표준편차 {sd:.3f} | {a.device}")
    hist = open(rd / "history.csv", "w", newline=""); hw = csv.writer(hist); hw.writerow(["epoch", "sec", "train_loss", "val_acc", "val_within1"])
    best, bad, bep, t0 = -1.0, 0, 0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); te = time.time(); tot = n = 0
        for x, s, q, lm in trL:
            x, s, q, lm = x.to(dev, non_blocking=True), s.to(dev), q.to(dev), lm.to(dev)
            with torch.autocast("cuda", dtype=torch.float16): loss = crit(model(x, s), q, lm)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item() * len(q); n += len(q)
        o, s, y, _ = run(vaL); pr, _ = decode(a.mode, o, s)
        acc = float((pr == y).float().mean()); w1 = float(((pr - y).abs() <= 1).float().mean())
        hw.writerow([ep, round(time.time() - te), round(tot / n, 5), round(acc, 5), round(w1, 5)]); hist.flush()
        P(f"ep {ep:2d} ({time.time() - te:.0f}s) loss {tot / n:.4f} | val acc {acc:.4f} ±1 {w1:.4f}")
        if acc > best:
            best, bad, bep = acc, 0, ep; torch.save({"state_dict": model.state_dict(), "epoch": ep}, rd / "best.pth")
        else:
            bad += 1
            if bad >= a.patience: P(f"[EarlyStop] ep {ep}, best ep {bep}"); break
    ck = torch.load(rd / "best.pth", map_location="cpu"); model.load_state_dict(ck["state_dict"])
    T = None
    if a.mode == "in":
        vo, _, vy, _ = run(vaL); T = float(fit_temperature(vo, vy))
    torch.save({"mode": a.mode, "mu": mu, "sd": sd, "img_size": a.img_size, "state_dict": ck["state_dict"], "temperature": T, "epoch": bep}, rd / "best.pth")
    rep = {"mode": a.mode, "best_epoch": bep, "epochs_run": ep, "temperature": T, "minutes": round((time.time() - t0) / 60, 1)}
    out = {"id": np.array(sp["test"])}
    for name, serv in (("pipe", s_pipe), ("db", s_db)):
        o, s, y, lm = run(mk(sp["test"], eval_transform(a.img_size), False, serv)); pr, sc = decode(a.mode, o, s)
        out[f"q_{name}"], out[f"score_{name}"], out["label"] = pr.numpy(), sc.numpy(), y.numpy()
        if a.mode == "reg": out["log_mass"] = o.numpy(); out["log_mass_true"] = lm.numpy()
        err = (pr - y).abs()
        rep[f"test_{name}"] = {"acc": float((err == 0).float().mean()), "within1": float((err <= 1).float().mean()),
                               "mid_acc": float((err == 0)[(y >= 1) & (y <= 3)].float().mean())}
        P(f"[TEST 1인분 {name}] acc {rep[f'test_{name}']['acc']:.4f} ±1 {rep[f'test_{name}']['within1']:.4f} Q2~4 {rep[f'test_{name}']['mid_acc']:.4f}")
    np.savez(rd / "test_pred.npz", **out); (rd / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2)); P("DONE")


if __name__ == "__main__":
    main()
