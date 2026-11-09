"""분할별 음식 분류기 (파이프라인 1단계). n296.food_classifier 와 같은 모델·학습 설정:
timm convnext_tiny.fb_in22k_ft_in1k, 288px, AdamW lr 1e-4 wd 0.05, warmup 1 epoch + cosine, label smoothing 0.1, drop_path 0.1,
fp16 AMP, 검증 top1 최고 체크포인트, patience 6, 최대 30 epoch, 끝나고 val 로 temperature 보정.
클래스 = 그 분할의 train 음식. 데이터 = 122 양추정 사진 (records.json).
분류기 val = 분할 val 중 train 음식인 사진. 1,000장보다 적으면(122_2_forunseen) train 의 10% 접시(시점·번호 묶음)를 떼어 val 로 쓴다.
test 는 분할 test 전체(처음 보는 음식 포함)에 top5 + 보정 확신도를 저장한다. 처음 보는 음식은 정답 클래스가 없어 정확도에서 빠진다.
출력 EXP/runs/cls_<split>/{best.pth, report.json, test_pred.json, train.log}
usage: python train_classifier.py --split 122_2_forunseen --device cuda:0"""
import argparse, json, math, random, re, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/home/leehogwang/food_b")
from n296.food_classifier import MODEL_NAME, calibration_report, eval_transform, fit_temperature, train_transform  # noqa: E402


def group(r):  # 같은 접시(시점만 다름)는 같은 묶음: side_..._Q1_00003 ↔ top_..._Q1_00003
    return re.sub(r"^(side|top)_", "", r["name"]).rsplit(".", 1)[0]


class DS(Dataset):
    def __init__(s, recs, ids, cidx, tf): s.r, s.c, s.tf = [recs[i] for i in ids], cidx, tf
    def __len__(s): return len(s.r)
    def __getitem__(s, i):
        r = s.r[i]; im = Image.open(r["img"]).convert("RGB")
        return s.tf(im), s.c.get(r["food"], -1), r["id"]


@torch.no_grad()
def logits_of(model, loader, dev):
    model.eval(); L, Y, I = [], [], []
    for x, y, i in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            L.append(model(x.to(dev, non_blocking=True)).float().cpu())
        Y.append(y); I += i.tolist()
    return torch.cat(L), torch.cat(Y), I


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=6); ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--img_size", type=int, default=288); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42); random.seed(42); dev = torch.device(a.device)
    recs = json.loads((EXP / "records.json").read_text()); sp = json.loads((EXP / "splits.json").read_text())[a.split]
    classes = sorted({recs[i]["food"] for i in sp["train"]}); cidx = {c: k for k, c in enumerate(classes)}
    tr = list(sp["train"]); va = [i for i in sp["val"] if recs[i]["food"] in cidx]; how = "분할 val 중 train 음식"
    if len(va) < 1000:
        gs = sorted({group(recs[i]) for i in tr}); random.Random(42).shuffle(gs); hold = set(gs[:len(gs) // 10])
        va += [i for i in tr if group(recs[i]) in hold]; tr = [i for i in tr if group(recs[i]) not in hold]; how += " + train 10% 접시"
    te = list(sp["test"])
    if a.smoke: tr, va, te, a.epochs = tr[:256], va[:128], te[:128], 1
    rd = EXP / ("runs_smoke" if a.smoke else "runs") / f"cls_{a.split}"; rd.mkdir(parents=True, exist_ok=True); log = open(rd / "train.log", "a")
    def P(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
    P(f"[cls_{a.split}] {len(classes)}종 | train {len(tr)} val {len(va)} ({how}) test {len(te)} | {a.device}")
    mk = lambda ids, tf, sh: DataLoader(DS(recs, ids, cidx, tf), batch_size=a.batch_size, shuffle=sh, num_workers=a.workers,
                                       pin_memory=True, drop_last=sh, persistent_workers=True)
    trL, vaL, teL = mk(tr, train_transform(a.img_size), True), mk(va, eval_transform(a.img_size), False), mk(te, eval_transform(a.img_size), False)
    import timm
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(classes), drop_path_rate=0.1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05); steps, warm = a.epochs * len(trL), len(trL)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.cuda.amp.GradScaler(); crit = nn.CrossEntropyLoss(label_smoothing=0.1); best, bad, bep = -1.0, 0, 0
    for ep in range(1, a.epochs + 1):
        model.train(); tot = n = 0
        for x, y, _ in trL:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16): loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item() * len(y); n += len(y)
        lg, lb, _ = logits_of(model, vaL, dev); acc = float((lg.argmax(1) == lb).float().mean())
        P(f"ep{ep:02d} loss {tot / n:.4f} val top1 {acc:.4f}")
        if acc > best:
            best, bad, bep = acc, 0, ep
            torch.save({"state_dict": model.state_dict(), "model_name": MODEL_NAME, "classes": classes, "img_size": a.img_size, "temperature": 1.0, "epoch": ep}, rd / "best.pth")
        else:
            bad += 1
            if bad >= a.patience: P(f"early stop ep {ep}, best ep {bep}"); break
    ck = torch.load(rd / "best.pth", map_location="cpu"); model.load_state_dict(ck["state_dict"])
    vl, vy, _ = logits_of(model, vaL, dev); T = fit_temperature(vl, vy); ck["temperature"] = T; torch.save(ck, rd / "best.pth")
    tl, ty, ti = logits_of(model, teL, dev); prob = F.softmax(tl / T, 1); p5, i5 = prob.topk(min(5, len(classes)), 1)
    pred = {int(i): {"top": [classes[k] for k in i5[j].tolist()], "prob": [round(float(v), 4) for v in p5[j]]} for j, i in enumerate(ti)}
    (rd / "test_pred.json").write_text(json.dumps(pred, ensure_ascii=False))
    known = ty >= 0; top1 = p5[:, 0]
    rep = {"split": a.split, "classes": classes, "best_epoch": bep, "temperature": T, "val": calibration_report(vl, vy, T),
           "test_known": calibration_report(tl[known], ty[known], T) if known.any() else None,
           "test_n": int(len(ty)), "test_n_known_food": int(known.sum()),
           "test_top1_prob_mean": {"known_food": round(float(top1[known].mean()), 4) if known.any() else None,
                                   "unseen_food": round(float(top1[~known].mean()), 4) if (~known).any() else None},
           "test_rate_prob_ge_0.7": {"known_food": round(float((top1[known] >= 0.7).float().mean()), 4) if known.any() else None,
                                     "unseen_food": round(float((top1[~known] >= 0.7).float().mean()), 4) if (~known).any() else None}}
    (rd / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    P(json.dumps({k: rep[k] for k in ("best_epoch", "temperature", "test_n_known_food", "test_top1_prob_mean", "test_rate_prob_ge_0.7")}, ensure_ascii=False))
    P("DONE")


if __name__ == "__main__":
    main()
