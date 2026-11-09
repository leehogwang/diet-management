"""처음 보는 음식을 알아채는 음식 분류기 (open-set). train_classifier.py 와 같은 백본·증강·옵티마이저에 다음을 바꾼다.
1) softmax 확신도로 판정하지 않는다: val 에 '처음 보는 음식'을 넣고, 판정 threshold 는 val 로 정한다.
   - 122_2_forunseen: 분할 val 중 train 에 없는 음식(11종, test 음식과도 겹치지 않음) = unknown val.
     known 은 train 10% 접시(시점 묶음)를 떼어 반은 known val(threshold), 반은 known test(평가)로 쓴다.
   - 122_1_forseen: 처음 보는 음식이 없다. known val = 분할 val, known test = 분할 test (threshold 는 known val 95% 통과 기준).
3) ArcFace(metric learning): 임베딩 512차원, 정규화, scale 30, margin 0.3. 같은 음식은 촘촘하게, 다른 음식은 멀게 학습.
2) 판정 점수(클수록 아는 음식): msp(softmax 최대), energy(logsumexp), proto_cos(ArcFace 클래스 가중치와 최대 cos),
   center_cos(train 임베딩 클래스 평균과 최대 cos), knn(train 임베딩과 k번째 최근접 cos), maha(-Mahalanobis 최소 거리).
   threshold = known val 의 95% 가 통과하는 값 (+ unknown val 이 있으면 val 균형정확도 최대 값도 함께 보고).
--baseline 은 기존 CE 분류기(runs/cls_<split>/best.pth)를 학습 없이 같은 점수·같은 평가 세트로 채점한다 (비교용).
출력 EXP/runs/cls_osr_<split>[_baseline]/{best.pth, report.json, test_pred.json, train.log}
usage: python train_classifier_osr.py --split 122_2_forunseen --device cuda:0 [--baseline] [--eval_only] [--smoke]"""
import argparse, json, math, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/home/leehogwang/food_b"); sys.path.insert(0, str(EXP / "src"))
from n296.food_classifier import MODEL_NAME, eval_transform, fit_temperature, train_transform  # noqa: E402
from train_classifier import DS, group  # noqa: E402

SCORES = ["msp", "energy", "proto_cos", "center_cos", "knn", "maha"]
TPR = 0.95


class ArcHead(nn.Module):
    def __init__(s, d, C, scale, margin):
        super().__init__(); s.W = nn.Parameter(torch.randn(C, d) * 0.01); s.s, s.m = scale, margin

    def cos(s, z): return F.normalize(z, dim=1) @ F.normalize(s.W, dim=1).t()

    def forward(s, z, y=None):
        c = s.cos(z)
        if y is None: return s.s * c
        th = torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6))
        tgt = torch.where(th < math.pi - s.m, torch.cos(th + s.m), c - s.m * math.sin(s.m))
        return s.s * torch.where(F.one_hot(y, c.size(1)).bool(), tgt, c)


class ArcNet(nn.Module):
    def __init__(s, C, dim=512, scale=30.0, margin=0.3, pretrained=True):
        super().__init__(); import timm
        s.bb = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0, drop_path_rate=0.1)
        s.proj = nn.Sequential(nn.Linear(s.bb.num_features, dim), nn.BatchNorm1d(dim)); s.head = ArcHead(dim, C, scale, margin)

    def embed_logits(s, x):
        z = s.proj(s.bb(x)).float(); return z, s.head(z)


class CENet(nn.Module):  # 기존 분류기 래퍼: 임베딩 = 분류층 직전 특징
    def __init__(s, C):
        super().__init__(); import timm
        s.m = timm.create_model(MODEL_NAME, pretrained=False, num_classes=C)

    def embed_logits(s, x):
        z = s.m.forward_head(s.m.forward_features(x), pre_logits=True).float(); return z, s.m.get_classifier()(z)


@torch.no_grad()
def extract(model, loader, dev):
    model.eval(); Z, L, Y, I = [], [], [], []
    for x, y, i in loader:
        with torch.autocast("cuda", dtype=torch.float16): z, lg = model.embed_logits(x.to(dev, non_blocking=True))
        Z.append(z.float().cpu()); L.append(lg.float().cpu()); Y.append(y); I += i.tolist()
    return {"z": torch.cat(Z), "l": torch.cat(L), "y": torch.cat(Y), "id": I}


def auroc(pos, neg):  # pos(아는 음식) 점수가 neg 보다 클 확률
    s = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)  # 동점은 평균 순위
    csum = np.cumsum(cnt); avg = (csum - (cnt - 1) / 2.0); r = avg[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Scorer:
    def __init__(s, tr, T, proto=None, k=10):
        zt = F.normalize(tr["z"], dim=1); y = tr["y"]; C = int(y.max()) + 1
        s.T, s.k, s.proto, s.zt = T, k, proto, zt
        present = [c for c in range(C) if (y == c).any()]  # train 사진이 없는 클래스는 뺀다
        mu_all = torch.zeros(C, zt.size(1)); mu_all[present] = torch.stack([zt[y == c].mean(0) for c in present])
        mu = mu_all[present]; s.cent = F.normalize(mu, dim=1); s.mu = mu
        d = zt - mu_all[y]; cov = d.t() @ d / len(d); cov += 1e-3 * torch.eye(cov.size(0)) * cov.diagonal().mean()
        s.P = torch.linalg.inv(cov)

    def __call__(s, e):
        z = F.normalize(e["z"], dim=1); lg = e["l"] / s.T; out = {}
        out["msp"] = F.softmax(lg, 1).max(1).values
        out["energy"] = torch.logsumexp(lg, 1)
        out["proto_cos"] = (z @ F.normalize(s.proto, dim=1).t()).max(1).values if s.proto is not None else None
        out["center_cos"] = (z @ s.cent.t()).max(1).values
        sims = torch.cat([(z[i:i + 1024] @ s.zt.t()).topk(s.k, 1).values[:, -1] for i in range(0, len(z), 1024)])
        out["knn"] = sims
        dm = []
        for i in range(0, len(z), 1024):
            d = z[i:i + 1024, None, :] - s.mu[None]                       # (b, C, D)
            dm.append(-torch.einsum("bcd,de,bce->bc", d, s.P, d).min(1).values)
        out["maha"] = torch.cat(dm)
        return {k: v.numpy() for k, v in out.items() if v is not None}


def operating(score_k, score_u, pred_k, y_k, t):
    """threshold t 이상 = 분류기 사용, 미만 = Gemini 로 넘김."""
    r = {"threshold": float(t)}
    keep = score_k >= t
    r["known_coverage"] = float(keep.mean()); r["known_acc_kept"] = float((pred_k[keep] == y_k[keep]).mean()) if keep.any() else None
    r["known_acc_overall"] = float(((pred_k == y_k) & keep).mean())          # 분류기가 맞힌 비율 (넘긴 것은 빠짐)
    if score_u is not None:
        r["unknown_to_gemini"] = float((score_u < t).mean()); r["unknown_mislabeled"] = float((score_u >= t).mean())
    return r


def evaluate(scorer, sets, classes):
    """sets: kval, uval(없을 수 있음), ktest, utest(없을 수 있음) → 점수별 지표."""
    S = {k: (scorer(v) if v is not None else None) for k, v in sets.items()}
    P = {k: (v["l"].argmax(1).numpy() if v is not None else None) for k, v in sets.items()}
    Y = {k: (v["y"].numpy() if v is not None else None) for k, v in sets.items()}
    rep = {"known_val_top1": float((P["kval"] == Y["kval"]).mean()), "known_test_top1": float((P["ktest"] == Y["ktest"]).mean()), "scores": {}}
    for name in S["kval"]:
        sk, st = S["kval"][name], S["ktest"][name]; su_v = S["uval"][name] if S["uval"] else None; su_t = S["utest"][name] if S["utest"] else None
        r = {}
        t95 = float(np.quantile(sk, 1 - TPR))
        if su_v is not None:
            r["val_auroc"] = auroc(sk, su_v); r["val_fpr_at95tpr"] = float((su_v >= t95).mean())
            cand = np.unique(np.concatenate([sk, su_v])); cand = cand[:: max(1, len(cand) // 2000)]
            bal = [((sk >= c).mean() + (su_v < c).mean()) / 2 for c in cand]; tb = float(cand[int(np.argmax(bal))])
        if su_t is not None:
            r["test_auroc"] = auroc(st, su_t); r["test_fpr_at95tpr_valthr"] = float((su_t >= t95).mean())
        r["at_val_tpr95"] = {"val": operating(sk, su_v, P["kval"], Y["kval"], t95), "test": operating(st, su_t, P["ktest"], Y["ktest"], t95)}
        if su_v is not None:
            r["at_val_balanced"] = {"val": operating(sk, su_v, P["kval"], Y["kval"], tb), "test": operating(st, su_t, P["ktest"], Y["ktest"], tb)}
        rep["scores"][name] = r
    return rep, S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=6); ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--img_size", type=int, default=288); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--margin", type=float, default=0.3); ap.add_argument("--scale", type=float, default=30.0); ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--primary", default="maha", help="test_pred 의 라우팅에 쓸 점수 (7장 결과로 maha 채택). auto = unknown val AUROC 최고")
    ap.add_argument("--route_tpr", type=float, default=0.99, help="unknown val 이 없을 때 라우팅 threshold = known val 이 이 비율만큼 통과하는 값")
    ap.add_argument("--baseline", action="store_true"); ap.add_argument("--eval_only", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); torch.manual_seed(42); np.random.seed(42); random.seed(42); dev = torch.device(a.device)
    recs = json.loads((EXP / "records.json").read_text()); sp = json.loads((EXP / "splits.json").read_text())[a.split]
    classes = sorted({recs[i]["food"] for i in sp["train"]}); cidx = {c: k for k, c in enumerate(classes)}
    tr = list(sp["train"]); uval = [i for i in sp["val"] if recs[i]["food"] not in cidx]
    kval = [i for i in sp["val"] if recs[i]["food"] in cidx]; ktest = [i for i in sp["test"] if recs[i]["food"] in cidx]
    utest = [i for i in sp["test"] if recs[i]["food"] not in cidx]
    if len(kval) < 1000:  # forunseen: train 10% 접시를 떼어 known val / known test 로 반씩 (기존 분류기와 같은 seed·묶음 → 기존 모델도 이 사진은 학습에 안 씀)
        gs = sorted({group(recs[i]) for i in tr}); random.Random(42).shuffle(gs); hold = gs[:len(gs) // 10]
        hv, ht = set(hold[0::2]), set(hold[1::2])
        kval += [i for i in tr if group(recs[i]) in hv]; ktest += [i for i in tr if group(recs[i]) in ht]
        tr = [i for i in tr if group(recs[i]) not in hv | ht]
    if a.smoke: tr, kval, uval, ktest, utest, a.epochs = tr[:256], kval[:128], uval[:128], ktest[:128], utest[:128], 1
    rd = EXP / ("runs_smoke" if a.smoke else "runs") / (f"cls_osr_{a.split}" + ("_baseline" if a.baseline else "")); rd.mkdir(parents=True, exist_ok=True)
    log = open(rd / "train.log", "a")

    def P(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
    P(f"[{rd.name}] {len(classes)}종 | train {len(tr)} known val {len(kval)} unknown val {len(uval)} | known test {len(ktest)} unknown test {len(utest)} | {a.device}")
    mk = lambda ids, tf, sh: DataLoader(DS(recs, ids, cidx, tf), batch_size=a.batch_size, shuffle=sh, num_workers=a.workers,
                                       pin_memory=True, drop_last=sh, persistent_workers=not sh) if ids else None
    ev = eval_transform(a.img_size)
    L = {"kval": mk(kval, ev, False), "uval": mk(uval, ev, False), "ktest": mk(ktest, ev, False), "utest": mk(utest, ev, False), "train_eval": mk(tr, ev, False)}

    if a.baseline:
        ck = torch.load(EXP / "runs" / f"cls_{a.split}" / "best.pth", map_location="cpu")
        assert ck["classes"] == classes; model = CENet(len(classes)); model.m.load_state_dict(ck["state_dict"]); model.to(dev)
    else:
        model = ArcNet(len(classes), a.dim, a.scale, a.margin, pretrained=not a.eval_only).to(dev)
        if not a.eval_only:
            trL = mk(tr, train_transform(a.img_size), True)
            head = list(model.proj.parameters()) + list(model.head.parameters())
            opt = torch.optim.AdamW([{"params": model.bb.parameters(), "lr": a.lr}, {"params": head, "lr": a.lr * 10}], weight_decay=0.05)
            steps, warm = a.epochs * len(trL), len(trL)
            sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
            scaler = torch.amp.GradScaler("cuda"); best, bad, bep = -1.0, 0, 0
            for ep in range(1, a.epochs + 1):
                model.train(); tot = n = 0
                for x, y, _ in trL:
                    x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.float16): z = model.proj(model.bb(x))
                    loss = F.cross_entropy(model.head(z.float(), y), y)
                    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
                    tot += loss.item() * len(y); n += len(y)
                kv = extract(model, L["kval"], dev); acc = float((kv["l"].argmax(1) == kv["y"]).float().mean())
                msg = f"ep{ep:02d} loss {tot / n:.4f} known val top1 {acc:.4f}"; sel = acc
                if L["uval"]:  # 선택 기준 = known 정확도 + unknown val 판별력(proto_cos AUROC) 평균
                    uv = extract(model, L["uval"], dev); W = model.head.W.detach().cpu()
                    pc = lambda e: (F.normalize(e["z"], dim=1) @ F.normalize(W, dim=1).t()).max(1).values.numpy()
                    au = auroc(pc(kv), pc(uv)); sel = (acc + au) / 2; msg += f" | unknown val AUROC(proto_cos) {au:.4f} | 선택점수 {sel:.4f}"
                P(msg)
                if sel > best:
                    best, bad, bep = sel, 0, ep
                    torch.save({"state_dict": model.state_dict(), "classes": classes, "img_size": a.img_size, "dim": a.dim, "scale": a.scale, "margin": a.margin, "epoch": ep}, rd / "best.pth")
                else:
                    bad += 1
                    if bad >= a.patience: P(f"early stop ep {ep}, best ep {bep}"); break
        ck = torch.load(rd / "best.pth", map_location="cpu"); model.load_state_dict(ck["state_dict"]); model.to(dev)

    E = {k: (extract(model, l, dev) if l is not None else None) for k, l in L.items()}
    T = fit_temperature(E["kval"]["l"], E["kval"]["y"])
    scorer = Scorer(E["train_eval"], T, proto=None if a.baseline else model.head.W.detach().cpu().float())
    rep, S = evaluate(scorer, {k: E[k] for k in ("kval", "uval", "ktest", "utest")}, classes)
    names = list(rep["scores"])
    primary = (max(names, key=lambda n: rep["scores"][n]["val_auroc"]) if E["uval"] else "center_cos") if a.primary == "auto" else a.primary
    rep.update({"split": a.split, "model": "CE baseline (runs/cls_%s)" % a.split if a.baseline else f"ArcFace dim {a.dim} s {a.scale} m {a.margin}",
                "best_epoch": ck.get("epoch"), "temperature": T, "primary": primary,
                "n": {k: (len(E[k]["id"]) if E[k] else 0) for k in E}})
    (rd / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    # test_pred: 기존 형식(top, prob) + 판정 점수 + 라우팅. threshold = unknown val 이 있으면 val 균형, 없으면 known val route_tpr 통과
    if "at_val_balanced" in rep["scores"][primary]: t = rep["scores"][primary]["at_val_balanced"]["test"]["threshold"]; rule = "val_balanced"
    else: t = float(np.quantile(S["kval"][primary], 1 - a.route_tpr)); rule = f"known_val_tpr{round(a.route_tpr * 100)}"
    rep["route"] = {"score": primary, "rule": rule, "threshold": t,
                    "test": operating(S["ktest"][primary], S["utest"][primary] if S["utest"] else None, P_k := E["ktest"]["l"].argmax(1).numpy(), E["ktest"]["y"].numpy(), t)}
    (rd / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2)); out = {}
    for k in ("ktest", "utest"):
        e = E[k]
        if e is None: continue
        pr = F.softmax(e["l"] / T, 1); p5, i5 = pr.topk(min(5, len(classes)), 1)
        for j, i in enumerate(e["id"]):
            sc = {n: round(float(S[k][n][j]), 5) for n in S[k]}
            out[int(i)] = {"top": [classes[c] for c in i5[j].tolist()], "prob": [round(float(v), 4) for v in p5[j]], "score": sc,
                           "route": "classifier" if sc[primary] >= t else "gemini"}
    (rd / "test_pred.json").write_text(json.dumps(out, ensure_ascii=False))
    P(f"route {primary} {rule} threshold {t:.4f} | {rep['route']['test']} | known test top1 {rep['known_test_top1']:.4f}")
    for n in names:
        r = rep["scores"][n]; o = r["at_val_tpr95"]["test"]
        P(f"  {n:10s} val AUROC {r.get('val_auroc', float('nan')):.4f} test AUROC {r.get('test_auroc', float('nan')):.4f} | TPR95: known 통과 {o['known_coverage']:.3f}"
          f" 정확도(통과분) {o['known_acc_kept'] or 0:.4f} unknown→Gemini {o.get('unknown_to_gemini', float('nan')):.3f}")
    P("DONE")


if __name__ == "__main__":
    main()
