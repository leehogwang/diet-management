"""n296_stage3_3 영양 모델(n296.model.Stage3NutritionModel: 텍스트 임베딩 + ConvNeXt-Base 이미지 + FFT 주파수 분기 + Mixer) 학습·평가.
분할 하나(122_1_forseen | 122_2_forunseen | n5k)만 써서 한 번 학습. 설정은 configs/aihub122.yaml, 학습 루프는 실험 40 3번(r3) 과 같다:
  표준화 타깃 SmoothL1, AdamW 3e-4 (wd 1e-4), batch 64, ReduceLROnPlateau(0.5, 3), grad clip 5, AMP, modality dropout 0.3,
  입력 224 (학습: RandomResizedCrop 0.7~1 + hflip + ColorJitter / 평가: resize 256 → center crop 224),
  검증 G4(칼로리·지방·단백질·탄수화물 G 평균) 최저 체크포인트, patience 15, 시간 상한 --budget_min. ImageNet 시작, seed 42.
학습 텍스트 = 정답(122 음식명, N5k 재료 목록). test 는 text_gt(정답 텍스트) · text_gemini(Gemini 가 사진만 보고 추정한 텍스트) 두 조건.
출력 EXP/runs/<split>/{best.pth, history.csv, result.json, preds_text_{gt,gemini}.npz, train.log}
--text ing: 122 텍스트를 파이프라인 형식 "음식명 (재료: …)" 으로 (src/texts_ing.py, 임베딩 emb_cache_ing.npz). 출력 폴더 끝에 _ing.
--cat: 영양 모델에 대분류 입력 추가 (model.cat_dim 64, 헤드 입력에 붙임). 학습은 정답표 대분류(20% 확률로 기타로 가림), --text ing 필요. 출력 폴더 끝에 _cat.
  --text ing 이면 test 조건이 gt / gemini / pipe 세 가지 (texts_ing.cond_inputs). pipe = 분류기 → Gemini 음식명 → Gemini 재료·대분류 (실제 파이프라인).
usage: python train.py --split n5k --device cuda:0 [--budget_min 150] [--smoke] [--text ing [--cat]] [--eval_only]"""
import argparse, copy, csv, json, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision, yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

EXP = Path(__file__).resolve().parents[1]
REPO = Path("/home/leehogwang/food_b")
sys.path.insert(0, str(REPO))
from n296.model import Stage3NutritionModel  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts_ing  # noqa: E402
gt_text_fn = None  # main 에서 --text 에 따라 정한다 (DataLoader 워커는 fork 라 그대로 물려받는다)

KEYS = ["calories", "fat", "protein", "carb", "mass"]          # 레코드 y 순서
M_ORDER = ["calories", "mass", "fat", "carb", "protein"]       # 모델 헤드 순서
R_IDX = [KEYS.index(k) for k in M_ORDER]                       # y → 모델 순서
B_IDX = [M_ORDER.index(k) for k in KEYS]                       # 모델 순서 → y


def metrics(P, T):
    """P,T [N,5] (칼로리·지방·단백질·탄수화물·중량). PMAE = Σ|p-t|/Σt ×100 (G, README 의 PMAE 와 같음), A = mean(|p-t|/(|t|+1)) ×100.
    G4/A4 = 앞 4개 평균(실험 40 주 지표), mean_pmae = 칼로리·지방·단백질 PMAE 평균(n296 README 정의), MAE 원 단위."""
    P, T = np.asarray(P, np.float64), np.asarray(T, np.float64); e = np.abs(P - T)
    G = e.sum(0) / T.sum(0) * 100; A = (e / (np.abs(T) + 1)).mean(0) * 100; mae = e.mean(0)
    return {"G4": round(float(G[:4].mean()), 2), "A4": round(float(A[:4].mean()), 2), "mean_pmae": round(float(G[[0, 1, 2]].mean()), 2),
            "G": dict(zip(KEYS, G.round(2).tolist())), "A": dict(zip(KEYS, A.round(2).tolist())), "MAE": dict(zip(KEYS, mae.round(2).tolist())),
            "n": int(len(T))}


def gt_text(r):
    if r["ds"] == "122": return r["food"]
    ing = list(dict.fromkeys(r["ingr"])); random.Random(r["name"]).shuffle(ing); return ", ".join(ing)


def load_pred():
    f = EXP / "gemini_pred.jsonl"; out = {}
    if f.exists():
        for l in open(f):
            j = json.loads(l); out[j["id"]] = j["food"] if "food" in j else ", ".join(dict.fromkeys(j["ingr"]))
    return out


CAT_DROP = 0.2  # 학습 때 대분류를 기타로 가리는 확률 (train_quantity --cat_dropout 과 같음). test 의 Gemini 대분류 기타에 대비


class DS(Dataset):
    """cmap: None 이면 정답 텍스트·대분류, 아니면 {id: (텍스트, 대분류)} (Gemini/파이프라인 조건)."""
    def __init__(s, recs, ids, train, cmap, emb, q4=False, cat_drop=0.0):
        s.r = [recs[i] for i in ids]; s.train, s.cmap, s.emb, s.q4, s.cd = train, cmap, emb, q4, cat_drop
        s.cj = torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.02)
    def __len__(s): return len(s.r)
    def __getitem__(s, i):
        r = s.r[i]; im = Image.open(r["img"]).convert("RGB")
        if max(im.size) > 640: im.thumbnail((640, 640), Image.BILINEAR)
        if s.train:
            im = TF.resized_crop(im, *torchvision.transforms.RandomResizedCrop.get_params(im, (0.7, 1.0), (3 / 4, 4 / 3)), [224, 224])
            if random.random() < 0.5: im = TF.hflip(im)
            im = s.cj(im)
        else: im = TF.center_crop(TF.resize(im, 256), [224, 224])
        if s.cmap is not None:
            v = s.cmap.get(r["id"]); ok = v is not None and v[0] in s.emb
            te = torch.tensor(s.emb[v[0]]) if ok else torch.zeros(3072); cat = v[1] if v else texts_ing.UNKNOWN
        else: te, ok, cat = torch.tensor(s.emb[gt_text_fn(r)]), True, texts_ing.gt_cat(r)
        if s.train and random.random() < s.cd: cat = texts_ing.UNKNOWN
        return {"x": TF.to_tensor(im), "t": te, "c": texts_ing.cat_index(cat), "ok": ok,
                "y": torch.tensor(target(r, s.q4), dtype=torch.float32), "id": r["id"]}


RATIO = [0.25, 0.5, 0.75, 1.0, 1.25]


def target(r, q4):
    """q4=True 면 1인분(Q4) 정답 = 실제 정답 ÷ Q 비율 (stage3_3 파이프라인의 영양 모델 학습 방식). 122 만."""
    return [v / RATIO[r["q"]] for v in r["y"]] if q4 else r["y"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--budget_min", type=float, default=150)
    ap.add_argument("--patience", type=int, default=15); ap.add_argument("--workers", type=int, default=12); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--target", choices=["actual", "q4"], default="actual", help="q4: 1인분 정답으로 학습 (파이프라인용, 출력 폴더 <split>_q4)")
    ap.add_argument("--cat", action="store_true", help="영양 모델에 대분류 입력 (출력 폴더 끝에 _cat)")
    ap.add_argument("--eval_only", action="store_true", help="학습 없이 best.pth 로 test 만 다시 평가해 result.json 의 test 를 갱신")
    ap.add_argument("--text", choices=["name", "ing"], default="name", help="ing: 122 텍스트 = 음식명 (재료: …), 출력 폴더 끝에 _ing")
    a = ap.parse_args(); q4 = a.target == "q4"
    assert not a.cat or a.text == "ing", "--cat 은 --text ing 과 함께"
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    dev = torch.device(a.device)
    recs = json.loads((EXP / "records.json").read_text()); sp = json.loads((EXP / "splits.json").read_text())[a.split]
    if a.smoke: sp = {k: v[:128] for k, v in sp.items()}; a.epochs = 1
    rd = EXP / ("runs_smoke" if a.smoke else "runs") / (a.split + ("_q4" if q4 else "") + ("_ing" if a.text == "ing" else "") + ("_cat" if a.cat else "")); rd.mkdir(parents=True, exist_ok=True)
    log = open(rd / "train.log", "a")
    def P(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
    global gt_text_fn
    if a.text == "ing":
        gt_text_fn, read_emb = texts_ing.gt_text, texts_ing.load_emb
        read_conds = lambda: {c: texts_ing.cond_inputs(c, a.split) for c in ("gt", "gemini", "pipe", "pipe_osr")}
    else:
        gt_text_fn, read_emb = gt_text, lambda: json.loads((EXP / "emb_cache.json").read_text())
        read_conds = lambda: {"gt": None, "gemini": {i: (t, texts_ing.UNKNOWN) for i, t in load_pred().items()}}
    emb = read_emb(); conds = read_conds()
    miss = {gt_text_fn(recs[i]) for k in ("train", "val") for i in sp[k]} - emb.keys()
    assert not miss, f"정답 텍스트 임베딩 없음 {len(miss)}개: {sorted(miss)[:3]}"

    cfg = yaml.safe_load((REPO / "configs/aihub122.yaml").read_text())
    if a.cat: cfg["model"]["cat_dim"] = 64
    model = Stage3NutritionModel(cfg, pretrained=True); frozen = model.freeze_pretrained_layers(cfg); model = model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    Ytr = np.array([target(recs[i], q4) for i in sp["train"]], np.float32)
    ymu, ysd = torch.tensor(Ytr.mean(0)).to(dev), torch.tensor(Ytr.std(0) + 1e-6).to(dev)

    def loader(ids, train, tm="gt"):
        return DataLoader(DS(recs, ids, train, conds[tm], emb, q4, CAT_DROP if (train and a.cat) else 0.0), batch_size=64, shuffle=train, num_workers=a.workers, pin_memory=True, drop_last=train)

    def fwd(b, train):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(b["x"].to(dev, non_blocking=True), b["t"].to(dev, non_blocking=True), modality_dropout=train,
                      cat=b["c"].to(dev, non_blocking=True) if a.cat else None)
        z = torch.cat([o[k].float() for k in M_ORDER], 1)                       # 표준화 공간, 모델 순서
        y = (z * ysd[R_IDX] + ymu[R_IDX])[:, B_IDX].clamp(min=0)                 # 원 단위, 레코드 순서
        return y, z

    @torch.no_grad()
    def predict(ids, tm):
        model.eval(); Ps, Ts, Is, Ok = [], [], [], []
        for b in loader(ids, False, tm):
            p, _ = fwd(b, False); Ps.append(p.cpu().numpy()); Ts.append(b["y"].numpy()); Is += b["id"].tolist(); Ok += b["ok"].tolist()
        return np.concatenate(Ps), np.concatenate(Ts), np.array(Is), np.array(Ok, bool)

    if a.eval_only:
        ck = torch.load(rd / "best.pth", map_location="cpu"); model.load_state_dict(ck["state_dict"]); res = json.loads((rd / "result.json").read_text())
        for tm in conds:
            Pt, Tt, It, ok = predict(sp["test"], tm)
            np.savez(rd / f"preds_text_{tm}.npz", pred=Pt, target=Tt, id=It, ok=ok)
            m = metrics(Pt[ok], Tt[ok]); m["n_missing_text"] = int((~ok).sum()); res["test"][f"text_{tm}"] = m
            P(f"[EVAL_ONLY text_{tm}] G4 {m['G4']:.2f} A4 {m['A4']:.2f} | G {m['G']} | n {m['n']} (텍스트 없음 {m['n_missing_text']})")
        (rd / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False)); P("DONE"); return
    trL = loader(sp["train"], True)
    P(f"[{a.split}] train {len(sp['train'])} val {len(sp['val'])} test {len(sp['test'])} | 학습 파라미터 {sum(p.numel() for p in params) / 1e6:.1f}M"
      f" (얼린 모듈 {len(frozen)}) | batch 64 | budget {a.budget_min}분 | {a.device}")
    hist = open(rd / "history.csv", "w", newline=""); hw = csv.writer(hist)
    hw.writerow(["epoch", "sec", "lr", "train_loss", "val_G4", "val_A4", "val_mean_pmae"] + [f"val_G_{k}" for k in KEYS]); hist.flush()
    best, bad, bep, best_sd, t0, capped = float("inf"), 0, 0, None, time.time(), False
    for ep in range(1, a.epochs + 1):
        model.train(); te = time.time(); tot = n = 0
        for b in trL:
            _, z = fwd(b, True)
            yn = (b["y"].to(dev)[:, R_IDX] - ymu[R_IDX]) / ysd[R_IDX]
            loss = F.smooth_l1_loss(z, yn)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            tot += loss.item() * len(b["id"]); n += len(b["id"])
            if time.time() - t0 > a.budget_min * 60: break
        Pv, Tv, _, _ = predict(sp["val"], "gt"); v = metrics(Pv, Tv); sch.step(v["G4"])
        lr = opt.param_groups[0]["lr"]
        hw.writerow([ep, round(time.time() - te), lr, round(tot / max(n, 1), 5), v["G4"], v["A4"], v["mean_pmae"]] + [v["G"][k] for k in KEYS]); hist.flush()
        P(f"ep {ep:3d} ({time.time() - te:.0f}s) lr {lr:.2e} loss {tot / max(n, 1):.4f} | val G4 {v['G4']:.2f} A4 {v['A4']:.2f} | {v['G']}")
        if v["G4"] < best:
            best, bad, bep = v["G4"], 0, ep; best_sd = copy.deepcopy({k: t.detach().cpu() for k, t in model.state_dict().items()})
        else:
            bad += 1
            if bad >= a.patience: P(f"[EarlyStop] ep {ep}, best ep {bep}"); break
        if time.time() - t0 > a.budget_min * 60: capped = True; P(f"[TimeCap] {a.budget_min}분 도달, ep {ep}, best ep {bep}"); break
    model.load_state_dict(best_sd); torch.save({"state_dict": best_sd, "target_mean": ymu.cpu(), "target_std": ysd.cpu(), "keys": KEYS}, rd / "best.pth")
    res = {"split": a.split, "best_epoch": bep, "epochs_run": ep, "time_capped": capped, "val_G4": best, "minutes": round((time.time() - t0) / 60, 1),
           "n": {k: len(v) for k, v in sp.items()}, "train_mean": dict(zip(KEYS, Ytr.mean(0).round(3).tolist())), "test": {}}
    # 기준선: train 평균값을 모든 test 에 예측
    res["target"] = a.target; res["text"] = a.text; res["cat"] = a.cat
    Tt_all = np.array([target(recs[i], q4) for i in sp["test"]], np.float32)
    res["test"]["baseline_train_mean"] = metrics(np.tile(Ytr.mean(0), (len(Tt_all), 1)), Tt_all)
    # 학습 중 끝난 Gemini 추정·임베딩을 평가 직전에 다시 읽는다
    emb.update(read_emb()); conds.update(read_conds())
    for tm in conds:
        Pt, Tt, It, ok = predict(sp["test"], tm)
        np.savez(rd / f"preds_text_{tm}.npz", pred=Pt, target=Tt, id=It, ok=ok)
        m = metrics(Pt[ok], Tt[ok]); m["n_missing_text"] = int((~ok).sum()); res["test"][f"text_{tm}"] = m
        P(f"[TEST text_{tm}] G4 {m['G4']:.2f} A4 {m['A4']:.2f} mean_pmae {m['mean_pmae']:.2f} | G {m['G']} | n {m['n']} (텍스트 없음 {m['n_missing_text']})")
    P(f"[TEST baseline] G4 {res['test']['baseline_train_mean']['G4']:.2f}")
    (rd / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False)); P("DONE")


if __name__ == "__main__":
    main()
