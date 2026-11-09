"""학습된 양추정 모델(runs/<split>_quantity/best.pth)을 파이프라인과 같은 대분류로 다시 test 예측한다 (재학습 없음).
대분류 조건 (texts_ing.cond_inputs 와 같은 출처):
  gt     : 정답표 대분류
  gemini : Gemini 음식명(사진만) → Gemini 가 사진 + 그 음식명을 보고 고른 대분류 (파이프라인 2단계 출력)
  pipe   : 분류기(확신도 >= 0.7) 아니면 Gemini 음식명 → Gemini 대분류 (실제 파이프라인)
  masked : 모두 기타
Gemini 결과가 없는 사진은 기타로 넣고, 그 수를 report 에 남긴다.
출력 runs/<split>_quantity/{test_logits_pipe.npz, report_pipe.json} (기존 test_logits.npz·report.json 은 그대로).
usage: python eval_quantity_cats.py --split 122_2_forunseen --device cuda:0"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import texts_ing  # noqa: E402
from train_quantity import DS, EXP, MODEL_NAME, Q_LEVELS, QuantityNet, UNKNOWN, CATEGORY_NAMES, category, eval_transform, quantity_report  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--tag", default=""); a = ap.parse_args(); dev = torch.device(a.device)
    recs = json.loads((EXP / "records.json").read_text()); sp = json.loads((EXP / "splits.json").read_text())[a.split]
    rd = EXP / "runs" / f"{a.split}_quantity{a.tag}"; ck = torch.load(rd / "best.pth", map_location="cpu"); T = ck["temperature"]
    model = QuantityNet(MODEL_NAME, len(Q_LEVELS), len(CATEGORY_NAMES), pretrained=False).to(dev); model.load_state_dict(ck["state_dict"]); model.eval()
    ids = sp["test"]; conds = {"gt": {i: category(recs[i]) for i in ids}, "masked": {i: UNKNOWN for i in ids}}
    missing = {}
    for c in ("gemini", "pipe", "pipe_osr"):
        m = texts_ing.cond_inputs(c, a.split); conds[c] = {i: (m[i][1] if i in m else UNKNOWN) for i in ids}; missing[c] = sum(i not in m for i in ids)
    rep = {"split": a.split, "temperature": T, "n_missing_gemini": missing}; out = {"id": np.array(ids)}
    for name, cats in conds.items():
        o, y = [], []
        with torch.no_grad():
            for x, c, t, *_ in DataLoader(DS(recs, ids, eval_transform(ck["img_size"]), cats), batch_size=64, num_workers=a.workers, pin_memory=True):
                with torch.autocast("cuda", dtype=torch.float16): o.append(model(x.to(dev, non_blocking=True), c.to(dev)).float().cpu())
                y.append(t)
        lg, lb = torch.cat(o), torch.cat(y)
        rep[f"test_cat_{name}"] = quantity_report(lg, lb, T); out[f"logits_{name}"] = lg.numpy(); out["label"] = lb.numpy()
        out[f"cat_{name}"] = np.array([cats[i] for i in ids])
        rep[f"known_category_{name}"] = float(np.mean([cats[i] != UNKNOWN for i in ids]))
        rep[f"category_acc_{name}"] = float(np.mean([cats[i] == conds["gt"][i] for i in ids]))
        r_ = rep[f"test_cat_{name}"]
        print(f"[{a.split} 대분류 {name}] acc {r_['acc']:.4f} ±1 {r_['acc_within1']:.4f} ratio MAE {r_['ratio_mae_argmax']:.4f} | "
              f"기타 아님 {rep[f'known_category_{name}']:.3f} 정답표와 같음 {rep[f'category_acc_{name}']:.3f}", flush=True)
    np.savez(rd / "test_logits_pipe.npz", **out); (rd / "report_pipe.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2)); print("DONE")


if __name__ == "__main__":
    main()
