"""food_split_txt 분할(122_1_forseen / 122_2_forunseen / n5k rgb)로 레코드·분할 생성.
레코드 하나 = 사진 하나: id, ds(122|n5k), name(파일명 또는 dish), img(사진 경로), food(122 음식명, 텍스트), ingr(N5k 재료),
y(칼로리·지방·단백질·탄수화물·중량), q(122 등급 0~4, N5k -1), view(side|top|overhead|camera_X).
- 122: 실험 40 records.json(측면 19,713장, 512px 사본 unified40/img) + food_b manifest_122.csv(위쪽 사진 포함 39,294장, 512px 사본 shared quantity_122/img).
- N5k: rgb_train/rgb_test. 사진 = overhead rgb.png, 없으면 측면 영상 프레임(camera_A~D frame_010). val = train 의 10%(촬영 시각 60초 묶음, seed 42, 실험 40 과 같은 방식).
출력: EXP/records.json, EXP/splits.json, EXP/prep_report.json.  usage: python prep.py"""
import csv, json, os, random, sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
SPLIT = Path("/home/leehogwang/food_split_txt")
E40 = Path("/home/leehogwang/Food-Nutrition-Estimation/experiments/40_unified")
U40 = Path("/mnt/data/food_leehogwang/unified40/img")
Q122 = Path("/mnt/data/food_leehogwang/shared_models/m3_122/n296_stage3_3/data/quantity_122/img")
MAN = Path("/home/leehogwang/food_b/data/manifest_122.csv")
N5K = Path("/mnt/data/food_leehogwang/nutrition5k/nutrition5k_dataset")
QL = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "Q5": 4}


def rd(p): return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]


def n5k_meta():
    out = {}
    for f in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"):
        for line in open(N5K / "metadata" / f):
            p = line.strip().split(",")
            if len(p) < 6: continue
            out[p[0]] = {"y": [float(p[1]), float(p[3]), float(p[5]), float(p[4]), float(p[2])],   # cal, fat, prot, carb, mass
                         "ingr": [p[i + 1] for i in range(6, len(p) - 6, 7)]}
    return out


def n5k_image(d):
    o = N5K / "imagery/realsense_overhead" / d / "rgb.png"
    if o.exists() and o.stat().st_size > 0: return str(o), "overhead"
    fr = N5K / "imagery/side_angles" / d / "frames_sampled5"
    if fr.is_dir():
        for c in "ABCD":
            f = fr / f"camera_{c}_frame_010.jpeg"
            if f.exists() and f.stat().st_size > 0: return str(f), f"camera_{c}"
        fs = sorted(x for x in os.listdir(fr) if x.endswith(".jpeg"))
        if fs: return str(fr / fs[0]), fs[0].split("_frame")[0]
    return None, None


def main():
    recs, S, rep = [], {}, {}
    e40 = json.loads((E40 / "records.json").read_text())
    r40 = {r["filename"]: r for r in e40 if r["ds"] == "122"}
    man = {r["file_name"]: r for r in csv.DictReader(open(MAN, encoding="utf-8"))}
    # manifest 음식명 → 실험 40 정규화 음식명(텍스트), 측면 사진으로 대응
    fmap = {}
    for fn, r in r40.items():
        if fn in man: fmap.setdefault(man[fn]["food_name"], Counter())[r["food"]] += 1
    fmap = {k: v.most_common(1)[0][0] for k, v in fmap.items()}
    by122 = {}

    def rec122(fn):
        if fn in by122: return by122[fn]
        if fn in r40:
            r = r40[fn]; img = U40 / f"{r['id']}.jpg"
            x = {"ds": "122", "name": fn, "img": str(img), "food": r["food"], "raw_food": r["raw_food"], "y": r["y"], "q": r["q"],
                 "view": fn.split("_")[0], "src": r["src"], "e40_id": r["id"]}
        else:
            m = man[fn]; img = Q122 / (os.path.splitext(fn)[0] + ".jpg")
            x = {"ds": "122", "name": fn, "img": str(img), "food": fmap.get(m["food_name"], m["food_name"]), "raw_food": m["label_food_name"],
                 "y": [float(m[k]) for k in ("calories", "fat", "protein", "carb", "mass")], "q": QL[m["q_level"]],
                 "view": m["view"], "src": m["source_split"], "e40_id": None}
        assert Path(x["img"]).exists(), x["img"]
        x["id"] = len(recs); recs.append(x); by122[fn] = x["id"]; return x["id"]

    for t in ("122_1_forseen", "122_2_forunseen"):
        S[t] = {s: [rec122(fn) for fn in rd(SPLIT / t / f"{t}_{s}_ids.txt")] for s in ("train", "val", "test")}
    # 누수 확인
    for t in S:
        a = {k: set(v) for k, v in S[t].items()}
        assert not (a["train"] & a["val"]) and not (a["train"] & a["test"]) and not (a["val"] & a["test"]), t
    tf = {recs[i]["food"] for i in S["122_2_forunseen"]["test"]}
    assert not tf & {recs[i]["food"] for i in S["122_2_forunseen"]["train"]}, "unseen test 음식이 train 에 있음"

    # ── N5k rgb ──
    meta = n5k_meta(); e40n = {r["dish"]: r["id"] for r in e40 if r["ds"] == "n5k"}
    drop = Counter()

    def ok(d):
        if d not in meta: drop["메타 없음"] += 1; return None
        if meta[d]["y"][0] <= 0: drop["칼로리 0"] += 1; return None
        p, v = n5k_image(d)
        if p is None: drop["사진 없음"] += 1; return None
        return p, v

    tr_all, te = [], []
    for f, L in (("rgb_train_ids.txt", tr_all), ("rgb_test_ids.txt", te)):
        for d in rd(SPLIT / "n5k" / f):
            o = ok(d)
            if o: L.append((d, *o))
    ts = sorted(tr_all, key=lambda x: int(x[0].split("_")[1])); groups, cur = [], [ts[0]]
    for a, b in zip(ts, ts[1:]):
        if int(b[0].split("_")[1]) - int(a[0].split("_")[1]) <= 60: cur.append(b)
        else: groups.append(cur); cur = [b]
    groups.append(cur); random.Random(42).shuffle(groups); va, n = [], 0
    for g in groups:
        if n >= 0.1 * len(tr_all): break
        va += g; n += len(g)
    vs = {x[0] for x in va}; tr = [x for x in tr_all if x[0] not in vs]
    S["n5k"] = {}
    for s, L in (("train", tr), ("val", va), ("test", te)):
        S["n5k"][s] = []
        for d, p, v in L:
            S["n5k"][s].append(len(recs))
            recs.append({"id": len(recs), "ds": "n5k", "name": d, "img": p, "ingr": meta[d]["ingr"], "y": meta[d]["y"], "q": -1,
                         "view": v, "e40_id": e40n.get(d) if v == "overhead" else None})
    rep["n5k_drop"] = dict(drop)
    for t in S:
        rep[t] = {s: {"n": len(v), "views": dict(Counter(recs[i]["view"] for i in v)),
                      "foods": len({recs[i].get("food") for i in v}) if t != "n5k" else None} for s, v in S[t].items()}
    (EXP / "records.json").write_text(json.dumps(recs, ensure_ascii=False))
    (EXP / "splits.json").write_text(json.dumps(S))
    (EXP / "prep_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
