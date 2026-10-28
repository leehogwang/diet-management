"""AIHub 296 라벨(json)을 스캔해 학습용 manifest CSV 를 만든다.

지원 레이아웃
  aihub : 원본 폴더 구조
          <root>/<Training|Validation>/02.라벨링데이터/TL*/.../<view>/*.json
          <root>/<Training|Validation>/01.원천데이터/TS*/.../<view>/*.jpg
  flat  : test_dataset_296 처럼 <root>/labels/*.json, <root>/images/*.jpg

타깃
  calories = nutrition.e, mass = nutrition.g, fat = nutrition.fat, protein = nutrition.pro
  carb     = 라벨에 탄수화물 필드가 없어 Atwater 역산값 (test_manifest.csv 의 carb100 과 동일한 방식)
  sugar    = nutrition.sug (-99 는 결측 → NaN, 학습 타깃 아님)

사용 예
  python -m n296.build_manifest --config configs/stage3_296.yaml                     # 원본 루트
  python -m n296.build_manifest --config configs/stage3_296.yaml \
      --root /home/students/cs/202321024/datasets/test_dataset_296 --layout flat \
      --out data/manifest_test296.csv --no-exclude
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import atwater_carb, load_config, resolve


def _label_to_image_aihub(json_path: Path) -> Path:
    parts = list(json_path.parts)
    parts = [p.replace("02.라벨링데이터", "01.원천데이터") for p in parts]
    # TL1 → TS1, VL1 → VS1
    parts = [("TS" + p[2:]) if p.startswith("TL") else ("VS" + p[2:]) if p.startswith("VL") else p for p in parts]
    return Path(*parts).with_suffix(".jpg")


def _iter_pairs(root: Path, layout: str, view: str):
    if layout == "flat":
        for jp in sorted((root / "labels").glob("*.json")):
            yield jp, root / "images" / (jp.stem + ".jpg"), "flat"
        return
    for split_dir in ("Training", "Validation"):
        label_root = root / split_dir / "02.라벨링데이터"
        if not label_root.exists():
            continue
        for jp in sorted(label_root.rglob("*.json")):
            if view and jp.parent.name != view:
                continue
            yield jp, _label_to_image_aihub(jp), split_dir


def _parse(json_path: Path) -> dict | None:
    try:
        d = json.loads(json_path.read_text(encoding="utf-8"))["data"]
    except Exception:
        return None
    info, nut = d.get("image_info", {}), d.get("nutrition", {})
    bbox = d.get("2d_annotation") or {}
    stem = json_path.stem  # A_13_A13010_대하구이_02_01
    tok = stem.split("_")
    food_code = tok[2] if len(tok) > 2 else ""
    food_name = "_".join(tok[3:-2]) if len(tok) > 5 else ""

    def num(x):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return np.nan
        return np.nan if v <= -99 else v

    e, g, fat, pro = num(nut.get("e")), num(nut.get("g")), num(nut.get("fat")), num(nut.get("pro"))
    return {
        "file_name": info.get("file_name", stem + ".jpg"),
        "food_code": food_code,
        "food_name": food_name,
        "weight": num(info.get("weight")),          # 실측 무게(그릇 포함 여부 확인 필요)
        "s_weight": num(info.get("s_weight")),      # 표준 1인분 무게 = nutrition.g
        "img_width": info.get("width"),
        "img_height": info.get("height"),
        "bbox_x": bbox.get("x"), "bbox_y": bbox.get("y"),
        "bbox_w": bbox.get("width"), "bbox_h": bbox.get("height"),
        "calories": e,
        "mass": g,
        "fat": fat,
        "protein": pro,
        "carb": float(atwater_carb(e, fat, pro)) if not np.isnan([e, fat, pro]).any() else np.nan,
        "sugar": num(nut.get("sug")),
        "sodium": num(nut.get("na")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=None, help="기본값: config data.root_296")
    ap.add_argument("--layout", choices=["aihub", "flat"], default="aihub")
    ap.add_argument("--out", default=None, help="기본값: config paths.manifest")
    ap.add_argument("--no-exclude", action="store_true", help="exclude_manifest 를 적용하지 않음")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.root or cfg["data"]["root_296"])
    out = resolve(args.out or cfg["paths"]["manifest"])
    if not root.exists():
        raise SystemExit(f"데이터 루트가 없습니다: {root}")

    rows, missing = [], 0
    for jp, ip, split_dir in tqdm(list(_iter_pairs(root, args.layout, cfg["data"]["view"])), desc="labels"):
        if not ip.exists():
            missing += 1
            continue
        r = _parse(jp)
        if r is None:
            continue
        r.update(image_path=str(ip), json_path=str(jp), source_split=split_dir, has_image=1)
        rows.append(r)
    df = pd.DataFrame(rows)
    print(f"라벨 {len(df)}개 (이미지 없음 {missing}개)")

    excl = cfg["data"].get("exclude_manifest")
    if excl and not args.no_exclude and Path(excl).exists():
        ex_names = set(Path(p).name for p in pd.read_csv(excl)["image_path"])
        before = len(df)
        df = df[~df["file_name"].isin(ex_names)]
        print(f"exclude_manifest 로 {before - len(df)}개 제외 (테스트셋 누수 방지)")

    bad = df[["calories", "fat", "protein", "mass"]].isna().any(axis=1)
    if bad.any():
        print(f"필수 타깃 결측 {int(bad.sum())}개 제외")
        df = df[~bad]

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"저장: {out}  (음식 {df['food_code'].nunique()}종, 이미지 {len(df)}장)")


if __name__ == "__main__":
    main()
