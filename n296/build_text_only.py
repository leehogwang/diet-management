"""영양성분 DB(이미지 없음) → 텍스트 전용 학습 샘플 manifest.

config 의 data.enable_text_only / xlsx_path / csv_path_std 에 대응한다.
원본 학습 코드가 없어 컬럼 매핑과 1인분 환산 방식은 [추정]이다. 실행 전
--dry-run 으로 인식된 컬럼을 반드시 확인할 것.

타깃 (1인분 기준, 296 manifest 와 단위를 맞춤)
  mass     = 1인분 중량(g)          (없으면 100g)
  calories = 에너지(kcal/100g) × mass/100
  fat, protein 도 같은 방식, carb = Atwater 역산 (296 라벨과 같은 정의)

사용 예
  python -m n296.build_text_only --config configs/stage3_296.yaml --dry-run
  python -m n296.build_text_only --config configs/stage3_296.yaml --out data/manifest_text_only.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import atwater_carb, load_config, resolve

CANDIDATES = {
    "food_name": ["식품명", "음식명", "foodNm", "food_name"],
    "category": ["데이터구분명", "식품대분류명", "대분류", "구분"],
    "energy": ["에너지(kcal)", "에너지(㎉)", "에너지", "enerc"],
    "protein": ["단백질(g)", "단백질", "prot"],
    "fat": ["지방(g)", "지방", "fatce"],
    "carb": ["탄수화물(g)", "탄수화물", "chocdf"],
    "basis": ["영양성분함량기준량", "nutConSrtrQua"],
    "serving": ["식품중량", "1회제공량", "foodSize", "총내용량"],
}


def _pick(df: pd.DataFrame, key: str) -> str | None:
    for c in CANDIDATES[key]:
        if c in df.columns:
            return c
    return None


def _grams(x) -> float:
    if pd.isna(x):
        return np.nan
    m = re.search(r"([\d.]+)\s*(g|ml|mL)?", str(x))
    return float(m.group(1)) if m else np.nan


def _load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 알 수 없음: {path}")


def convert(path: Path, db_category: str | None, dry_run: bool) -> pd.DataFrame:
    raw = _load(path)
    cols = {k: _pick(raw, k) for k in CANDIDATES}
    print(f"[{path.name}] 행 {len(raw)}개, 인식된 컬럼: {cols}")
    if dry_run:
        print(raw.head(3).T.to_string())
        return pd.DataFrame()
    need = ["food_name", "energy", "protein", "fat"]
    if any(cols[k] is None for k in need):
        raise SystemExit(f"필수 컬럼을 찾지 못함 {[k for k in need if cols[k] is None]} — CANDIDATES 수정 필요")

    df = raw
    if db_category and cols["category"]:
        df = df[df[cols["category"]].astype(str).str.contains(db_category, na=False)]
        print(f"  '{db_category}' 필터 후 {len(df)}개")

    basis = df[cols["basis"]].map(_grams) if cols["basis"] else pd.Series(100.0, index=df.index)
    serving = df[cols["serving"]].map(_grams) if cols["serving"] else pd.Series(np.nan, index=df.index)
    mass = serving.fillna(100.0)
    scale = mass / basis.fillna(100.0)

    out = pd.DataFrame({
        "food_name": df[cols["food_name"]].astype(str).str.strip(),
        "food_code": "TEXT_" + df.index.astype(str),
        "calories": pd.to_numeric(df[cols["energy"]], errors="coerce") * scale,
        "protein": pd.to_numeric(df[cols["protein"]], errors="coerce") * scale,
        "fat": pd.to_numeric(df[cols["fat"]], errors="coerce") * scale,
        "mass": mass,
        "carb_reported": (pd.to_numeric(df[cols["carb"]], errors="coerce") * scale) if cols["carb"] else np.nan,
    })
    out = out.dropna(subset=["calories", "protein", "fat"])
    out["carb"] = atwater_carb(out["calories"], out["fat"], out["protein"])
    out["image_path"] = ""
    out["has_image"] = 0
    out["source_split"] = f"text_only:{path.name}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="data/manifest_text_only.csv")
    ap.add_argument("--dry-run", action="store_true", help="컬럼 인식 결과만 출력")
    args = ap.parse_args()

    cfg = load_config(args.config)["data"]
    frames = []
    for key in ("xlsx_path", "csv_path_std"):
        p = cfg.get(key)
        if not p:
            continue
        p = resolve(p)
        if not p.exists():
            print(f"건너뜀 (파일 없음): {p}")
            continue
        frames.append(convert(p, cfg.get("db_category"), args.dry_run))
    if args.dry_run:
        return
    if not frames:
        raise SystemExit("사용할 DB 파일이 없습니다. config data.xlsx_path / csv_path_std 를 확인하세요.")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["food_name"])
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"저장: {out} ({len(df)}개). config paths.text_only_manifest 에 이 경로를 넣으세요.")


if __name__ == "__main__":
    main()
