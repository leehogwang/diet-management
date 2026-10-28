"""AIHub 122 「음식 분류를 위한 음식종류 및 양에 따른 칼로리 데이터셋」 → 학습 manifest.

(AIHub 검색 이름: 「음식 이미지 및 영양정보 텍스트」, datasetkey 74)

라벨 zip 은 풀지 않고 직접 읽는다. 라벨은 작은 파일 150만 개라 풀면 블록 낭비로 약 6GB 를 차지한다.

정답 (공식 활용가이드)
  영양DB(400종)의 중량·영양값은 Q4(100%) 기준. 양 등급 비율 Q1=0.25, Q2=0.5, Q3=0.75, Q4=1.0, Q5=1.25.
  정답 = DB 값 × Q 비율.  calories=에너지, mass=중량, fat, carb=탄수화물(DB 실측값), protein
  sugar(당류), sodium(나트륨)은 참고용으로 manifest 에만 저장 (모델 헤드는 5개 유지)

데이터 종류
  quantity       : 양추정 라벨 (음식명 + Q 등급 + 시점 side/top). 사진마다 정답이 다르다. ← 기본 학습 데이터
  classification : 음식분류 라벨 (음식 코드만). 양을 알 수 없어 Q4(1.0)로 가정. --include-classification 일 때만
  text_only      : 영양DB 400종 자체를 이미지 없는 샘플로 (--text-only-out)

음식 코드 → 음식명 [추정]: 분류 코드 400개를 정렬한 순서가 영양DB 행 순서와 같다고 본다.
  근거: 코드 앞 5자리 묶음별 개수로 DB 를 자르면 모든 묶음이 같은 종류 음식이다
  (01011=쌀밥 1개, 12011=김치 15종, 16011=한과 6종 등). 공식 대응표는 아니다.

이미지: --images-root 아래를 한 번 훑어 파일명으로 찾는다 (zip 을 푼 폴더 구조와 무관).
  이미지를 아직 받지 않았으면 image_path 가 빈 채로 저장되고, 학습 시 이미지 없는 행은 제외된다.

사용 예
  python -m n296.build_manifest_122 --config configs/aihub122.yaml
  python -m n296.build_manifest_122 --config configs/aihub122.yaml --images-root <이미지 푼 폴더>
"""
from __future__ import annotations

import argparse
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import load_config, resolve

Q_RATIO = {"Q1": 0.25, "Q2": 0.5, "Q3": 0.75, "Q4": 1.0, "Q5": 1.25}
DB_COLS = {
    "mass": "중량(g)", "calories": "에너지(kcal)", "carb": "탄수화물(g)", "sugar": "당류(g)",
    "fat": "지방(g)", "protein": "단백질(g)", "sodium": "나트륨(mg)",
}
# 양추정 폴더 이름 → 영양DB 음식명. 영양DB에 정확히 없는 10종 중 대응이 분명한 것만 연결 [추정].
# 류산슬덮밥·물회·병어찜은 대응 음식이 없어 제외. 오곡밥은 기타잡곡밥으로 근사.
QUANTITY_ALIASES = {
    "감자튀김(스틱형)": "감자튀김",
    "감자튀김(웨지감자)": "감자튀김",
    "김밥": "일반김밥",
    "오므라이스(사각배달)": "오므라이스",
    "오므라이스(접시)": "오므라이스",
    "영양돌솥밥": "돌솥밥",
    "오곡밥": "기타잡곡밥",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _zip_name(info: zipfile.ZipInfo) -> str:
    """UTF-8 플래그가 없는 zip 의 한글 파일명(cp949) 복원."""
    n = info.filename
    if not info.flag_bits & 0x800:
        try:
            n = n.encode("cp437").decode("cp949")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return n


def _parse_xml(data: bytes) -> dict:
    root = ET.fromstring(data)
    size = root.find("size")
    out = {
        "img_width": int(size.findtext("width", "0")) if size is not None else np.nan,
        "img_height": int(size.findtext("height", "0")) if size is not None else np.nan,
    }
    for obj in root.findall("object"):
        if obj.findtext("name") == "dish":
            bb = obj.find("bndbox")
            x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
            x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
            out.update(bbox_x=x1, bbox_y=y1, bbox_w=x2 - x1, bbox_h=y2 - y1)
            break
    return out


def load_db(path: Path) -> pd.DataFrame:
    db = pd.read_excel(path)
    db["food_name"] = db["음 식 명"].astype(str).str.strip()
    for k, col in DB_COLS.items():
        db[k] = pd.to_numeric(db[col], errors="coerce")  # "-" 등은 결측
    return db[["food_name"] + list(DB_COLS)].reset_index(drop=True)


def _find(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.rglob(pattern) if p.suffix == ".zip")


def read_quantity(label_root: Path, db_idx: dict) -> tuple[list[dict], dict]:
    rows, skipped = [], {}
    pat = re.compile(r"xml/([^/]+)/(Q\d)/((side|top)_[^/]+)\.xml$")
    for zp in _find(label_root, "양추정_라벨링_*.zip"):
        split = "Validation" if "_VAL" in zp.name else "Training"
        z = zipfile.ZipFile(zp)
        for info in tqdm(z.infolist(), desc=zp.name, leave=False):
            m = pat.search(_zip_name(info))
            if not m:
                continue
            folder, q, stem, view = m.groups()
            name = QUANTITY_ALIASES.get(folder, folder)
            if name not in db_idx:
                skipped[folder] = skipped.get(folder, 0) + 1
                continue
            idx = stem.rsplit("_", 1)[-1]
            rows.append({
                "file_name": stem + ".JPG", "task": "quantity", "source_split": split,
                "food_name": name, "label_food_name": folder, "food_code": name,
                "q_level": q, "q_ratio": Q_RATIO[q], "view": view,
                "split_group": f"{folder}_{q}_{idx}",  # 같은 접시의 side/top 을 한 그룹으로 [추정]
                **_parse_xml(z.read(info)),
            })
    return rows, skipped


CLS_XML_PAT = re.compile(r"xml/(\d{8})/([^/]+)\.xml$")


def classification_code2name(label_root: Path, db: pd.DataFrame) -> dict[str, str]:
    """음식분류 코드(8자리) → 영양DB 음식명. 정렬한 코드 순서 = DB 행 순서 [추정] (모듈 docstring 참고)."""
    codes = set()
    for zp in _find(label_root, "음식분류_라벨링_*.zip"):
        for info in zipfile.ZipFile(zp).infolist():
            m = CLS_XML_PAT.search(info.filename)
            if m:
                codes.add(m.group(1))
    codes = sorted(codes)
    if len(codes) != len(db):
        raise SystemExit(f"분류 코드 {len(codes)}개 ≠ 영양DB {len(db)}종 — 코드→음식명 대응을 만들 수 없음")
    return dict(zip(codes, db["food_name"]))


def read_classification(label_root: Path, db: pd.DataFrame) -> list[dict]:
    rows = []
    pat = CLS_XML_PAT
    zips = _find(label_root, "음식분류_라벨링_*.zip")
    code2name = classification_code2name(label_root, db)
    for zp in zips:
        split = "Validation" if "_VAL" in zp.name else "Training"
        z = zipfile.ZipFile(zp)
        for info in tqdm(z.infolist(), desc=zp.name, leave=False):
            m = pat.search(info.filename)
            if not m:
                continue
            code, stem = m.groups()
            rows.append({
                "file_name": stem + ".jpg", "task": "classification", "source_split": split,
                "food_name": code2name[code], "label_food_name": code, "food_code": code2name[code],
                "q_level": "Q4?", "q_ratio": 1.0, "view": "unknown",  # 양 정보 없음 → 100% 가정 [추정]
                "split_group": stem,
                **_parse_xml(z.read(info)),
            })
    return rows


def index_images(images_root: Path) -> dict[str, str]:
    """파일명(소문자) → 경로. 확장자 대소문자(JPG/jpg) 차이를 흡수한다."""
    idx = {}
    for dirpath, _, files in os.walk(images_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                idx[f.lower()] = os.path.join(dirpath, f)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--label-root", default=None, help="기본값: config data.label_root_122")
    ap.add_argument("--images-root", nargs="*", default=None, help="기본값: config data.images_root_122")
    ap.add_argument("--include-classification", action="store_true", default=None)
    ap.add_argument("--out", default=None, help="기본값: config paths.manifest")
    ap.add_argument("--text-only-out", default=None, help="기본값: config paths.text_only_manifest")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    label_root = Path(args.label_root or d["label_root_122"])
    db = load_db(Path(d["db_path_122"]))
    db_idx = {n: i for i, n in enumerate(db["food_name"])}

    rows, skipped = read_quantity(label_root, db_idx)
    print(f"양추정: {len(rows)}장 (영양DB 대응 없음으로 제외: {skipped})")
    include_cls = d.get("include_classification", False) if args.include_classification is None else True
    if include_cls:
        cls_rows = read_classification(label_root, db)
        print(f"음식분류: {len(cls_rows)}장 (Q4 가정)")
        rows += cls_rows
    df = pd.DataFrame(rows)

    views = d.get("views_122")
    if views:
        df = df[df["view"].isin(views) | (df["view"] == "unknown")]

    # 정답 = DB(Q4 기준) × Q 비율
    nut = db.set_index("food_name")
    for k in DB_COLS:
        df[k] = df["food_name"].map(nut[k]) * df["q_ratio"]

    roots = args.images_root if args.images_root is not None else d.get("images_root_122") or []
    roots = [roots] if isinstance(roots, str) else roots
    img_idx = {}
    for r in roots:
        if Path(r).exists():
            img_idx.update(index_images(Path(r)))
        else:
            print(f"이미지 폴더 없음 (건너뜀): {r}")
    df["image_path"] = df["file_name"].str.lower().map(img_idx).fillna("")
    df["has_image"] = 1
    found = int((df["image_path"] != "").sum())
    print(f"이미지 찾음: {found}/{len(df)}" + ("" if found else "  ← 이미지를 받은 뒤 --images-root 로 다시 실행하세요"))

    out = resolve(args.out or cfg["paths"]["manifest"])
    out.parent.mkdir(parents=True, exist_ok=True)
    (df.to_parquet(out, index=False) if out.suffix == ".parquet" else df.to_csv(out, index=False))
    summary = df.groupby(["task", "source_split"]).size().to_dict()
    print(f"저장: {out} ({len(df)}행, 음식 {df['food_name'].nunique()}종) {summary}")

    t_out = args.text_only_out or cfg["paths"].get("text_only_manifest")
    if t_out:
        t = db.copy()
        t["food_code"] = t["food_name"]
        t["q_level"], t["q_ratio"], t["task"] = "Q4", 1.0, "text_only"
        t["image_path"], t["has_image"], t["source_split"] = "", 0, "text_only:영양DB"
        t = t.dropna(subset=["calories", "fat", "protein", "carb", "mass"])
        t_out = resolve(t_out)
        t.to_csv(t_out, index=False)
        print(f"텍스트 전용(영양DB) 저장: {t_out} ({len(t)}종)")


if __name__ == "__main__":
    main()
