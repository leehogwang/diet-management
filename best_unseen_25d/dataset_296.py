"""
AIHub 296 (비전영역, 음식이미지 및 정보소개 텍스트 데이터) 매니페스트 빌더.

122 데이터셋(dataset_aihub.py)과 달리 296은:
  - Q1~Q5 양(量) 폴더가 없다 — 대신 라벨 JSON에 샘플셋별 실측 중량(image_info.weight)이 있다.
  - Train/Validation 폴더가 완전히 별개의 촬영본이다 (파일명 전역 유일, 겹침 0건).
  - 탄수화물이 없다 — 대신 칼로리(e)/지방(fat)/단백질(pro)로 Atwater 역산한다.
  - TS1~TS16 / VS1~VS2 는 카테고리가 아니라 압축 분할 단위다 (같은 음식코드가 여러 샤드에 흩어짐).

구조:
  01-1.정식개방데이터/{Training,Validation}/
    ├── 01.원천데이터/{TS*,VS*}/{A,B,C,D}/<그룹>/<음식코드>/<샘플번호>/{측면,정위,단면}/*.jpg
    └── 02.라벨링데이터/{TL*,VL*}/                                                    /*.json  (jpg와 1:1 대응)
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

DEFAULT_296_ROOT = (
    "/HDD/food_data/71564/296.비전영역,_음식이미지_및_정보소개_텍스트_데이터"
    "/01-1.정식개방데이터"
)
DEFAULT_MANIFEST_DIR = Path(__file__).parent / "data"

VIEW_SIDE = "측면"
VIEW_TOP = "정위"
VIEW_CROSS = "단면"

_SHARD_IMG_RE = re.compile(r"^([TV])L(\d+)$")  # TL3 -> TS3, VL2 -> VS2


def _label_to_image_path(json_path: Path, root: Path) -> Path:
    """라벨 JSON 경로에서 짝이 되는 원천 이미지 경로를 문자열 치환으로 유도한다.

    두 트리는 '02.라벨링데이터'/'01.원천데이터'와 'TL*'/'TS*'(또는 'VL*'/'VS*') 외에는
    완전히 동일한 하위 경로를 쓰고, 파일 개수·이름도 1:1로 대응한다 (사전 확인됨).
    """
    rel_parts = list(json_path.relative_to(root).parts)
    # rel_parts = [split, '02.라벨링데이터', 'TL1', 'A', '14', 'A14050', '01', '측면', 'xxx.json']
    rel_parts[1] = "01.원천데이터"
    m = _SHARD_IMG_RE.match(rel_parts[2])
    if not m:
        raise ValueError(f"Unexpected shard folder name: {rel_parts[2]} ({json_path})")
    rel_parts[2] = f"{m.group(1)}S{m.group(2)}"
    rel_parts[-1] = rel_parts[-1][:-len(".json")] + ".jpg"
    return root.joinpath(*rel_parts)


def _parse_label_file(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text(encoding="utf-8"))["data"]
    info = payload.get("image_info", {})
    nutr = payload.get("nutrition", {})
    ann = payload.get("2d_annotation", {})
    return {
        "weight": info.get("weight"),
        "s_weight": info.get("s_weight"),
        "img_width": info.get("width"),
        "img_height": info.get("height"),
        "g": nutr.get("g"),
        "e": nutr.get("e"),
        "fat": nutr.get("fat"),
        "pro": nutr.get("pro"),
        "sug": nutr.get("sug"),
        "bbox_x": ann.get("x"),
        "bbox_y": ann.get("y"),
        "bbox_w": ann.get("width"),
        "bbox_h": ann.get("height"),
    }


def build_manifest(root=DEFAULT_296_ROOT, split="Training", rebuild=False,
                    manifest_dir=DEFAULT_MANIFEST_DIR) -> pd.DataFrame:
    """<split>의 전체 라벨 트리를 스캔해 매니페스트를 만든다 (모든 view 포함, 캐시됨).

    라벨 JSON 18만~2만여 개를 매번 파싱하면 수 분이 걸리므로 parquet로 캐시한다.
    """
    root = Path(root)
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    cache_path = manifest_dir / f"296_manifest_{split.lower()}.parquet"
    if cache_path.exists() and not rebuild:
        return pd.read_parquet(cache_path)

    label_root = root / split / "02.라벨링데이터"
    if not label_root.exists():
        raise FileNotFoundError(f"296 label root not found: {label_root}")

    rows = []
    for dirpath in sorted(p for p in label_root.rglob("*") if p.is_dir()):
        json_files = sorted(dirpath.glob("*.json"))
        if not json_files:
            continue
        view_dir = dirpath.name          # 측면 / 정위 / 단면
        sample_id = dirpath.parent.name  # 01, 02, ...
        food_code = dirpath.parent.parent.name  # A14050
        for json_path in json_files:
            parsed = _parse_label_file(json_path)
            image_path = _label_to_image_path(json_path, root)
            stem_parts = json_path.stem.split("_")
            food_name = stem_parts[3] if len(stem_parts) > 3 else food_code
            rows.append({
                "image_path": str(image_path),
                "json_path": str(json_path),
                "food_code": food_code,
                "food_name": food_name,
                "sample_id": sample_id,
                "view": view_dir,
                **parsed,
            })

    df = pd.DataFrame(rows)
    df.to_parquet(cache_path, index=False)
    print(f"[dataset_296] built manifest: split={split} rows={len(df)} -> {cache_path}")
    return df


def _with_100g_cols(df: pd.DataFrame) -> pd.DataFrame:
    """100g당 값 + Atwater 역산 탄수화물(carb100) 컬럼을 붙인다.

    e/fat/pro/g는 음식코드별 상수(표준 1인분 기준)라서, 같은 food_code면 100g당 값도 항상 동일하다.
    """
    df = df.copy()
    df["e100"] = df["e"] / df["g"] * 100.0
    df["fat100"] = df["fat"] / df["g"] * 100.0
    df["pro100"] = df["pro"] / df["g"] * 100.0
    df["carb100"] = (df["e100"] - 9 * df["fat100"] - 4 * df["pro100"]) / 4.0
    return df


def _iqr_keep_codes(df: pd.DataFrame, cols=("e100", "fat100", "pro100", "carb100")) -> set:
    """음식코드 단위 100g당 값에 IQR×3을 적용해 살아남는 food_code 집합을 반환한다.

    text_baseline_the_best/train.py의 _iqr_filter와 같은 규칙(IQR×3, 100g당 값 기준)이지만,
    296은 행 단위가 아니라 food_code 단위로 걸러야 한다 (한 음식코드 = 여러 샘플/뷰 행).
    """
    coded = _with_100g_cols(df).drop_duplicates("food_code")[["food_code", *cols]]
    mask = pd.Series(True, index=coded.index)
    for col in cols:
        q1, q3 = coded[col].quantile(0.25), coded[col].quantile(0.75)
        iqr = q3 - q1
        mask &= coded[col].between(max(0.0, q1 - 3 * iqr), q3 + 3 * iqr)
    return set(coded.loc[mask, "food_code"])


def load_split(root=DEFAULT_296_ROOT, view=VIEW_SIDE, rebuild=False,
               manifest_dir=DEFAULT_MANIFEST_DIR) -> dict:
    """Training/Validation 매니페스트를 읽어 필터링 후 {train, val, test} DataFrame으로 나눈다.

    - IQR 허용 음식코드 집합은 Training에서만 계산한다 (100g당 영양값은 음식코드별 상수이므로
      Validation에서 같은 코드를 다시 계산해도 값이 같다 — Training이 800종 전체를 포함하는 상위집합).
    - weight(실측 중량) 결측 행은 양쪽에서 각각 제거한다 (샘플별 실측값이라 행 단위로 다름).
    - test = Training에는 있지만 Validation 폴더 자체에 없는 음식코드 (구조적 train-only),
      그중 weight 보유 + IQR 통과분만 남긴다.
    - train/val = 두 폴더 모두에 있고, 두 split 모두에서 weight 보유 + IQR 통과하는 음식코드만 남긴다
      (한쪽에서만 살아남으면 평가/학습 중 하나가 불가능해지므로 함께 제외한다).
    """
    train_all = build_manifest(root, "Training", rebuild=rebuild, manifest_dir=manifest_dir)
    val_all = build_manifest(root, "Validation", rebuild=rebuild, manifest_dir=manifest_dir)

    train_view = train_all[train_all["view"] == view].reset_index(drop=True)
    val_view = val_all[val_all["view"] == view].reset_index(drop=True)

    keep_codes = _iqr_keep_codes(train_view)

    train_struct_codes = set(train_view["food_code"])
    val_struct_codes = set(val_view["food_code"])
    test_codes = train_struct_codes - val_struct_codes  # 구조적 train-only 음식

    train_f = _with_100g_cols(train_view)
    val_f = _with_100g_cols(val_view)
    train_f = train_f[train_f["weight"].notna() & train_f["food_code"].isin(keep_codes)]
    val_f = val_f[val_f["weight"].notna() & val_f["food_code"].isin(keep_codes)]

    test_df = train_f[train_f["food_code"].isin(test_codes)].reset_index(drop=True)

    common_codes = (set(train_f["food_code"]) & set(val_f["food_code"])) - test_codes
    train_df = train_f[train_f["food_code"].isin(common_codes)].reset_index(drop=True)
    val_df = val_f[val_f["food_code"].isin(common_codes)].reset_index(drop=True)

    assert set(train_df["food_code"]).isdisjoint(set(test_df["food_code"])), \
        "train/test food_code leak"
    assert set(val_df["food_code"]) <= common_codes and set(train_df["food_code"]) == common_codes

    return {"train": train_df, "val": val_df, "test": test_df}


def _print_stats(splits: dict):
    for name, df in splits.items():
        n_codes = df["food_code"].nunique()
        n_names = df["food_name"].nunique()
        print(f"  {name:5s}: rows={len(df):>7,d}  food_codes={n_codes:>4d}  food_names={n_names:>4d}")
    print(f"\n  train ∩ test food_code overlap: "
          f"{len(set(splits['train']['food_code']) & set(splits['test']['food_code']))}")
    print(f"  test food codes: {sorted(splits['test']['food_code'].unique())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_296_ROOT)
    parser.add_argument("--view", default=VIEW_SIDE)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    splits = load_split(root=args.root, view=args.view, rebuild=args.rebuild_manifest)
    if args.stats or True:
        _print_stats(splits)
