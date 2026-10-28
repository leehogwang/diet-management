"""깊이 사전계산 대상 이미지 목록 생성 (학습용 .venv 에서 실행). 기존 데이터 로더를 읽기 전용으로 사용.

출력: depth/depth_manifest.jsonl — 한 줄에 {key, kind(file|zip), path, filename, bbox}
key 는 depth/depth_data.py 의 record_key() 와 동일 규칙.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_baseline_the_best"))

from dataset_omnifood8k import load_omnifood_dataset
from dataset_aihub122_classification_nutrition import load_122_classification_dataset
from dataset_aihub122_multimodal import load_aihub122_dataset
from depth.depth_data import record_key


def main():
    out = Path(__file__).resolve().parent / "depth_manifest.jsonl"
    rows = {}
    so, _ = load_omnifood_dataset()
    s2, _, _ = load_122_classification_dataset()
    s3, _ = load_aihub122_dataset(emb_cache_path=ROOT / "text_baseline_the_best/emb_cache_with_ingredients.json")
    for name, splits in (("omnifood", so), ("std369", s2), ("aihub122q", s3)):
        n = 0
        for recs in splits.values():
            for r in recs:
                k = record_key(r)
                if k in rows:
                    continue
                if "zip_path" in r:
                    rows[k] = {"key": k, "kind": "zip", "path": r["zip_path"], "filename": r["filename"], "bbox": r.get("bbox")}
                else:
                    rows[k] = {"key": k, "kind": "file", "path": r["image_path"], "filename": None, "bbox": None}
                n += 1
        print(f"[manifest] {name}: {n} images")
    with open(out, "w", encoding="utf-8") as f:
        for row in rows.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[manifest] total {len(rows)} -> {out}")


if __name__ == "__main__":
    main()
