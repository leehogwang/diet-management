"""side_angles 프레임 인덱스 구축 (2026-08-22).

## 왜
라벨은 있는데 overhead RGB-D 가 없어 **한 번도 안 쓴 접시가 1,494장** 있다(train 1,156 / val 136 / test 202).
전부 side_angles(카메라 A~D × 약 29프레임)를 보유한다. 현재 학습이 쓰는 접시는 2,485장뿐이므로
train 만 +1,156(+47%) 늘릴 수 있다.

## 무엇을 만드나
접시 -> 사용 가능한 프레임 경로 목록. 학습 때 매 에폭 무작위로 한 장을 뽑아 쓰는 증강이 된다
(같은 접시의 회전 프레임 ~115장이 자연스러운 뷰 증강).

## 주의 — 누수 차단
test 709접시도 side 를 갖고 있으나 **사전학습에 절대 넣지 않는다.** 인덱스에는 split 을 함께
기록하고, 학습 스크립트가 train split 만 쓰도록 강제한다.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/students/cs/202520906/nutrition5k_pipeline")
sys.path.insert(0, str(ROOT))
SIDE = Path("/home/students/cs/202520906/nutrition5k/imagery/side_angles")
OV = Path("/home/students/cs/202520906/nutrition5k/imagery/realsense_overhead")
OUT = ROOT / "outputs/task_runs/task_side/side_frames.json"


def main():
    recs = {}
    for sp in ["train", "val", "test"]:
        for line in (ROOT / f"outputs/manifests/{sp}.jsonl").open():
            it = json.loads(line)
            recs.setdefault(it["dish_id"], sp)

    idx, n_frames, stats = {}, 0, {"train": 0, "val": 0, "test": 0}
    only_side = {"train": 0, "val": 0, "test": 0}
    for i, (d, sp) in enumerate(sorted(recs.items())):
        fd = SIDE / d / "frames_sampled5"
        if not fd.is_dir():
            continue
        fr = sorted(f for f in os.listdir(fd) if f.endswith(".jpeg"))
        if not fr:
            continue
        has_ov = (OV / d / "rgb.png").exists() and (OV / d / "depth_raw.png").exists()
        idx[d] = {"split": sp, "n_frames": len(fr), "frames": fr, "has_overhead": has_ov}
        n_frames += len(fr); stats[sp] += 1
        if not has_ov:
            only_side[sp] += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"root": str(SIDE), "sub": "frames_sampled5",
                               "dishes": idx}, ensure_ascii=False))
    print(f"\n라벨 접시 {len(recs)} 중 side 보유 {len(idx)}  총 프레임 {n_frames:,}")
    for sp in ["train", "val", "test"]:
        print(f"  {sp:5s} side 보유 {stats[sp]:4d}  그중 overhead 없는 접시(신규) {only_side[sp]:4d}")
    print(f"\n★ 사전학습에 쓸 수 있는 것 = train split {stats['train']}접시"
          f" (현행 학습 2,485 대비 +{stats['train']-2485})")
    print(f"★ test {stats['test']}접시는 인덱스에만 있고 학습에는 절대 쓰지 않는다")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
