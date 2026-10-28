"""UniDepth V2 (ViT-L) 깊이 사전계산 — UniDepth 전용 환경(/mnt/data/food_leehogwang/.venv_unidepth)에서 실행.

원본 전체 이미지로 깊이를 추정한 뒤(주변 맥락이 있어야 metric 깊이가 정확), 학습 데이터셋과 같은 bbox 로 자르고
ResizeWithPadding 과 같은 방식(비율 유지 + 가운데 정렬, 패딩=0)으로 112x112 로 줄여 저장.

usage: python precompute_depth.py --shard 0 --nshards 2 --device cuda:0
       python precompute_depth.py --merge      (모든 shard 완료 후 depth_all.npy 로 병합)
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
OUT = Path("/mnt/data/food_leehogwang/depth_cache_unidepthv2")
SIZE, MAX_SIDE, CHUNK = 112, 1024, 2000


def read_zip(path, fname):
    with zipfile.ZipFile(path) as z:
        try:
            return z.read(fname)
        except KeyError:  # 데이터셋 로더와 같은 파일명 인코딩 대체 규칙
            try:
                return z.read(fname.encode("cp949").decode("cp437", errors="ignore"))
            except KeyError:
                suffix = fname.split("_")[-1]
                m = [n for n in z.namelist() if n.endswith(suffix)]
                return z.read(m[0])


class Images(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            if r["kind"] == "zip":  # AIHub 122 로더는 exif 회전 후 bbox 를 적용
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(read_zip(r["path"], r["filename"])))).convert("RGB")
            else:  # OmniFood / 369종 로더는 exif 회전 없이 사용
                img = Image.open(r["path"]).convert("RGB")
        except Exception as e:
            return {"i": i, "ok": False, "err": repr(e)[:200]}
        W, H = img.size
        crop = (0, 0, W, H)
        b = r.get("bbox")
        if b is not None and len(b) == 4:  # 데이터셋과 같은 유효성 검사
            x1, y1, x2, y2 = b
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= W + 10 and y2 <= H + 10:
                crop = (x1, y1, min(x2, W), min(y2, H))
        s = min(1.0, MAX_SIDE / max(W, H))
        if s < 1:
            img = img.resize((round(W * s), round(H * s)), Image.Resampling.BILINEAR)
        rgb = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)  # uint8 [3,h,w]
        return {"i": i, "ok": True, "rgb": rgb, "crop": torch.tensor([c * s for c in crop]), "long": float(max(W, H))}


def pad_resize(d):
    """d: [h,w] → [SIZE,SIZE], 비율 유지 + 가운데 정렬, 패딩 0. 반환: (맵, 축소 배율)"""
    h, w = d.shape
    sc = SIZE / max(h, w)
    nh, nw = max(1, round(h * sc)), max(1, round(w * sc))
    r = F.interpolate(d[None, None], size=(nh, nw), mode="bilinear", align_corners=False)[0, 0]
    out = torch.zeros(SIZE, SIZE, dtype=d.dtype, device=d.device)
    y0, x0 = (SIZE - nh) // 2, (SIZE - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out, sc


def run_shard(args):
    from unidepth.models import UniDepthV2

    rows = [json.loads(l) for l in (HERE / "depth_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = rows[args.shard::args.nshards]
    OUT.mkdir(parents=True, exist_ok=True)
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(args.device).eval()
    for c0 in range(0, len(rows), CHUNK):
        fn = OUT / f"shard{args.shard}_chunk{c0 // CHUNK:04d}.npz"
        if fn.exists():
            continue
        part = rows[c0:c0 + CHUNK]
        dep = np.zeros((len(part), SIZE, SIZE), np.float16)
        meta = np.zeros((len(part), 2), np.float32)
        ok = np.zeros(len(part), bool)
        dl = DataLoader(Images(part), batch_size=None, shuffle=False, num_workers=args.workers)
        for x in dl:
            i = x["i"]
            if not x["ok"]:
                print(f"[skip] {part[i]['key']} {x['err']}", flush=True)
                continue
            with torch.inference_mode():
                p = model.infer(x["rgb"].to(args.device))
            d = p["depth"][0, 0].float()  # [h,w] 미터, 입력 해상도
            fx = float(p["intrinsics"][0, 0, 0])
            x1, y1, x2, y2 = [int(round(v)) for v in x["crop"].tolist()]
            d = d[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
            m, sc = pad_resize(d)
            dep[i] = m.cpu().numpy().astype(np.float16)
            meta[i] = (fx * sc, x["long"])
            ok[i] = True
        np.savez(fn, keys=np.array([r["key"] for r in part], dtype=object), depth=dep, meta=meta, ok=ok)
        print(f"[saved] {fn.name} ok={ok.sum()}/{len(part)}", flush=True)
    print(f"SHARD {args.shard} DONE", flush=True)


def merge():
    files = sorted(OUT.glob("shard*_chunk*.npz"))
    keys, deps, metas = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        k = z["ok"]
        keys += list(z["keys"][k])
        deps.append(z["depth"][k])
        metas.append(z["meta"][k])
    np.save(OUT / "depth_all.npy", np.concatenate(deps))
    np.save(OUT / "depth_meta.npy", np.concatenate(metas))
    (OUT / "depth_keys.json").write_text(json.dumps([str(k) for k in keys], ensure_ascii=False), encoding="utf-8")
    print(f"MERGED {len(keys)} depth maps from {len(files)} chunks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else run_shard(a)
