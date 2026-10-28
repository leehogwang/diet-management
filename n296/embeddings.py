"""텍스트 → Gemini 임베딩 (3072차원) + JSON 캐시.

캐시 형식: {"<텍스트>": [float, ...]}  (원본 config 의 emb_cache.json 과 같은 방식으로 가정)

사용 예
  export GEMINI_KEY=...
  python -m n296.embeddings --config configs/stage3_296.yaml                     # manifest 의 모든 텍스트
  python -m n296.embeddings --config configs/stage3_296.yaml --provider fake     # API 없이 스모크 테스트용
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import load_config, resolve


def render_text(row: dict | pd.Series, template: str) -> str:
    return template.format(**{k: ("" if pd.isna(v) else v) for k, v in dict(row).items()}).strip()


def load_cache(path: Path) -> dict[str, list[float]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def fake_embedding(text: str, dim: int) -> list[float]:
    """텍스트 해시로 만든 고정 난수 벡터. 파이프라인 동작 확인용이며 실제 학습에 쓰면 안 된다."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def gemini_embed(texts: list[str], cfg: dict) -> list[list[float]]:
    from google import genai
    from google.genai import types

    key = os.environ.get(cfg.get("api_key_env", "GEMINI_KEY")) or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit(f"API 키가 없습니다. export {cfg.get('api_key_env', 'GEMINI_KEY')}=... 후 다시 실행하세요.")
    client = genai.Client(api_key=key)
    for attempt in range(5):
        try:
            # 문자열 리스트를 그대로 넘기면 SDK 가 하나의 콘텐츠로 합쳐 임베딩 1개만 돌려줄 수 있다.
            # 텍스트마다 별도 Content 로 감싸 텍스트 수만큼 임베딩을 받는다.
            resp = client.models.embed_content(
                model=cfg["model"],
                contents=[types.Content(parts=[types.Part(text=t)]) for t in texts],
                config=types.EmbedContentConfig(
                    task_type=cfg.get("task_type"),
                    output_dimensionality=cfg["dim"],
                ),
            )
            vecs = [e.values for e in resp.embeddings]
            if len(vecs) != len(texts):
                raise SystemExit(f"임베딩 {len(vecs)}개 ≠ 요청 텍스트 {len(texts)}개 — 캐시에 저장하지 않음")
            if len(vecs[0]) != cfg["dim"]:
                raise SystemExit(f"임베딩 차원 {len(vecs[0])} ≠ config dim {cfg['dim']}")
            return vecs
        except SystemExit:
            raise
        except Exception as e:  # rate limit 등
            wait = 2 ** attempt * 5
            print(f"  Gemini 오류({e}); {wait}s 후 재시도")
            time.sleep(wait)
    raise RuntimeError("Gemini 임베딩 실패")


def ensure_embeddings(texts: list[str], cache_path: Path, emb_cfg: dict, provider: str | None = None) -> dict:
    provider = provider or emb_cfg.get("provider", "gemini")
    cache = load_cache(cache_path)
    todo = sorted(set(t for t in texts if t and t not in cache))
    print(f"텍스트 {len(set(texts))}종 중 새로 임베딩할 것 {len(todo)}종 (provider={provider})")
    bs = emb_cfg.get("batch_size", 50)
    for i in tqdm(range(0, len(todo), bs), desc="embed", disable=not todo):
        chunk = todo[i:i + bs]
        vecs = [fake_embedding(t, emb_cfg["dim"]) for t in chunk] if provider == "fake" else gemini_embed(chunk, emb_cfg)
        cache.update(zip(chunk, vecs))
        save_cache(cache, cache_path)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", nargs="*", default=None, help="기본값: config 의 manifest + text_only_manifest")
    ap.add_argument("--provider", choices=["gemini", "fake"], default=None)
    ap.add_argument("--cache", default=None, help="기본값: config paths.emb_cache")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = args.manifest or [p for p in (cfg["paths"]["manifest"], cfg["paths"].get("text_only_manifest")) if p]
    texts = []
    for p in paths:
        p = resolve(p)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        texts += [render_text(r, cfg["data"]["text_template"]) for _, r in df.iterrows()]
    ensure_embeddings(texts, resolve(args.cache or cfg["paths"]["emb_cache"]), cfg["embedding"], args.provider)


if __name__ == "__main__":
    main()
