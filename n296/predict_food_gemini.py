"""음식 사진 → Gemini 음식명·재료 예측 (힌트·후보 목록 없이 zero-shot).

test_dataset_296 의 모든 이미지를 Gemini 에 넣고 음식명과 재료만 예측시킨다. 채점은 하지 않는다.
정답 컬럼(food_code, gt_food_name)은 나중에 평가할 때 조인하기 편하도록 결과에 같이 적지만
프롬프트에는 넣지 않는다.

출력 (--out-dir)
  predictions_raw.jsonl  이미지 1장당 1줄, 중간에 끊겨도 이어서 실행 가능
  predictions.json       manifest 순서로 정리한 전체 결과
  predictions.csv        같은 내용, ingredients 는 " | " 로 이어 붙임

사용 예
  export GEMINI_KEY=...
  python -m n296.predict_food_gemini                  # 전체 2664장
  python -m n296.predict_food_gemini --limit 5        # 앞 5장만 시험
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/home/students/cs/202321024/datasets/test_dataset_296")
DEFAULT_MODEL = "gemini-3.1-flash-lite"  # recipe_pipeline/pipeline_run.py Step 1 과 같은 모델

PROMPT = (
    "이 사진 속 음식의 이름과 재료를 예측하세요.\n"
    "- food_name: 음식 이름 (한국어)\n"
    "- ingredients: 이 음식에 들어간 재료 목록 (한국어)"
)

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "food_name": {"type": "STRING"},
        "ingredients": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["food_name", "ingredients"],
}


def get_api_key(key_file: str | None) -> str:
    if key_file:
        text = Path(key_file).expanduser().read_text().strip()
        return text.split("=", 1)[1].strip().strip("'\"") if "=" in text else text  # "NAME=키" 형식 허용
    for name in ("GEMINI_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    raise SystemExit("API 키가 없습니다. export GEMINI_KEY=... 또는 --key-file 로 전달하세요.")


class RateLimiter:
    """스레드 공용 요청 간격 제한. 무료 등급은 분당 15회(RPM)라 넘기면 429 가 난다."""

    def __init__(self, rpm: float | None):
        self.interval = 60.0 / rpm if rpm else 0.0
        self.next_t = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        time.sleep(max(0.0, t - now))


def gemini_json(client, model: str, image_bytes: bytes | None, prompt: str, schema: dict, temperature: float,
                max_retries: int, limiter: RateLimiter) -> tuple[dict | None, str | None, str | None]:
    """(이미지 +) 프롬프트 → JSON 응답. (parsed, raw, error) 를 돌려준다. 429·파싱 오류는 재시도, 일일 한도는 즉시 포기.
    image_bytes 가 None 이면 텍스트만 보낸다."""
    from google.genai import types

    contents = [prompt] if image_bytes is None else [types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt]
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            limiter.wait()
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            return json.loads(resp.text), resp.text, None
        except Exception as e:  # noqa: BLE001 — API/파싱 오류 모두 재시도
            last_err = f"{type(e).__name__}: {e}"
            if is_daily_quota(last_err):  # 일일 한도는 재시도해도 소용없음
                break
            time.sleep(30 if "429" in last_err else min(60, 2 ** (attempt + 1)))
    return None, None, last_err


def clean_list(items) -> list[str]:
    return [str(x).strip() for x in (items or []) if str(x).strip()]


def predict_one(client, model: str, image_path: Path, temperature: float, max_retries: int,
                limiter: RateLimiter) -> dict:
    parsed, raw, err = gemini_json(client, model, image_path.read_bytes(), PROMPT, RESPONSE_SCHEMA,
                                   temperature, max_retries, limiter)
    if parsed is None:
        return {"pred_food_name": None, "pred_ingredients": [], "raw_response": None, "error": err}
    return {
        "pred_food_name": str(parsed.get("food_name", "")).strip(),
        "pred_ingredients": clean_list(parsed.get("ingredients")),
        "raw_response": raw,
        "error": None,
    }


def is_daily_quota(err: str | None) -> bool:
    return bool(err) and "429" in err and "PerDay" in err


def load_done(raw_path: Path) -> dict[str, dict]:
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("error") is None:  # 실패한 건 다시 시도
                    done[rec["image"]] = rec
    return done


def write_outputs(manifest: pd.DataFrame, done: dict[str, dict], out_dir: Path) -> list[dict]:
    records = [done[img] for img in manifest["image"] if img in done]
    (out_dir / "predictions.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    df = pd.DataFrame(records)
    if not df.empty:
        df["pred_ingredients"] = df["pred_ingredients"].map(" | ".join)
        df = df.drop(columns=["raw_response"])
    df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATASET / "test_manifest.csv"))
    ap.add_argument("--images", default=str(DATASET / "images"))
    ap.add_argument("--out-dir", default=str(ROOT / "gemini_food_pred"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--rpm", type=float, default=None, help="분당 최대 요청 수 (무료 등급: 15)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--timeout", type=float, default=90, help="요청 1건 타임아웃(초)")
    args = ap.parse_args()

    from google import genai

    manifest = pd.read_csv(args.manifest)
    manifest["image"] = manifest["image_path"].map(os.path.basename)
    if args.limit:
        manifest = manifest.head(args.limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "predictions_raw.jsonl"
    done = load_done(raw_path)
    todo = manifest[~manifest["image"].isin(done)]
    print(f"전체 {len(manifest)}장, 완료 {len(manifest) - len(todo)}장, 남은 {len(todo)}장 (model={args.model})")

    from google.genai import types

    # 타임아웃이 없으면 응답 없는 요청에서 워커가 영원히 멈춘다 (단위: ms)
    client = genai.Client(api_key=get_api_key(args.key_file),
                          http_options=types.HttpOptions(timeout=int(args.timeout * 1000)))
    lock = threading.Lock()
    limiter = RateLimiter(args.rpm)

    def job(row) -> dict:
        pred = predict_one(client, args.model, Path(args.images) / row.image, args.temperature, args.max_retries, limiter)
        return {
            "image": row.image,
            "sample_id": int(row.sample_id),
            "food_code": row.food_code,
            "gt_food_name": row.food_name,
            **pred,
            "model": args.model,
        }

    stop = threading.Event()

    def job_or_skip(row) -> dict | None:
        return None if stop.is_set() else job(row)

    n_err = 0
    with open(raw_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(job_or_skip, row) for row in todo.itertuples(index=False)]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            rec = fut.result()
            if rec is None:
                continue
            if is_daily_quota(rec["error"]):
                # 일일 한도 도달: 남은 작업은 job_or_skip 이 바로 None 을 돌려 빠르게 끝난다.
                # (cancel_futures 로 취소하면 as_completed 가 끝나지 않아 멈춘다.)
                # 이 이미지는 기록하지 않아 다음 실행에서 다시 처리된다.
                if not stop.is_set():
                    print(f"\n일일 한도 도달, 중지합니다: {rec['error'][:200]}")
                stop.set()
                continue
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            if rec["error"] is None:
                done[rec["image"]] = rec
            else:
                n_err += 1

    records = write_outputs(manifest, done, out_dir)
    print(f"저장: {out_dir}  (성공 {len(records)}/{len(manifest)}, 이번 실행 실패 {n_err})")
    if stop.is_set():
        print("일일 한도로 중지했습니다. 한도가 초기화된 뒤 같은 명령을 실행하면 이어서 진행합니다.")
    if n_err:
        print("실패한 이미지는 같은 명령을 다시 실행하면 재시도합니다.")


if __name__ == "__main__":
    main()
