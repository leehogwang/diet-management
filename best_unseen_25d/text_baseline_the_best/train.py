"""
Text-only 100g nutrition regression baseline.
음식명 -> GeminiEmbedding(OpenRouter) -> MLP -> (cal, fat, carb, prot) per 100g
학습: log1p 공간 MSE, 평가: 원래 공간 PMAE
"""
import argparse, copy, json, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


# ── Data ──────────────────────────────────────────────────────────────────────

def _to_100g(df: pd.DataFrame, name_col: str, serving_col: str,
             kcal_col: str, fat_col: str, carb_col: str, prot_col: str) -> pd.DataFrame:
    """공통 100g 변환: 결과는 [식품명, cal, fat, carb, prot] 컬럼."""
    for col in [kcal_col, fat_col, carb_col, prot_col, serving_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[name_col, kcal_col, fat_col, carb_col, prot_col, serving_col])
    df = df[df[serving_col] > 0].copy()
    df["식품명"] = df[name_col].str.strip()
    df = df[~df["식품명"].str.contains("&")]
    df["cal"]   = df[kcal_col]  / df[serving_col] * 100
    df["fat"]   = df[fat_col]   / df[serving_col] * 100
    df["carb"]  = df[carb_col]  / df[serving_col] * 100
    df["prot"]  = df[prot_col]  / df[serving_col] * 100
    df["mass"]  = df[serving_col]  # 1회제공량(g) — 절대량 타깃 복원용 (100g 변환과 별개로 보존)
    # 동일 DB 내 중복 식품명: 평균 대신 첫 등장만 사용, 이후는 skip
    return df.drop_duplicates(subset="식품명", keep="first")[["식품명", "cal", "fat", "carb", "prot", "mass"]].reset_index(drop=True)


def _iqr_filter(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)
    for col in ["cal", "fat", "carb", "prot"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask &= df[col].between(max(0.0, q1 - 3 * iqr), q3 + 3 * iqr)
    return df[mask].reset_index(drop=True)


def load_nutrition_db(xlsx_path: str, db_category: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, header=3)
    df = df[df["DB군"] == db_category].copy()
    return _to_100g(df, "식품명", "1회제공량", "에너지(㎉)", "지방(g)", "탄수화물(g)", "단백질(g)")


def load_nutrition_db_122(xlsx_path: str) -> pd.DataFrame:
    """44.음식분류_AI_데이터_영양DB.xlsx — 외식메뉴 400개, 컬럼명 상이."""
    df = pd.read_excel(xlsx_path, header=0)
    return _to_100g(df, "음 식 명", "중량(g)", "에너지(kcal)", "지방(g)", "탄수화물(g)", "단백질(g)")


def load_nutrition_db_processed(xlsx_path: str, processed_names_path: str) -> pd.DataFrame:
    """DB군='가공식품': processed_names.json으로 원본명→정규화명 매핑, 'x'는 제외."""
    df = pd.read_excel(xlsx_path, header=3)
    df = df[df["DB군"] == "가공식품"].copy()
    name_map = json.loads(Path(processed_names_path).read_text())
    df["식품명"] = df["식품명"].astype(str).str.strip().map(name_map)
    df = df[df["식품명"].notna() & (df["식품명"] != "x")]
    return _to_100g(df, "식품명", "1회제공량", "에너지(㎉)", "지방(g)", "탄수화물(g)", "단백질(g)")


def load_nutrition_db_std(csv_path: str, carb_max: float = 2.0) -> pd.DataFrame:
    """전국통합식품영양성분정보_음식_표준데이터.csv — 값이 이미 100g(ml) 기준이라 환산 불필요.
    탄수화물 0~carb_max(g)인 저탄수 음식만 선택 (고단백저탄수 구간 보강용)."""
    df = pd.read_csv(csv_path, encoding="cp949", low_memory=False)
    for col in ["에너지(kcal)", "지방(g)", "탄수화물(g)", "단백질(g)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["식품명", "에너지(kcal)", "지방(g)", "탄수화물(g)", "단백질(g)", "식품중량"])
    df = df[df["탄수화물(g)"].between(0, carb_max)].copy()
    df["식품명"] = df["식품명"].astype(str).str.strip()
    df = df[~df["식품명"].str.contains("&")]
    df["mass"] = pd.to_numeric(
        df["식품중량"].astype(str).str.replace("g", "", regex=False).str.replace("ml", "", regex=False),
        errors="coerce",
    )
    df = df.dropna(subset=["mass"])
    df = df[df["mass"] > 0]
    df["cal"], df["fat"], df["carb"], df["prot"] = (
        df["에너지(kcal)"], df["지방(g)"], df["탄수화물(g)"], df["단백질(g)"]
    )
    return df.drop_duplicates(subset="식품명", keep="first")[
        ["식품명", "cal", "fat", "carb", "prot", "mass"]
    ].reset_index(drop=True)


def load_combined_db(cfg_data: dict) -> pd.DataFrame:
    # DB2(xlsx_path_122, 122 이미지 데이터의 원 출처) 우선 — 겹치는 식품명은 DB1/DB3/DB4에서 skip (평균 내지 않음)
    df1 = load_nutrition_db(cfg_data["xlsx_path"], cfg_data["db_category"])
    frames = []
    if cfg_data.get("xlsx_path_122"):
        df2 = load_nutrition_db_122(cfg_data["xlsx_path_122"])
        frames.append(df2)
        print(f"[data] DB1={len(df1)}, DB2={len(df2)}")
    frames.append(df1)
    if cfg_data.get("processed_names_path"):
        df3 = load_nutrition_db_processed(cfg_data["xlsx_path"], cfg_data["processed_names_path"])
        frames.append(df3)
        print(f"[data] DB3(가공식품)={len(df3)}")
    if cfg_data.get("csv_path_std"):
        df4 = load_nutrition_db_std(cfg_data["csv_path_std"], cfg_data.get("std_carb_max", 2.0))
        frames.append(df4)
        print(f"[data] DB4(표준데이터 저탄수 carb<={cfg_data.get('std_carb_max', 2.0)})={len(df4)}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="식품명", keep="first").reset_index(drop=True)
    before = len(df)
    df = _iqr_filter(df)
    print(f"[data] combined {len(df)} unique food names ({before - len(df)} outliers removed by IQR×3)")
    return df


def split_df(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int):
    n = len(df)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    has_english = df["식품명"].str.contains(r"[a-zA-Z]").to_numpy()
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)

    n_non_english = int((~has_english).sum())
    if n_non_english < n_test:
        print(f"[split] WARNING: 영어 포함 이름 제외 시 test 후보가 {n_non_english}개뿐 "
              f"(목표 {n_test}개) — test 세트가 목표보다 작아집니다.")

    test_idx, rest_idx = [], []
    for i in idx:
        if len(test_idx) < n_test and not has_english[i]:
            test_idx.append(i)
        else:
            rest_idx.append(i)
    val_idx   = rest_idx[:n_val]
    train_idx = rest_idx[n_val:]
    return (df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True))


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_embeddings(names: list[str], model_name: str, cache_path: str, api_key_env: str = "GEMINI_KEY") -> np.ndarray:
    cache_file = Path(cache_path)
    cache: dict = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    missing = [n for n in names if n not in cache]
    if missing:
        api_key = os.environ[api_key_env]
        print(f"[emb] calling API ({api_key_env}) for {len(missing)} names...")
        if api_key_env == "OEPNROUTE_KEY":
            import requests
            url = "https://openrouter.ai/api/v1/embeddings"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            for i in range(0, len(missing), 100):
                batch = missing[i:i+100]
                resp = requests.post(url, json={"model": model_name, "input": batch}, headers=headers)
                resp.raise_for_status()
                for item in sorted(resp.json()["data"], key=lambda x: x["index"]):
                    cache[batch[item["index"]]] = item["embedding"]
                print(f"  {min(i+100, len(missing))}/{len(missing)}")
        else:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            for i in range(0, len(missing), 100):
                batch = missing[i:i+100]
                result = genai.embed_content(model=model_name, content=batch,
                                             task_type="SEMANTIC_SIMILARITY")
                for name, emb in zip(batch, result["embedding"]):
                    cache[name] = emb
                print(f"  {min(i+100, len(missing))}/{len(missing)}")

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, ensure_ascii=False))
        print(f"[emb] cache saved -> {cache_path}")

    return np.array([cache[n] for n in names], dtype=np.float32)


# ── Synonym Augmentation ──────────────────────────────────────────────────────

SYNONYM_PROMPT = (
    '음식명: "{food_name}"\n'
    "위 이름은 실제 매장·메뉴 표기라 브랜드명, 사이즈, 온도, 상품명 같은 수식어가 섞여 있을 수 있어. "
    "이 이름이 실제로 어떤 음식/음료인지 정체를 파악해서, 영양DB에 있을 법한 "
    "간결하고 표준적인 한국어 음식명 후보를 최대 {top_k}개 뽑아줘.\n"
    "규칙:\n"
    "1) 사이즈(L, tall), 온도(ice/hot), 브랜드·상품명, 매장 한정 수식어는 제거하고 "
    "음식의 핵심 정체성만 남겨 (예: '카푸치노 tall (iced)' → '카푸치노').\n"
    "2) 이름만으로 정체를 알 수 없는 상품명이면, 실제 성분·조리법에 대한 지식으로 "
    "가장 가까운 일반 음식을 추론해 (예: '파인클렌즈' → '파인애플식초').\n"
    "3) 일반화는 딱 한 단계만: 브랜드/수식어를 뗀 핵심 재료·맛 정보(예: 치즈, 해물, 고추장)는 "
    "반드시 유지하고, '피자'·'빵'·'음식'처럼 정보량이 사라지는 최상위 카테고리까지는 절대 올라가지 마 "
    "(예: '식스 치즈 오리지널(P)' → '치즈 피자'까지만, '피자'는 금지).\n"
    "4) 이미 표준적인 이름이면 최소한만 다듬어 (예: '느타리버섯볶음' → '느타리볶음').\n"
    "5) 공백을 붙이거나 뗀 것 외에는 원래 이름과 글자가 동일한 표기는 후보로 치지 마 "
    "(예: '가자미조림' 입력에 '가자미 조림'은 무효 — 실질적으로 다른 이름만 인정).\n"
    "6) 입력한 음식명 자체와 동일한 문자열은 포함하지 마.\n"
    "7) 입력이 특정 음식을 가리키는 게 아니라 수량·포장단위·상품 라인업명 등이라 "
    "정체를 알 수 없으면(예: '간식용19개'), 후보 대신 정확히 [\"X\"]만 반환해.\n"
    "8) (7에 해당하지 않는데) 확신이 가는 동의어가 없으면 빈 리스트를 반환하고, "
    "있으면 가장 그럴듯한 순서로 나열해.\n"
    "9) {top_k}개보다 적으면 있는 만큼만 반환해.\n"
    '10) JSON 형태로만 반환해: {{"synonyms": ["이름1", ...]}}'
)


def get_synonyms(names: list[str], model: str, cache_path: str, top_k: int, api_key_env: str) -> dict:
    import google.generativeai as genai
    cache_file = Path(cache_path)
    cache: dict = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    missing = [n for n in names if n not in cache]
    if missing:
        api_key = os.environ[api_key_env]
        genai.configure(api_key=api_key)
        genai_model = genai.GenerativeModel(
            model_name=model,
            generation_config={"response_mime_type": "application/json"},
        )
        print(f"[syn] generating synonyms for {len(missing)} food names...")
        for i, name in enumerate(missing, 1):
            prompt = SYNONYM_PROMPT.format(food_name=name, top_k=top_k)
            try:
                resp = genai_model.generate_content(prompt)
                raw = json.loads(resp.text)
                raw_syns = raw.get("synonyms", [])
                if raw_syns == ["X"]:
                    syns = ["X"]
                else:
                    name_no_space = name.replace(" ", "")
                    syns = [s.strip() for s in raw_syns
                            if isinstance(s, str) and s.strip() and s.strip() != name
                            and s.strip().replace(" ", "") != name_no_space][:top_k]
            except Exception as e:
                print(f"  [syn] error for '{name}': {e} — skipping")
                syns = []
            cache[name] = syns
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, ensure_ascii=False))
            if i % 50 == 0 or i == len(missing):
                print(f"  {i}/{len(missing)}")
            time.sleep(0.5)
        print(f"[syn] cache saved -> {cache_path}")
    return cache


# ── loop overrides ────────────────────────────────────────────────────────────

def apply_overrides(base: dict, overrides: dict, quiet: bool = False) -> dict:
    """dot-path override 적용 (예: {"training.lr": 0.01}).

    모든 경로 세그먼트가 base에 이미 존재해야 한다 — 없으면 KeyError.
    오타로 잘못된 경로를 주면 조용히 새 키를 만드는 대신 즉시 실패시켜서,
    "override가 적용 안 된 채로 같은 config가 N번 도는" 사고를 막는다.
    """
    cfg = copy.deepcopy(base)
    for key_path, value in overrides.items():
        parts = str(key_path).split(".")
        node = cfg
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"loop override '{key_path}': '{part}' not found in config")
            node = node[part]
        leaf = parts[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError(f"loop override '{key_path}': key not found in config")
        old = node[leaf]
        node[leaf] = value
        if not quiet:
            if old == value:
                print(f"[loop] warning: override '{key_path}' = {value!r} unchanged (same as base)")
            else:
                print(f"[loop] override '{key_path}': {old!r} -> {value!r}")

    # read-back 검증 — 방금 적용한 값이 실제로 반영됐는지 재확인
    for key_path, value in overrides.items():
        node = cfg
        for part in str(key_path).split(".")[:-1]:
            node = node[part]
        assert node[str(key_path).split(".")[-1]] == value, \
            f"apply_overrides failed to apply '{key_path}'"
    return cfg


def _selftest_apply_overrides():
    base = {"training": {"lr": 0.001, "batch_size": 64}}
    out = apply_overrides(base, {"training.lr": 0.01}, quiet=True)
    assert out["training"]["lr"] == 0.01 and out["training"]["batch_size"] == 64
    assert base["training"]["lr"] == 0.001, "base config must not be mutated"
    try:
        apply_overrides(base, {"training.lrr": 0.01}, quiet=True)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown override path (typo protection)")


# ── Model ─────────────────────────────────────────────────────────────────────

class NutritionMLP(nn.Module):
    def __init__(self, in_dim: int, h1: int, h2: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),     nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, 4),
        )

    def forward(self, x):
        return self.net(x)


# ── Loss / Eval ───────────────────────────────────────────────────────────────

def pmae(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # +1 smoothing: fat/prot 0g 근처에서 분모 폭발 방지 (1kcal/1g 오차 = 100% 기준)
    return (pred - gt).abs() / (gt.abs() + 1.0)


def make_loader(emb: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool):
    X = torch.from_numpy(emb)
    Y = torch.from_numpy(targets.astype(np.float32))
    return DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """log1p 공간 예측 → 원래 공간으로 역변환 후 PMAE 계산"""
    model.eval()
    all_pred_log, all_gt_log = [], []
    for X, Y_log in loader:
        all_pred_log.append(model(X.to(device)).cpu())
        all_gt_log.append(Y_log)
    # 합리적 log1p 범위 [0,10] 클리핑 후 역변환 (cal 최대 ~500 → log1p≈6.2)
    pred = torch.expm1(torch.cat(all_pred_log).clamp(0, 10))
    gt   = torch.expm1(torch.cat(all_gt_log))
    per_nutrient = pmae(pred, gt).mean(0)
    names = ["cal", "fat", "carb", "prot"]
    metrics = {n: per_nutrient[i].item() * 100 for i, n in enumerate(names)}
    metrics["mean_pmae"] = per_nutrient.mean().item() * 100
    return metrics


def train(cfg: dict, tag: str = "") -> dict:
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    df = load_combined_db(cfg["data"])

    emb_cfg = cfg["embedding"]
    api_key_env = emb_cfg.get("api_key_env", "GEMINI_KEY")

    # 정체 불명(X) 음식명 판별 — split 전에 전체 데이터셋에서 걸러내야 train/val/test 모두에서 빠진다.
    aug_cfg = cfg.get("augmentation", {})
    syn_map = {}
    if aug_cfg:
        syn_map = get_synonyms(
            df["식품명"].tolist(),
            aug_cfg["model"], aug_cfg["synonym_cache"],
            aug_cfg["top_k"], api_key_env,
        )
        unidentifiable = {name for name, syns in syn_map.items() if syns == ["X"]}
        if unidentifiable:
            before = len(df)
            df = df[~df["식품명"].isin(unidentifiable)].reset_index(drop=True)
            print(f"[data] dropped {before - len(df)} rows with unidentifiable "
                  f"food name (X) — {sorted(unidentifiable)[:5]}{'...' if len(unidentifiable) > 5 else ''}")

    train_df, val_df, test_df = split_df(
        df, cfg["data"]["val_ratio"], cfg["data"]["test_ratio"], cfg["data"]["split_seed"]
    )
    print(f"[split] train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # 유사 음식명 증강 — train row 수/타깃은 그대로 두고, epoch마다 각 행의 이름을
    # 원본 ↔ synonym 중 하나로 확률적으로 바꿔치기해서 노출시킨다 (동적 word-dropout).
    train_names = train_df["식품명"].tolist()
    if aug_cfg.get("enabled"):
        all_names = set(train_names)
        for name in train_names:
            all_names.update(syn_map.get(name, []))
        get_embeddings(sorted(all_names), emb_cfg["model"], emb_cfg["cache_path"], api_key_env)  # 캐시 워밍업
        emb_cache = json.loads(Path(emb_cfg["cache_path"]).read_text())
        replace_prob = aug_cfg.get("replace_prob", 0.5)
        print(f"[aug] enabled: runtime synonym swap (replace_prob={replace_prob}), train size stays {len(train_names)}")

        def sample_train_emb() -> np.ndarray:
            names = [random.choice(syn_map[n]) if syn_map.get(n) and random.random() < replace_prob else n
                      for n in train_names]
            return np.array([emb_cache[n] for n in names], dtype=np.float32)
    else:
        train_emb_static = get_embeddings(train_names, emb_cfg["model"], emb_cfg["cache_path"], api_key_env)

        def sample_train_emb() -> np.ndarray:
            return train_emb_static

    val_emb   = get_embeddings(val_df["식품명"].tolist(),   emb_cfg["model"], emb_cfg["cache_path"], api_key_env)
    test_emb  = get_embeddings(test_df["식품명"].tolist(),  emb_cfg["model"], emb_cfg["cache_path"], api_key_env)

    target_cols = ["cal", "fat", "carb", "prot"]
    # log(1+y) 변환 — fat 0 근처에서 PMAE 폭발 방지, 스케일 안정화
    train_Y = np.log1p(train_df[target_cols].values.astype(np.float32))
    val_Y   = np.log1p(val_df[target_cols].values.astype(np.float32))
    test_Y  = np.log1p(test_df[target_cols].values.astype(np.float32))

    bs = cfg["training"]["batch_size"]
    val_loader   = make_loader(val_emb,   val_Y,   bs, shuffle=False)
    test_loader  = make_loader(test_emb,  test_Y,  bs, shuffle=False)

    in_dim = val_emb.shape[1]
    mc = cfg["model"]
    model = NutritionMLP(in_dim, mc["hidden1"], mc["hidden2"], mc["dropout"]).to(device)
    tc = cfg["training"]
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)

    best_val, patience_cnt, best_state = float("inf"), 0, None
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        train_loader = make_loader(sample_train_emb(), train_Y, bs, shuffle=True)
        for X, Y in train_loader:
            opt.zero_grad()
            loss = ((model(X.to(device)) - Y.to(device)) ** 2).mean()  # MSE in log-space
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_m = evaluate(model, val_loader, device)
        scheduler.step(val_m["mean_pmae"])

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  val mean_pmae={val_m['mean_pmae']:.2f}  "
                  f"cal={val_m['cal']:.1f}  fat={val_m['fat']:.1f}  "
                  f"carb={val_m['carb']:.1f}  prot={val_m['prot']:.1f}")

        if val_m["mean_pmae"] < best_val:
            best_val = val_m["mean_pmae"]
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= tc["patience"]:
                print(f"[early stop] epoch {epoch}, best val mean_pmae={best_val:.2f}")
                break

    model.load_state_dict(best_state)
    print("\n=== Final Test PMAE ===")
    test_m = evaluate(model, test_loader, device)
    for k, v in test_m.items():
        print(f"  {k}: {v:.2f}")

    out_path = Path(cfg["embedding"]["cache_path"]).parent / f"best_model{tag}.pth"
    torch.save({"state_dict": best_state, "config": cfg}, out_path)
    print(f"[saved] {out_path}")

    return test_m


if __name__ == "__main__":
    _selftest_apply_overrides()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="text_baseline_the_best/config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    loop = cfg.pop("loop", None)

    if not loop:
        train(cfg)
    else:
        # pre-flight: 학습을 시작하기 전에 모든 실험의 모든 override 키를 검증.
        # 오타 하나라도 있으면 어떤 학습도 돌지 않고 여기서 즉시 멈춘다.
        for i, overrides in enumerate(loop, 1):
            try:
                apply_overrides(cfg, overrides or {}, quiet=True)
            except KeyError as e:
                raise SystemExit(f"[loop] pre-flight check failed on experiment {i}/{len(loop)}: {e}")
        print(f"[loop] pre-flight check passed — {len(loop)} experiments queued\n")

        results = []
        for i, overrides in enumerate(loop, 1):
            overrides = overrides or {}
            print(f"\n{'=' * 60}\n[loop] experiment {i}/{len(loop)}  overrides={overrides}\n{'=' * 60}")
            exp_cfg = apply_overrides(cfg, overrides)
            test_m = train(exp_cfg, tag=f"_exp{i}")
            results.append((overrides, test_m["mean_pmae"]))

        print(f"\n{'=' * 60}\n[loop] summary ({len(loop)} experiments)\n{'=' * 60}")
        for i, (overrides, mean_pmae) in enumerate(results, 1):
            print(f"  exp{i}: mean_pmae={mean_pmae:.2f}  overrides={overrides}")
