"""실험 25 준비: 052 DB 음식 1,266종(test 겹침 제외, experiments/25_llm_fewshot_prior/db052_foods.json)에 대해
① Gemini 3.8-flash 제로샷 1인분(실험 24b와 같은 프롬프트) ② 음식명 Gemini 임베딩(gemini-embedding-2, 텍스트마다 개별 호출). → OUT/db_prior.json, OUT/db_emb.json"""
import json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
M=Path(__file__).resolve().parents[2];OUT=M/"experiments/25_decomp_db_prior"
foods=json.loads((M/"experiments/25_llm_fewshot_prior/db052_foods.json").read_text());names=sorted(foods)
KEYS=["calories_kcal","fat_g","protein_g","carb_g","serving_g"]
PROMPT=("한국 음식 '{n}'의 일반적인 1인분(1회 제공량) 기준 영양 정보를 추정하세요. 숫자만 JSON 으로 답하세요.\n"
        "- calories_kcal: 열량(kcal)\n- fat_g: 지방(g)\n- protein_g: 단백질(g)\n- carb_g: 탄수화물(g)\n- serving_g: 1인분 중량(g)")
from google import genai
from google.genai import types
cl=genai.Client(api_key=__import__("os").environ.get("GEMINI_KEY") or next(l.split("=",1)[1].strip().strip("\"' ") for l in open(M/".env") if l.strip().startswith("GEMINI_KEY")))
cfg=types.GenerateContentConfig(response_mime_type="application/json",temperature=0,response_schema={"type":"OBJECT","properties":{k:{"type":"NUMBER"} for k in KEYS},"required":KEYS})
def prior(n):
    for t in range(6):
        try: j=json.loads(cl.models.generate_content(model="gemini-3.8-flash",contents=PROMPT.format(n=n),config=cfg).text);return n,[float(j[k]) for k in KEYS]
        except Exception: time.sleep(3*(t+1))
    return n,None
def emb(n):
    for t in range(6):
        try: return n,cl.models.embed_content(model="gemini-embedding-2",contents=n).embeddings[0].values
        except Exception: time.sleep(3*(t+1))
    return n,None
for fn,name in [(prior,"db_prior.json"),(emb,"db_emb.json")]:
    f=OUT/name;done=json.loads(f.read_text()) if f.exists() else {}
    with ThreadPoolExecutor(8) as ex:
        for n,v in ex.map(fn,[n for n in names if n not in done]):
            if v: done[n]=v
    f.write_text(json.dumps(done,ensure_ascii=False));print(name,len(done),"/",len(names),flush=True)
print("완료",flush=True)
