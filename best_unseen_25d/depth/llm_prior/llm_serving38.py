"""실험 24b: Gemini 3.8-flash 에 음식명만 주고 일반적인 1인분 영양값 추정 (DB 값은 프롬프트에 넣지 않음). 텍스트 전용 호출.
대상: Stage 3 학습·평가에 나오는 모든 음식명(122 분량 78종 + 리허설용 122 표준 369종). → OUT/llm_serving.json {음식명: [kcal,fat,prot,carb,mass_g]}"""
import json,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
WT=Path(__file__).resolve().parents[2];M=WT;sys.path[:0]=[str(WT),str(WT/"text_baseline_the_best")]
import os;os.chdir(WT)
import dataset_aihub122_multimodal as q122,dataset_aihub122_classification_nutrition as std
OUT=M/"experiments/24b_llm_prior_38flash/llm_serving.json"
s3,_=q122.load_aihub122_dataset(emb_cache_path=WT/"text_baseline_the_best/emb_cache_with_ingredients.json");s2,_,_=std.load_122_classification_dataset()
names=sorted({r["food_name"] for s in (s3,s2) for v in s.values() for r in v});print("음식",len(names),flush=True)
PROMPT=("한국 음식 '{n}'의 일반적인 1인분(1회 제공량) 기준 영양 정보를 추정하세요. 숫자만 JSON 으로 답하세요.\n"
        "- calories_kcal: 열량(kcal)\n- fat_g: 지방(g)\n- protein_g: 단백질(g)\n- carb_g: 탄수화물(g)\n- serving_g: 1인분 중량(g)")
KEYS=["calories_kcal","fat_g","protein_g","carb_g","serving_g"]
from google import genai
from google.genai import types
cl=genai.Client(api_key=__import__("os").environ.get("GEMINI_KEY") or next(l.split("=",1)[1].strip().strip("\"' ") for l in open(M/".env") if l.strip().startswith("GEMINI_KEY")))
cfg=types.GenerateContentConfig(response_mime_type="application/json",temperature=0,
    response_schema={"type":"OBJECT","properties":{k:{"type":"NUMBER"} for k in KEYS},"required":KEYS})
done=json.loads(OUT.read_text()) if OUT.exists() else {}
def ask(n):
    for t in range(6):
        try:
            j=json.loads(cl.models.generate_content(model="gemini-3.8-flash",contents=PROMPT.format(n=n),config=cfg).text);return n,[float(j[k]) for k in KEYS]
        except Exception as e: time.sleep(3*(t+1))
    return n,None
with ThreadPoolExecutor(8) as ex:
    for n,v in ex.map(ask,[n for n in names if n not in done]):
        if v: done[n]=v
OUT.write_text(json.dumps(done,ensure_ascii=False,indent=0));print("완료",len(done),"/",len(names),"누락",[n for n in names if n not in done][:10],flush=True)
