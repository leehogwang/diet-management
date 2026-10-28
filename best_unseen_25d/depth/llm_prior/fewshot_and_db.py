"""실험 25 준비.
① 퓨샷 LLM 1인분: 질의 음식마다, test 음식(23종)과 질의 음식 자신을 뺀 122 DB 음식 중 이름 임베딩이 가장 비슷한 10종의 DB 1인분 값을 예시로 주고 추정.
   test 23종에서 제로샷(실험 24)과 DB 대비 정확도 비교. 모델: gemini-3.5-flash-lite(실험 24와 동일) / gemini-3.8-flash.
② 052 통합영양성분DB '음식' 항목(1회 제공량 기준) 정리: test 음식과 이름이 겹치거나 포함관계인 항목 제외 → 남는 음식 수.
출력: OUT/fewshot_test.json, OUT/db052_foods.json, OUT/summary.json"""
import json,sys,re,time,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np,pandas as pd
WT=Path(__file__).resolve().parents[2];M=WT;sys.path[:0]=[str(WT),str(WT/"text_baseline_the_best")];os.chdir(WT)
import dataset_aihub122_multimodal as q122,dataset_aihub122_classification_nutrition as std
OUT=M/"experiments/25_llm_fewshot_prior"
TEST={l.strip() for l in (WT/"foodlist_test_122.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
s3,_=q122.load_aihub122_dataset(emb_cache_path=WT/"text_baseline_the_best/emb_cache_with_ingredients.json");s2,_,_=std.load_122_classification_dataset()
base={}
for s in s3.values():
    for r in s: base.setdefault(r["food_name"],[]).append(np.array(r["targets"],float)/q122.PORTION_FACTORS[r["portion"]])
for s in s2.values():
    for r in s: base.setdefault(r["food_name"],[]).append(np.array(r["targets"],float))
base={f:np.median(v,0) for f,v in base.items()}  # [kcal,fat,prot,carb,mass] DB 1인분
emb=json.loads((WT/"text_baseline_the_best/emb_cache_with_ingredients.json").read_text())
E={f:np.asarray(emb[f],float)/np.linalg.norm(emb[f]) for f in base if f in emb}
test_foods=sorted(f for f in base if f in TEST);pool=[f for f in E if f not in TEST]
print("1인분 음식",len(base),"test",len(test_foods),"예시 후보",len(pool),flush=True)
KEYS=["calories_kcal","fat_g","protein_g","carb_g","serving_g"]
from google import genai
from google.genai import types
cl=genai.Client(api_key=__import__("os").environ.get("GEMINI_KEY") or next(l.split("=",1)[1].strip().strip("\"' ") for l in open(M/".env") if l.strip().startswith("GEMINI_KEY")))
cfg=types.GenerateContentConfig(response_mime_type="application/json",temperature=0,response_schema={"type":"OBJECT","properties":{k:{"type":"NUMBER"} for k in KEYS},"required":KEYS})
def prompt(n,k=10):
    q=E.get(n);cand=[f for f in pool if f!=n]
    ex=sorted(cand,key=lambda f:-float(E[f]@q))[:k] if q is not None else cand[:k]
    lines="\n".join(f"- {f}: 열량 {base[f][0]:.0f}kcal, 지방 {base[f][1]:.1f}g, 단백질 {base[f][2]:.1f}g, 탄수화물 {base[f][3]:.1f}g, 1인분 {base[f][4]:.0f}g" for f in ex)
    return (f"아래는 한국 영양DB의 1인분 기준 영양값 예시입니다.\n{lines}\n\n같은 DB 기준으로 한국 음식 '{n}'의 1인분 영양값을 추정하세요. 숫자만 JSON 으로 답하세요."),ex
def ask(args):
    n,model=args;p,ex=prompt(n)
    for t in range(6):
        try: j=json.loads(cl.models.generate_content(model=model,contents=p,config=cfg).text);return n,[float(j[k]) for k in KEYS],ex
        except Exception: time.sleep(3*(t+1))
    return n,None,ex
zero=json.loads((M/"experiments/24_llm_serving_prior/llm_serving.json").read_text())
def G(pred):
    f=[x for x in test_foods if x in pred];P=np.array([pred[x] for x in f]);D=np.array([base[x] for x in f]);e=np.abs(P-D)
    return {"n":len(f),"G":[round(float(v),1) for v in e.sum(0)/D.sum(0)*100],"G_mean4":round(float((e.sum(0)/D.sum(0))[:4].mean()*100),2),
            "A_mean4":round(float((e/(np.abs(D)+1))[:,:4].mean()*100),2)}
res={"zero_shot_3.5-flash-lite(실험24)":G(zero)};few={}
for model in ["gemini-3.5-flash-lite","gemini-3.8-flash"]:
    with ThreadPoolExecutor(8) as ex: out=list(ex.map(ask,[(n,model) for n in test_foods]))
    pr={n:v for n,v,_ in out if v};few[model]={n:{"pred":v,"examples":e} for n,v,e in out};res[f"few_shot_{model}"]=G(pr);print(model,res[f"few_shot_{model}"],flush=True)
(OUT/"fewshot_test.json").write_text(json.dumps(few,ensure_ascii=False,indent=1))
# ② 052 DB
d=pd.read_excel(M/"052.건강관리를_위한_음식_이미지_데이터/통합_영양성분DB.xlsx",sheet_name="원본",header=3);d=d[d["DB군"]=="음식"]
cols={"에너지(㎉)":"kcal","지방(g)":"fat","단백질(g)":"prot","탄수화물(g)":"carb","1회제공량":"serving_g"}
for c in cols: d[c]=pd.to_numeric(d[c],errors="coerce")
d=d.dropna(subset=list(cols));d=d[(d["내용량_단위"]=="g")&(d["1회제공량"]>0)]
norm=lambda s:re.sub(r"[\s\(\)\[\]_·,]","",str(s))
tn=[norm(t) for t in TEST]
def leak(n):
    x=norm(n);return any(t in x or (len(x)>=2 and x in t) for t in tn)
d["leak"]=d["식품명"].map(leak);g=d[~d["leak"]].groupby("식품명")[list(cols)].median().reset_index()
foods={r["식품명"]:[float(r[c]) for c in cols] for _,r in g.iterrows()}
(OUT/"db052_foods.json").write_text(json.dumps(foods,ensure_ascii=False))
res["db052"]={"음식 행":int(len(d)),"test와 겹쳐 제외한 행":int(d["leak"].sum()),"제외된 이름 예시":sorted(d[d["leak"]]["식품명"].unique().tolist())[:30],"남은 고유 음식":len(foods)}
(OUT/"summary.json").write_text(json.dumps(res,ensure_ascii=False,indent=1));print(json.dumps(res,ensure_ascii=False,indent=1),flush=True)
