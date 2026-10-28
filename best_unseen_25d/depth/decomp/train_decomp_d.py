"""실험 25-d: 실험 25-c와 같고 보정 δ에 L2 벌점(λ·mean(δ²), 기본 λ=1.0)만 추가. 사진 배치·DB 배치의 LLM 값이 있는 샘플 모두.
원래 설명 — 실험 25-c: 실험 25-b와 같고 "1인분 = LLM 값 × 보정"만 바꿈.
  LLM 값이 있으면 log1p(1인분) = log1p(LLM 1인분) + δ(텍스트, LLM 특징). δ 헤드의 마지막 층을 0으로 시작 → 처음 출력은 LLM 값 그대로.
  LLM 값이 없거나(OmniFood) 학습 중 30% 드롭된 샘플은 기존처럼 자유 1인분 헤드를 쓴다.
원래 설명 — 실험 25-b: 실험 25와 같고 "배율 헤드 직접 지도"만 추가.
  배율 헤드 출력 s(log 배율)를 학습 라벨로 지도: 122 분량 사진 → log(정답 배율 0.25~1.25), 122 분류 369종(1인분 사진) → log(1)=0, OmniFood → 지도 안 함.
  (1인분 × 배율 사이에서 크기를 떠넘기는 문제를 막아 두 헤드가 각자 역할을 하게 함. test 에서는 배율을 예측하므로 누수 없음)
원래 설명 — 실험 25: 분해형 헤드(1인분 × 분량) + 사진 없는 052 DB 음식 학습 + LLM 1인분(3.8-flash). Stage 3 한 fold 학습.
- 1인분 헤드: 텍스트 특징(64) + LLM 1인분 인코더(64) → log1p(1인분 5개). 이미지 안 봄.
- 분량 헤드: 게이트된 이미지·주파수·깊이 특징(1536) → log 배율(스칼라). 텍스트 안 봄. 분량 보조 CE(0.5)도 이 특징에서.
- 최종 = 1인분 × exp(배율). 정답 배율을 곱하지 않음.
- 매 스텝: 사진 배치(122 분량 + 122 분류 369종 30% + OmniFood 10%) 손실 + 052 DB 텍스트 전용 128개로 1인분 헤드 손실.
기존 depth/train_depth.py 는 import 만 함(평가 함수 재사용). usage: python train_decomp.py --fold k --out_dir OUT --device cuda:N"""
import argparse,functools,json,random,re,sys,time
from pathlib import Path
import numpy as np,torch,torch.nn as nn,yaml
from torch.utils.data import DataLoader
WT=Path(__file__).resolve().parents[2];M=WT;sys.path[:0]=[str(WT),str(WT/"text_baseline_the_best")]
import os;os.chdir(WT)
import depth.train_depth as td
from depth.model_depth import DepthGatedNutritionModel
from depth.depth_data import DepthStore,WithDepth,depth_collate
from train_curriculum_4stages import MultiRehearsalDataset,unified_nutrition_collate_fn
import dataset_omnifood8k as omni,dataset_aihub122_classification_nutrition as std,dataset_aihub122_multimodal as q122
KEYS=["calories","fat","protein","carb","mass"]
ap=argparse.ArgumentParser();ap.add_argument("--fold",type=int,required=True);ap.add_argument("--out_dir",required=True);ap.add_argument("--device",default="cuda:0")
ap.add_argument("--seed",type=int,default=None);ap.add_argument("--db_weight",type=float,default=1.0);ap.add_argument("--db_batch",type=int,default=128)
ap.add_argument("--delta_l2",type=float,default=1.0);ap.add_argument("--scale_weight",type=float,default=1.0);ap.add_argument("--std_weight",type=float,default=0.3);ap.add_argument("--epochs",type=int,default=500);ap.add_argument("--patience",type=int,default=10)
a=ap.parse_args();seed=a.seed if a.seed is not None else 42+a.fold;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
dev=torch.device(a.device);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True);P=print
def print(*x,**k): P(*x,**k,flush=True)
# ── LLM 1인분 / DB ──
TEST={l.strip() for l in (WT/"foodlist_test_122.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
VAL=set(json.loads((WT/"foodlist_folds_122.json").read_text(encoding="utf-8"))["folds"][a.fold])
SERV=json.loads((M/"experiments/24b_llm_prior_38flash/llm_serving.json").read_text())
R25=M/"experiments/25_decomp_db_prior";DBF=json.loads((M/"experiments/25_llm_fewshot_prior/db052_foods.json").read_text())
DBP=json.loads((R25/"db_prior.json").read_text());DBE=json.loads((R25/"db_emb.json").read_text())
norm=lambda s:re.sub(r"[\s\(\)\[\]_·,<>]","",str(s));vn=[norm(v) for v in VAL]
def hit(n,ref):
    x=norm(n);return any(t in x or (len(x)>=2 and x in t) for t in ref)
dbn=[n for n in DBF if n in DBP and n in DBE and not hit(n,vn)]
print(f"[db] 052 음식 {len(DBF)} → 이 fold 검증 음식과 겹침 제외 후 {len(dbn)}종")
_L=torch.log1p(torch.tensor([v for n,v in SERV.items() if n not in TEST]+[DBP[n] for n in dbn],dtype=torch.float32).clamp(min=0));PMU,PSD=_L.mean(0),_L.std(0)
dbE=torch.tensor([DBE[n] for n in dbn],dtype=torch.float32,device=dev);dbP=torch.tensor([DBP[n] for n in dbn],dtype=torch.float32,device=dev)
dbT=torch.log1p(torch.tensor([DBF[n] for n in dbn],dtype=torch.float32,device=dev).clamp(min=0))  # [kcal,fat,prot,carb,serving_g]
# ── 모델 ──
class Decomp(nn.Module):
    def __init__(self,m):
        super().__init__();self.m=m;h=256
        self.prior_mlp=nn.Sequential(nn.Linear(5,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU());self.prior_unknown=nn.Parameter(torch.randn(64)*0.02);self.prior_drop=0.3
        self.base_head=nn.Sequential(nn.Linear(64+64,h),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h,5))
        self.delta_head=nn.Sequential(nn.Linear(64+64,h),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h,5));nn.init.zeros_(self.delta_head[-1].weight);nn.init.zeros_(self.delta_head[-1].bias)
        self.scale_head=nn.Sequential(nn.Linear(1536,h),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h,1));nn.init.zeros_(self.scale_head[-1].weight);nn.init.zeros_(self.scale_head[-1].bias)
        self.portion_head=nn.Linear(1536,5);self.last_gate=None;self.last_feat=None;self._prior=None;self.last_scale=None;self.delta_terms=[]
        with torch.no_grad(): self.base_head[-1].bias.copy_(dbT.mean(0).cpu())
    def new_params(self): return [p for n,p in self.named_parameters() if not n.startswith("m.")]
    def base(self,text_emb,vals,known,train):
        pz=self.prior_mlp((torch.log1p(vals.clamp(min=0))-PMU.to(vals.device))/PSD.to(vals.device));B=len(vals);keep=known.clone()
        if train and self.prior_drop>0: keep&=~(torch.rand(B,device=vals.device)<self.prior_drop)
        pz=torch.where(keep[:,None],pz,self.prior_unknown.expand(B,-1))
        x=torch.cat([self.m.text_encoder(text_emb),pz],1);free=self.base_head(x)
        dl=self.delta_head(x);anchored=torch.log1p(vals.clamp(min=0))+dl
        if keep.any(): self.delta_terms.append((dl[keep]**2).mean())  # LLM 값 × 보정 (로그 공간에서 더하기)
        return torch.where(keep[:,None],anchored,free)  # log1p(1인분)
    def forward(self,images,has_image,text_emb,apply_dropout,depth=None,depth_meta=None,has_depth=None):
        m=self.m;B,d=images.shape[0],images.device;keep=has_image.clone().bool()
        if apply_dropout and m.image_drop>0: keep&=~(torch.rand(B,device=d)<m.image_drop)
        img=m.mask_token.expand(B,-1).clone();mix=m.mix_mask_token.expand(B,-1).clone()
        if keep.any(): img[keep],mix[keep]=m._image_branches(images[keep])
        df=m.depth_mask_token.expand(B,-1).clone();dk=keep&(has_depth.bool() if has_depth is not None else torch.ones_like(keep))
        if dk.any(): df[dk]=m.depth_encoder(depth[dk],depth_meta[dk])
        feats=[img,mix,df];K=3;g=m.gate_floor+(1-K*m.gate_floor)*torch.softmax(m.gate(torch.cat(feats,1)),1);self.last_gate=g.detach()
        f=torch.cat([x*(K*g[:,k:k+1]) for k,x in enumerate(feats)],1);self.last_feat=f
        vals,known=self._prior;lb=self.base(text_emb,vals,known,apply_dropout);s=self.scale_head(f).clamp(-3,3);self.last_scale=s.squeeze(-1)
        pa=torch.expm1(lb).clamp(min=0)*torch.exp(s);o=torch.log1p(pa)
        return {k:o[:,i:i+1] for i,k in enumerate(KEYS)}
cfg=yaml.safe_load((WT/"text_baseline_the_best/config_multimodal_296_text64_ingredients.yaml").read_text());mc=cfg["model"]
m=DepthGatedNutritionModel(use_depth=True,image_dim=mc["image_dim"],text_in_dim=3072,text_hidden1=mc["text_hidden1"],text_hidden2=mc["text_hidden2"],head_hidden=mc["head_hidden"],
    dropout=mc["dropout"],freeze_backbone_layers=mc["freeze_backbone_layers"],modality_dropout_prob=0.0,freq_cfg=cfg.get("freq"),mass_feedback=False)
sd=torch.load(M/"experiments/10_depth/with_depth/checkpoint_stage2_122_std.pth",map_location="cpu",weights_only=False)["state_dict"]
miss,unexp=m.load_state_dict({k:v for k,v in sd.items() if not k.startswith("portion_head.")},strict=False);print("[init] ⑧ Stage 2 ← missing",miss,"unexpected",len(unexp))
model=Decomp(m).to(dev)
# ── 데이터 (train_depth.py Stage 3 와 같은 구성, 369종 비중만 0.1→std_weight, 앞 500장→전체) ──
store=DepthStore();E_omni,E_std,E_q=omni._TRANSFORMS["val"],std._TRANSFORMS["val"],q122._IMAGE_TRANSFORM["eval"]
W=lambda ds,tf,tr,fv=False:WithDepth(ds,store,tf,tr,flip_v=fv)
so,eo=omni.load_omnifood_dataset();s2,e2,v2=std.load_122_classification_dataset()
s3,e3=q122.load_aihub122_dataset(emb_cache_path=WT/"text_baseline_the_best/emb_cache_with_ingredients.json",val_foods=sorted(VAL))
q_ds=lambda r,tr:W(q122.AIHub122MultimodalDataset(r,e3,train=tr),E_q,tr)
train_ds=MultiRehearsalDataset(q_ds(s3["train"],True),[W(std.AIHub122ClassificationDataset(s2["train"],e2,v2,train=True),E_std,True),W(omni.OmniFoodDataset(so["train"][:500],eo,train=True),E_omni,True,True)],[a.std_weight,0.1])
_c=functools.partial(depth_collate,base_collate=unified_nutrition_collate_fn)
def coll(batch):
    o=_c(batch);n=[b.get("food_name","") for b in batch]
    o["is_std"]=torch.tensor([("portion_label" not in b) and bool(b.get("food_name","")) for b in batch])  # 122 분류 369종 사진(1인분)
    o["prior"]=torch.tensor([SERV.get(x,[0]*5) for x in n],dtype=torch.float32);o["prior_known"]=torch.tensor([x in SERV for x in n]);return o
mk=lambda ds,sh:DataLoader(ds,batch_size=64,shuffle=sh,collate_fn=coll,num_workers=8,pin_memory=True,persistent_workers=True)
trL,vaL,teL=mk(train_ds,True),mk(q_ds(s3["val"],False),False),mk(q_ds(s3["test"],False),False)
def fwd(model,b,dev,train):
    model._prior=(b["prior"].to(dev),b["prior_known"].to(dev));return model(b["images"].to(dev),b["has_image"].to(dev),b["text_emb"].to(dev),train,depth=b["depth"].to(dev),depth_meta=b["depth_meta"].to(dev),has_depth=b["has_depth"].to(dev))
td.forward=fwd  # td.evaluate 가 이 함수를 쓴다
newp=model.new_params();ids={id(p) for p in newp}
opt=torch.optim.AdamW([{"params":[p for p in model.parameters() if p.requires_grad and id(p) not in ids],"lr":5e-5},{"params":newp,"lr":1e-3}],weight_decay=1e-4)
sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=5);crit=nn.SmoothL1Loss()
print(f"[decomp] fold {a.fold} seed {seed} train {len(train_ds)} val {len(vaL.dataset)} test {len(teL.dataset)} db {len(dbn)} db_weight {a.db_weight}")
best,bad,bep=float("inf"),0,0;save=out/"model_stage3_final.pth"
for ep in range(1,a.epochs+1):
    model.train();t0=time.time();tot=dbl=scl=dpl=0.0
    for b in trL:
        model.delta_terms=[];t=torch.log1p(b["targets"].to(dev).clamp(min=0));p=fwd(model,b,dev,True)
        loss=sum(crit(p[k].squeeze(-1),t[:,i]) for i,k in enumerate(KEYS))
        q=b["portion"].to(dev)
        if (q>=0).any(): loss=loss+0.5*nn.functional.cross_entropy(model.portion_head(model.last_feat),q,ignore_index=-1)
        st=torch.full_like(model.last_scale,float("nan"));qv=q>=0
        st[qv]=torch.log(torch.tensor([0.25,0.5,0.75,1.0,1.25],device=dev)[q[qv]]);st[b["is_std"].to(dev)]=0.0
        sm=~torch.isnan(st);ls=((model.last_scale[sm]-st[sm])**2).mean() if sm.any() else model.last_scale.sum()*0
        loss=loss+a.scale_weight*ls;scl+=ls.item()*len(t)
        i=torch.randint(0,len(dbn),(a.db_batch,),device=dev);lb=model.base(dbE[i],dbP[i],torch.ones(a.db_batch,dtype=torch.bool,device=dev),True)
        ld=crit(lb,dbT[i]);loss=loss+a.db_weight*ld
        if model.delta_terms: dpen=torch.stack(model.delta_terms).mean();loss=loss+a.delta_l2*dpen;dpl+=dpen.item()*len(t)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();tot+=loss.item()*len(t);dbl+=ld.item()*len(t)
    v=td.evaluate(model,vaL,dev);sch.step(v["mean4"])
    print(f"==> Epoch {ep:3d} ({time.time()-t0:.0f}s) loss {tot/len(train_ds):.4f} (db {dbl/len(train_ds):.4f} scale {scl/len(train_ds):.4f} δ² {dpl/len(train_ds):.4f}) | val A4 {v['mean4']:.2f} G4 {v['G_mean4']:.2f} | cal {v['calories']:.1f} fat {v['fat']:.1f} prot {v['protein']:.1f} carb {v['carb']:.1f} mass {v['mass']:.1f} | gate {v['gate']}")
    if v["mean4"]<best:
        best,bad,bep=v["mean4"],0,ep;torch.save({"epoch":ep,"state_dict":model.state_dict(),"val":v},save);print(f"  ★ best {best:.2f}")
    else:
        bad+=1
        if bad>=a.patience: print(f"[EarlyStop] Epoch {ep}. best epoch {bep}");break
model.load_state_dict(torch.load(save,map_location=dev,weights_only=False)["state_dict"])
r=td.evaluate(model,teL,dev,save_preds=out/"test_preds.npz");r["best_epoch"]=bep
(out/"test_result.json").write_text(json.dumps(r,indent=2));print(f"[TEST] A4 {r['mean4']:.2f} G4 {r['G_mean4']:.2f} | gate {r['gate']}")
