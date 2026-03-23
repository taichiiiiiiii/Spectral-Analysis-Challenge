"""Issue #88: サイクル11 - 残差分析ベースの改善

1. GBR PLS6/PLS8（多成分情報活用）
2. Huber epsilon微調整（1.0-1.2）
3. TheilSen回帰
4. 前処理多様化（EPO+siPLS+Huber等）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from itertools import combinations
from scipy.optimize import minimize
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, TheilSenRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from src.eda.data_loader import load_train, get_spectral_columns
from src.preprocessing.issue18_snv import apply_snv
from src.preprocessing.issue20_savgol import apply_savgol
from src.preprocessing.issue22_epo import compute_epo_projection, apply_epo
from src.preprocessing.issue19_msc import compute_msc_reference
from src.preprocessing.issue62_additional_preprocessing import apply_asls, apply_piecewise_msc
from src.modeling.issue65_feature_selection import sipls_select, ipls_select

DATA_DIR = Path(__file__).resolve().parents[2] / "Input_data"
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "modeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))

def pp(Xtr, Xte, g, n):
    if n=="SNV": return apply_snv(Xtr), apply_snv(Xte)
    elif n=="EPO(1)":
        P=compute_epo_projection(Xtr,g,n_components=1); return apply_epo(Xtr,P), apply_epo(Xte,P)
    elif n=="SNV+AsLS":
        return apply_asls(apply_snv(Xtr),lam=1e6), apply_asls(apply_snv(Xte),lam=1e6)
    elif n=="PMSC":
        ref=compute_msc_reference(Xtr); return apply_piecewise_msc(Xtr,ref,3), apply_piecewise_msc(Xte,ref,3)
    elif n=="SG2d+EPO":
        xs,xst=apply_savgol(Xtr,deriv=2,window_length=7),apply_savgol(Xte,deriv=2,window_length=7)
        P=compute_epo_projection(xs,g,n_components=1); return apply_epo(xs,P),apply_epo(xst,P)
    elif n=="SNV+SG2d":
        return apply_savgol(apply_snv(Xtr),deriv=2,window_length=7),apply_savgol(apply_snv(Xte),deriv=2,window_length=7)
    return Xtr.copy(),Xte.copy()

def fs(Xtr,Xte,y,n):
    if not n: return Xtr,Xte
    if n.startswith("siPLS"):
        p=n.replace("siPLS(","").rstrip(")").split(",")
        Xtr,Xte,_=sipls_select(Xtr,y,Xte,n_intervals=int(p[0]),n_components=3,n_combine=int(p[1]))
    elif n.startswith("iPLS"):
        ni=int(n.replace("iPLS(","").rstrip(")")); Xtr,Xte,_=ipls_select(Xtr,y,Xte,n_intervals=ni,n_components=3,n_best=1)
    return Xtr,Xte

def get_scores(Xtr,Xte,y,nc,tf):
    nc=max(1,min(nc,Xtr.shape[1]-1))
    yf=np.sqrt(y) if tf=="sqrt" else y.copy()
    pls=PLSRegression(n_components=nc); pls.fit(Xtr,yf)
    return pls.transform(Xtr),pls.transform(Xte),yf

def run(Xtr_raw,Xte_raw,y,g,c):
    Xtr,Xte=pp(Xtr_raw,Xte_raw,g,c["pp"]); Xtr,Xte=fs(Xtr,Xte,y,c.get("fs"))
    t=c["type"]; nc=c.get("nc",4); tf=c["tf"]
    if t=="pls":
        nc2=max(1,min(nc,Xtr.shape[1]-1)); yf=np.sqrt(y) if tf=="sqrt" else y.copy()
        pls=PLSRegression(n_components=nc2); pls.fit(Xtr,yf); p=pls.predict(Xte).ravel()
        return np.clip(p,0,None)**2 if tf=="sqrt" else p
    Ttr,Tte,yf=get_scores(Xtr,Xte,y,nc,tf)
    if t=="gbr":
        m=GradientBoostingRegressor(n_estimators=c.get("ne",200),max_depth=c.get("md",3),
            learning_rate=c.get("lr",0.05),subsample=0.8,min_samples_leaf=5,random_state=42,
            validation_fraction=0.15,n_iter_no_change=50,tol=0.01)
        m.fit(Ttr,yf); p=m.predict(Tte)
    elif t=="huber":
        sc=StandardScaler(); Ttr_s,Tte_s=sc.fit_transform(Ttr),sc.transform(Tte)
        m=HuberRegressor(epsilon=c.get("eps",1.35),max_iter=200,alpha=c.get("alpha",0.01))
        m.fit(Ttr_s,yf); p=m.predict(Tte_s)
    elif t=="theilsen":
        sc=StandardScaler(); Ttr_s,Tte_s=sc.fit_transform(Ttr),sc.transform(Tte)
        m=TheilSenRegressor(max_subpopulation=300,random_state=42)
        m.fit(Ttr_s,yf); p=m.predict(Tte_s)
    else:
        raise ValueError(t)
    return np.clip(p,0,None)**2 if tf=="sqrt" else p

def main():
    t0=time.time()
    df=load_train(DATA_DIR); sc=get_spectral_columns(df)
    X=df[sc].values; y=df["含水率"].values; g=df["樹種"].values
    logo=LeaveOneGroupOut(); folds=list(logo.split(X,y,g))
    sp=[np.unique(g[te])[0] for _,te in folds]

    print("="*70); print("Issue #88: サイクル11"); print("="*70)

    cfgs=[
        # Cycle10ベスト10（核心7モデル）
        {"name":"M1:EPO+PLS4+sqrt","pp":"EPO(1)","nc":4,"tf":"sqrt","type":"pls"},
        {"name":"H2a:SNV+iPLS50+Huber(1.1)+sqrt","pp":"SNV","nc":4,"tf":"sqrt","fs":"iPLS(50)","eps":1.1,"type":"huber"},
        {"name":"H6:SNV+AsLS+siPLS+Huber+sqrt","pp":"SNV+AsLS","nc":4,"tf":"sqrt","fs":"siPLS(30,3)","type":"huber"},
        {"name":"N10:PMSC+siPLS+GBR+sqrt","pp":"PMSC","nc":4,"tf":"sqrt","fs":"siPLS(30,3)","type":"gbr"},
        {"name":"G9:SG2d+EPO+GBR+raw","pp":"SG2d+EPO","nc":4,"tf":"raw","type":"gbr"},
        {"name":"Gnew1:EPO+GBR150+raw","pp":"EPO(1)","nc":4,"tf":"raw","ne":150,"lr":0.08,"type":"gbr"},
        {"name":"N2:SNV+SG2d+GBR+raw","pp":"SNV+SG2d","nc":4,"tf":"raw","type":"gbr"},
        # 追加ベースモデル
        {"name":"M5:SNV+AsLS+siPLS+PLS5+sqrt","pp":"SNV+AsLS","nc":5,"tf":"sqrt","fs":"siPLS(30,3)","type":"pls"},
        {"name":"M3:SNV+iPLS50+PLS4+sqrt","pp":"SNV","nc":4,"tf":"sqrt","fs":"iPLS(50)","type":"pls"},

        # 新: Huber epsilon探索
        {"name":"H_e100:SNV+iPLS50+Huber(1.0)+sqrt","pp":"SNV","nc":4,"tf":"sqrt","fs":"iPLS(50)","eps":1.0,"type":"huber"},
        {"name":"H_e105:SNV+iPLS50+Huber(1.05)+sqrt","pp":"SNV","nc":4,"tf":"sqrt","fs":"iPLS(50)","eps":1.05,"type":"huber"},
        {"name":"H_e115:SNV+iPLS50+Huber(1.15)+sqrt","pp":"SNV","nc":4,"tf":"sqrt","fs":"iPLS(50)","eps":1.15,"type":"huber"},
        {"name":"H_e120:SNV+iPLS50+Huber(1.2)+sqrt","pp":"SNV","nc":4,"tf":"sqrt","fs":"iPLS(50)","eps":1.2,"type":"huber"},

        # 新: EPO+siPLS+Huber
        {"name":"H_epo:EPO+siPLS30_3+Huber+sqrt","pp":"EPO(1)","nc":4,"tf":"sqrt","fs":"siPLS(30,3)","type":"huber"},

        # 新: GBR PLS6
        {"name":"G_pls6:SG2d+EPO+PLS6+GBR+raw","pp":"SG2d+EPO","nc":6,"tf":"raw","type":"gbr"},
        {"name":"G_pls6b:EPO+PLS6+GBR+raw","pp":"EPO(1)","nc":6,"tf":"raw","type":"gbr"},

        # 新: TheilSen
        {"name":"TS1:SNV+iPLS50+TheilSen+sqrt","pp":"SNV","nc":4,"tf":"sqrt","fs":"iPLS(50)","type":"theilsen"},
        {"name":"TS2:EPO+TheilSen+sqrt","pp":"EPO(1)","nc":4,"tf":"sqrt","type":"theilsen"},
    ]

    n=len(cfgs); all_p=[[] for _ in range(n)]
    print(f"\n--- 個別 ({n}) ---\n",flush=True)
    for i,c in enumerate(cfgs):
        t1=time.time(); fr=[]
        for fi,(tr,te) in enumerate(folds):
            try: p=run(X[tr],X[te],y[tr],g[tr],c); all_p[i].append(p); fr.append(rmse(y[te],p))
            except Exception as e: all_p[i].append(np.full(len(te),y[tr].mean())); fr.append(999.0); print(f"  ERR {c['name']} f{fi}: {e}")
        print(f"  [{i+1:>2}/{n}] {c['name']}: {np.mean(fr):.2f} ({time.time()-t1:.1f}s)",flush=True)

    def ev(idx,w=None):
        fr=[]
        for fi,(_,te) in enumerate(folds):
            ps=[all_p[m][fi] for m in idx]
            if w is not None: wn=np.array(w);wn/=wn.sum();e=sum(wi*p for wi,p in zip(wn,ps))
            else: e=np.mean(ps,axis=0)
            fr.append(rmse(y[te],np.clip(e,0,300)))
        return np.mean(fr),fr

    def ow(idx,nr=40):
        nn=len(idx)
        def obj(w): wn=np.abs(w)/np.sum(np.abs(w));r,_=ev(idx,wn);return r
        br,bw=np.inf,np.ones(nn)/nn
        for s in range(nr):
            w0=np.random.RandomState(s).dirichlet(np.ones(nn)); r=minimize(obj,w0,method="Nelder-Mead",options={"maxiter":5000})
            if r.fun<br: br=r.fun; bw=np.abs(r.x)/np.sum(np.abs(r.x))
        return bw,br

    iv=sorted([(ev([i])[0],i) for i in range(n)])
    print("\n個別:"); [print(f"  {r:.2f} | {cfgs[i]['name']}") for r,i in iv]

    top=min(14,n); ti=[i for _,i in iv[:top]]
    res=[]
    for k in [8,9,10,11,12]:
        if k>top: break
        br,bc=np.inf,None
        for combo in combinations(ti,k):
            r,_=ev(list(combo));
            if r<br: br=r;bc=list(combo)
        if bc:
            w,rw=ow(bc); res.append((f"Best{k}",rw,bc,w))
            print(f"  Best{k}: {rw:.4f}",flush=True)
            for i,wi in zip(bc,w):
                if wi>0.02: print(f"    {wi:.3f}: {cfgs[i]['name']}")

    res.sort(key=lambda x:x[1]); best=res[0]
    print(f"\n{'='*70}\nBEST: {best[1]:.4f}")
    print(f"前ベスト: 14.0977, 改善: {14.0977-best[1]:+.4f}\n{'='*70}")

    _,fr=ev(best[2],best[3])
    print("\nfold:")
    for s,r in zip(sp,fr): print(f"  {s}: {r:.2f}{' ※' if s=='ベイスギ' else ''}")
    nbs=[r for s,r in zip(sp,fr) if s!="ベイスギ"]; print(f"  除ベイスギ: {np.mean(nbs):.4f}")

    rows=[{"method":nm,"rmse":r,"models":str([cfgs[i]["name"] for i in c])} for nm,r,c,w in res]
    pd.DataFrame(rows).to_csv(OUT_DIR/"issue88_cycle11_results.csv",index=False)
    print(f"\n保存. 時間: {time.time()-t0:.0f}s")

if __name__=="__main__": main()
