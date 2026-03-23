"""Issue #87: 最終統合 - Cycle9ベース + 新発見Huberモデル

Cycle9ベスト(14.1594)にH5(PMSC+siPLS+Huber)とH2a(eps=1.1)を追加。
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
from sklearn.linear_model import HuberRegressor
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

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def pp(X_tr, X_te, g, pp_name):
    if pp_name == "SNV": return apply_snv(X_tr), apply_snv(X_te)
    elif pp_name == "EPO(1)":
        P = compute_epo_projection(X_tr, g, n_components=1)
        return apply_epo(X_tr, P), apply_epo(X_te, P)
    elif pp_name == "SNV+AsLS(1e6)":
        return apply_asls(apply_snv(X_tr), lam=1e6), apply_asls(apply_snv(X_te), lam=1e6)
    elif pp_name == "PMSC":
        ref = compute_msc_reference(X_tr)
        return apply_piecewise_msc(X_tr, ref, 3), apply_piecewise_msc(X_te, ref, 3)
    elif pp_name == "SG2d+EPO(1)":
        xs, xst = apply_savgol(X_tr, deriv=2, window_length=7), apply_savgol(X_te, deriv=2, window_length=7)
        P = compute_epo_projection(xs, g, n_components=1)
        return apply_epo(xs, P), apply_epo(xst, P)
    elif pp_name == "SNV+SG2d":
        return apply_savgol(apply_snv(X_tr), deriv=2, window_length=7), apply_savgol(apply_snv(X_te), deriv=2, window_length=7)
    return X_tr.copy(), X_te.copy()

def fs(X_tr, X_te, y, fs_name):
    if not fs_name: return X_tr, X_te
    if fs_name.startswith("siPLS"):
        p = fs_name.replace("siPLS(","").rstrip(")").split(",")
        X_tr, X_te, _ = sipls_select(X_tr, y, X_te, n_intervals=int(p[0]), n_components=3, n_combine=int(p[1]))
    elif fs_name.startswith("iPLS"):
        n = int(fs_name.replace("iPLS(","").rstrip(")"))
        X_tr, X_te, _ = ipls_select(X_tr, y, X_te, n_intervals=n, n_components=3, n_best=1)
    return X_tr, X_te

def pred_pls(Xtr, Xte, y, nc, tf):
    nc = max(1, min(nc, Xtr.shape[1]-1))
    yf = np.sqrt(y) if tf=="sqrt" else y.copy()
    pls = PLSRegression(n_components=nc); pls.fit(Xtr, yf)
    p = pls.predict(Xte).ravel()
    return np.clip(p,0,None)**2 if tf=="sqrt" else p

def pred_gbr(Xtr, Xte, y, nc, tf, n_est=200, md=3, lr=0.05):
    nc = max(1, min(nc, Xtr.shape[1]-1))
    yf = np.sqrt(y) if tf=="sqrt" else y.copy()
    pls = PLSRegression(n_components=nc); pls.fit(Xtr, yf)
    Ttr, Tte = pls.transform(Xtr), pls.transform(Xte)
    gbr = GradientBoostingRegressor(n_estimators=n_est, max_depth=md, learning_rate=lr,
        subsample=0.8, min_samples_leaf=5, random_state=42,
        validation_fraction=0.15, n_iter_no_change=50, tol=0.01)
    gbr.fit(Ttr, yf); p = gbr.predict(Tte)
    return np.clip(p,0,None)**2 if tf=="sqrt" else p

def pred_huber(Xtr, Xte, y, nc, tf, eps=1.35):
    nc = max(1, min(nc, Xtr.shape[1]-1))
    yf = np.sqrt(y) if tf=="sqrt" else y.copy()
    pls = PLSRegression(n_components=nc); pls.fit(Xtr, yf)
    Ttr, Tte = pls.transform(Xtr), pls.transform(Xte)
    sc = StandardScaler(); Ttr_s, Tte_s = sc.fit_transform(Ttr), sc.transform(Tte)
    h = HuberRegressor(epsilon=eps, max_iter=200, alpha=0.01)
    h.fit(Ttr_s, yf); p = h.predict(Tte_s)
    return np.clip(p,0,None)**2 if tf=="sqrt" else p

def run_model(X_tr, X_te, y, g, cfg):
    Xtr, Xte = pp(X_tr, X_te, g, cfg["pp"])
    Xtr, Xte = fs(Xtr, Xte, y, cfg.get("fs"))
    t = cfg["type"]
    if t=="pls": return pred_pls(Xtr, Xte, y, cfg.get("nc",4), cfg["tf"])
    elif t=="gbr": return pred_gbr(Xtr, Xte, y, cfg.get("nc",4), cfg["tf"], cfg.get("ne",200), cfg.get("md",3), cfg.get("lr",0.05))
    elif t=="huber": return pred_huber(Xtr, Xte, y, cfg.get("nc",4), cfg["tf"], cfg.get("eps",1.35))

def main():
    t0 = time.time()
    df = load_train(DATA_DIR); sc = get_spectral_columns(df)
    X = df[sc].values; y = df["含水率"].values; g = df["樹種"].values
    logo = LeaveOneGroupOut(); folds = list(logo.split(X, y, g))
    sp = [np.unique(g[te])[0] for _, te in folds]

    print("="*70)
    print("Issue #87: 最終統合")
    print("="*70)

    cfgs = [
        # Cycle9ベスト9
        {"name":"M1:EPO+PLS4+sqrt","pp":"EPO(1)","nc":4,"tf":"sqrt","type":"pls"},
        {"name":"M3:SNV+iPLS50+PLS4+sqrt","pp":"SNV","nc":4,"tf":"sqrt","fs":"iPLS(50)","type":"pls"},
        {"name":"M5:SNV+AsLS+siPLS+PLS5+sqrt","pp":"SNV+AsLS(1e6)","nc":5,"tf":"sqrt","fs":"siPLS(30,3)","type":"pls"},
        {"name":"C1:SNV+iPLS50+PLS5+sqrt","pp":"SNV","nc":5,"tf":"sqrt","fs":"iPLS(50)","type":"pls"},
        {"name":"N10:PMSC+siPLS+GBR+sqrt","pp":"PMSC","nc":4,"tf":"sqrt","fs":"siPLS(30,3)","type":"gbr"},
        {"name":"G9:SG2d+EPO+GBR+raw","pp":"SG2d+EPO(1)","nc":4,"tf":"raw","type":"gbr"},
        {"name":"N2:SNV+SG2d+GBR+raw","pp":"SNV+SG2d","nc":4,"tf":"raw","type":"gbr"},
        {"name":"Gnew1:EPO+GBR150+raw","pp":"EPO(1)","nc":4,"tf":"raw","ne":150,"lr":0.08,"type":"gbr"},
        {"name":"H2:SNV+iPLS50+Huber+sqrt","pp":"SNV","nc":4,"tf":"sqrt","fs":"iPLS(50)","type":"huber"},
        # 新発見
        {"name":"H5:PMSC+siPLS+Huber+sqrt","pp":"PMSC","nc":4,"tf":"sqrt","fs":"siPLS(30,3)","type":"huber"},
        {"name":"H2a:SNV+iPLS50+Huber(1.1)+sqrt","pp":"SNV","nc":4,"tf":"sqrt","fs":"iPLS(50)","eps":1.1,"type":"huber"},
        {"name":"H6:SNV+AsLS+siPLS+Huber+sqrt","pp":"SNV+AsLS(1e6)","nc":4,"tf":"sqrt","fs":"siPLS(30,3)","type":"huber"},
        # 追加
        {"name":"G11:EPO+GBR+raw","pp":"EPO(1)","nc":4,"tf":"raw","type":"gbr"},
        {"name":"M4:PMSC+siPLS+PLS4+sqrt","pp":"PMSC","nc":4,"tf":"sqrt","fs":"siPLS(30,3)","type":"pls"},
        {"name":"M2:SG2d+EPO+PLS3+raw","pp":"SG2d+EPO(1)","nc":3,"tf":"raw","type":"pls"},
    ]

    n = len(cfgs); all_p = [[] for _ in range(n)]
    print(f"\n--- 個別 ({n}) ---\n",flush=True)
    for i, c in enumerate(cfgs):
        t1=time.time(); fr=[]
        for fi,(tr,te) in enumerate(folds):
            try:
                p=run_model(X[tr],X[te],y[tr],g[tr],c); all_p[i].append(p); fr.append(rmse(y[te],p))
            except Exception as e:
                all_p[i].append(np.full(len(te),y[tr].mean())); fr.append(999.0); print(f"  ERR {c['name']} f{fi}: {e}")
        print(f"  [{i+1:>2}/{n}] {c['name']}: {np.mean(fr):.2f} ({time.time()-t1:.1f}s)",flush=True)

    def ev(idx, w=None):
        fr=[]
        for fi,(_,te) in enumerate(folds):
            ps=[all_p[m][fi] for m in idx]
            if w is not None: wn=np.array(w); wn/=wn.sum(); e=sum(wi*p for wi,p in zip(wn,ps))
            else: e=np.mean(ps,axis=0)
            fr.append(rmse(y[te],np.clip(e,0,300)))
        return np.mean(fr),fr

    def ow(idx, nr=40):
        nn=len(idx)
        def obj(w): wn=np.abs(w)/np.sum(np.abs(w)); r,_=ev(idx,wn); return r
        br,bw=np.inf,np.ones(nn)/nn
        for s in range(nr):
            w0=np.random.RandomState(s).dirichlet(np.ones(nn))
            res=minimize(obj,w0,method="Nelder-Mead",options={"maxiter":5000})
            if res.fun<br: br=res.fun; bw=np.abs(res.x)/np.sum(np.abs(res.x))
        return bw,br

    iv=sorted([(ev([i])[0],i) for i in range(n)])
    print("\n個別:"); [print(f"  {r:.2f} | {cfgs[i]['name']}") for r,i in iv]

    top=min(14,n); ti=[i for _,i in iv[:top]]
    res=[]
    for k in [8,9,10,11,12]:
        if k>top: break
        br,bc=np.inf,None
        for combo in combinations(ti,k):
            r,_=ev(list(combo))
            if r<br: br=r; bc=list(combo)
        if bc:
            w,rw=ow(bc); res.append((f"Best{k}",rw,bc,w))
            print(f"  Best{k}: {rw:.4f}",flush=True)
            for i,wi in zip(bc,w):
                if wi>0.02: print(f"    {wi:.3f}: {cfgs[i]['name']}")

    res.sort(key=lambda x:x[1]); best=res[0]
    print(f"\n{'='*70}\nBEST: {best[1]:.4f} ({best[0]})")
    print(f"前ベスト: 14.1594, 改善: {14.1594-best[1]:+.4f}\n{'='*70}")

    _,fr=ev(best[2],best[3])
    print("\nfold:")
    for s,r in zip(sp,fr): print(f"  {s}: {r:.2f}{' ※' if s=='ベイスギ' else ''}")
    nbs=[r for s,r in zip(sp,fr) if s!="ベイスギ"]; print(f"  除ベイスギ: {np.mean(nbs):.4f}")

    rows=[{"method":n,"rmse":r,"models":str([cfgs[i]["name"] for i in c])} for n,r,c,w in res]
    pd.DataFrame(rows).to_csv(OUT_DIR/"issue87_final_results.csv",index=False)
    print(f"\n保存完了. 時間: {time.time()-t0:.0f}s")

if __name__=="__main__": main()
