"""
Exp 6 -- Multi-user O(K) latency scaling.

Extends the 7-KPI single-user model to K coexisting UEs (state dim N = 7K) with a
BLOCK-DIAGONAL mask: user k's KPIs may only parent user k's KPIs (cross-user edges
forbidden). Because each target's admissible-parent count d_eff stays constant,
masked per-step inference is O(K); an unmasked Lasso (every KPI a candidate parent
of every other) is O(K^2). We time both vs K and check the 10 ms RIC budget.
"""
import os, csv, time, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scm  # circular-import-safe order
from estimator import build_mask, _lasso_cd, _detrend
from config import ENDO_IDX, IDX
from scm import generate
from metrics import percentiles

OUT = "results"; os.makedirs(OUT, exist_ok=True)
IEEE = {'font.family':'serif','font.size':10,'axes.labelsize':10,'axes.titlesize':10,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8}

def multiuser_mask(K):
    """Block-diagonal replication of the single-user mask over K users (N=7K)."""
    M1,_ = build_mask()                      # 7x7 single-user mask
    N = 7*K
    M = np.zeros((N,N), int)
    for k in range(K):
        M[k*7:(k+1)*7, k*7:(k+1)*7] = M1     # within-user block only
    targets = [k*7+j for k in range(K) for j in ENDO_IDX]   # 4K endogenous targets
    parents = {t: np.where(M[:,t]==1)[0] for t in targets}
    return M, targets, parents

def gen_multiuser(K, L, seed0=0):
    """Stack K independent single-user KPI streams -> (L, 7K)."""
    cols = [generate(L, L+5, seed=seed0+k, benign_step=L+5)[0] for k in range(K)]
    return np.hstack(cols)

def time_step(win, targets, parents, masked, lam=0.05, iters=45):
    Xl=_detrend(win[:-1]); sd=Xl.std(0); sd[sd<1e-9]=1; Xs=Xl/sd
    Yc=_detrend(win[1:])
    t0=time.perf_counter()
    for t in targets:
        y=Yc[:,t]; ys=y.std() or 1.0
        cols = parents[t] if masked else slice(None)
        _lasso_cd(Xs[:,cols], y/ys, lam, iters, 1e-4)
    return (time.perf_counter()-t0)*1e3

def run_exp(Ks=(1,2,4,8,12,16), W=120, reps=20):
    rows=[]
    res={'masked':{}, 'full':{}}
    for K in Ks:
        M,targets,parents = multiuser_mask(K)
        X = gen_multiuser(K, W+1+reps, seed0=10*K)
        ctrl_targets=[k*7+IDX['gamma'] for k in range(K)]  # control-edge (gamma) row only
        mt, ft, ct = [], [], []
        for r in range(reps):
            win = X[r:r+W+1]
            mt.append(time_step(win, targets, parents, masked=True))
            ft.append(time_step(win, targets, parents, masked=False))
            ct.append(time_step(win, ctrl_targets, parents, masked=True))
        res['masked'][K]=percentiles(mt); res['full'][K]=percentiles(ft); res.setdefault('ctrl',{})[K]=percentiles(ct)
        rows.append([K, 7*K, res['masked'][K]['p50'], res['masked'][K]['p95'],
                     res['full'][K]['p50'], res['full'][K]['p95']])
        print("K=%2d (N=%3d)  masked p50=%.2f p95=%.2f ms | unmasked p50=%.2f p95=%.2f ms"
              %(K,7*K,res['masked'][K]['p50'],res['masked'][K]['p95'],
                res['full'][K]['p50'],res['full'][K]['p95']))
    with open(f"{OUT}/table_exp6_multiuser.csv","w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["K_users","N_kpis","ctrl_edge_p50_ms","masked_full_p50_ms","masked_full_p95_ms","unmasked_p50_ms","unmasked_p95_ms"])
        for K in Ks:
            w.writerow([K,7*K,round(res['ctrl'][K]['p50'],3),round(res['masked'][K]['p50'],3),
                        round(res['masked'][K]['p95'],3),round(res['full'][K]['p50'],3),round(res['full'][K]['p95'],3)])
    # figure
    Ks=list(Ks)
    with plt.rc_context(IEEE):
        fig,ax=plt.subplots(figsize=(3.5,2.6))
        mp50=[res['masked'][K]['p50'] for K in Ks]; mp95=[res['masked'][K]['p95'] for K in Ks]
        fp50=[res['full'][K]['p50'] for K in Ks]; fp95=[res['full'][K]['p95'] for K in Ks]
        cp50=[res['ctrl'][K]['p50'] for K in Ks]
        ax.plot(Ks,cp50,"-^",color="C2",ms=4,lw=1.5,label="control-edge only p50 ($O(K)$)")
        ax.plot(Ks,mp50,"-o",color="C0",ms=4,lw=1.5,label="full-graph masked p50 ($O(K)$)")
        ax.plot(Ks,mp95,"--o",color="C0",ms=3,lw=1,alpha=0.8,label="masked p95")
        ax.plot(Ks,fp50,"-s",color="C3",ms=4,lw=1.5,label="unmasked p50 ($O(K^2)$)")
        ax.plot(Ks,fp95,"--s",color="C3",ms=3,lw=1,alpha=0.8,label="unmasked p95")
        ax.axhline(10,color="green",ls="-",lw=1.4,label="10 ms RIC budget")
        ax.set_yscale("log")
        ax.set_xlabel("number of active UEs $K$  (KPIs $N=7K$)"); ax.set_ylabel("per-step inference latency [ms]")
        ax.set_title("Multi-User Inference Latency Scaling"); ax.grid(alpha=0.3,which="both")
        ax.legend(fontsize=7,loc="upper left",framealpha=0.85); fig.tight_layout()
        fig.savefig(f"{OUT}/fig_exp6_multiuser.png",dpi=200,bbox_inches='tight')
        fig.savefig(f"{OUT}/fig_exp6_multiuser.pdf",bbox_inches='tight'); plt.close(fig)
    print("EXP6 done.")

if __name__=="__main__":
    t0=time.time(); run_exp(); print("wall %.1fs"%(time.time()-t0))