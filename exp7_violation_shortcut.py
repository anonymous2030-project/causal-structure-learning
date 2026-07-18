"""
Exp 7 (Task 2) -- Robustness to unmodeled cross-layer (L1->L3) shortcuts.

We deliberately violate the mediated-effect assumption: when SINR gamma drops below
a catastrophic threshold (deep fade), packets pile up DIRECTLY at the app-layer
queue (an L1->L3 edge that BYPASSES the MAC layer). The structural mask still
forbids L1->L3, so the estimator cannot represent this edge. We then check that
(a) global structure recovery drifts only mildly, and (b) the control edge
P_tx->gamma -- and the jamming detector built on it -- stay solid.
"""
import numpy as np, csv, os
import scm
from config import IDX, N_VARS, EstimatorConfig
from estimator import HierarchicalCausalDiscovery
from scm import true_graph
from metrics import edge_prf, shd, mean_ci

OUT="results"; os.makedirs(OUT, exist_ok=True)
g,lp,re,T,lam,Q,ptx = (IDX['gamma'],IDX['Lpath'],IDX['Reff'],IDX['T'],IDX['lam'],IDX['Q'],IDX['Ptx'])

def generate_shortcut(n, break_step, seed=0, benign_step=None, jam_std=1.2,
                      shortcut_gain=0.0, gamma_crit=-1.0):
    """scm.generate + an L1->L3 shortcut: gamma<gamma_crit -> direct queue jump."""
    rng=np.random.default_rng(seed)
    if benign_step is None: benign_step=max(0,break_step-200)
    X=np.zeros((n,N_VARS)); jam_on=np.zeros(n,bool); jam_on[break_step:]=True
    lp_level=0.0; lam_state=0.0
    for t in range(1,n):
        xp=X[t-1]
        lp_level=0.6*lp_level+rng.normal(0,0.6); Lpath=lp_level
        if rng.random()<0.02: lam_state=rng.choice([-1.0,1.0])
        lam_val=0.95*lam_state+rng.normal(0,0.5)
        Ptx=rng.normal(0,1.0)
        a_ptx=0.0 if jam_on[t] else 0.6
        noise_g=rng.normal(0, jam_std if jam_on[t] else 0.6)
        gamma=0.45*xp[g]+a_ptx*xp[ptx]-0.5*xp[lp]+noise_g
        Reff=0.40*xp[re]+0.60*xp[g]+rng.normal(0,0.4)
        Tn=0.45*xp[T]+0.60*xp[re]+rng.normal(0,0.4)
        Qn=0.50*xp[Q]+0.60*xp[lam]-0.50*xp[T]+rng.normal(0,0.4)
        # --- UNMODELED L1->L3 SHORTCUT (bypasses MAC) ---
        if shortcut_gain>0 and xp[g]<gamma_crit:
            Qn += shortcut_gain*(gamma_crit-xp[g])      # deep fade -> direct queue jump
        X[t]=[gamma,Lpath,Reff,Tn,lam_val,Qn,Ptx]
    return X, jam_on, true_graph

def detect_stream(X, W, cfg, break_step):
    est=HierarchicalCausalDiscovery(cfg); fired=None
    for t in range(W+1, len(X)):
        if est.gamma_weight(X[t-W-1:t])["link_broken"]:
            fired=t; break
    return fired

def run_exp(gains=(0.0,1.0,2.0,4.0), n_seeds=8, n_steps=800, W=120, H=100,
            break_step=520, benign_step=350):
    cfg=EstimatorConfig(window_W=W, cd_iters=45, delta_edge=0.10)
    rows=[]
    for sg in gains:
        gF1=[]; ctrl_ok=[]; benign_fa=0; jam_det=0
        for e in range(n_seeds):
            # jamming episode (shortcut active): structure + detection
            Xj,jam,tg = generate_shortcut(n_steps, break_step, seed=300+e,
                                          benign_step=n_steps+5, shortcut_gain=sg)
            est=HierarchicalCausalDiscovery(cfg)
            for t in range(break_step-60, break_step): est.update(Xj[t-W-1:t])
            G=est.G_prev
            gF1.append(edge_prf(tg(False), G)[2])
            ctrl_ok.append(1.0 if G[g,ptx]==1 else 0.0)         # control edge present pre-jam
            fj=detect_stream(Xj, W, cfg, break_step)
            jam_det += 1 if (fj is not None and fj>=break_step and fj<break_step+H) else 0
            # benign episode (shortcut active, NO jamming): false alarm?
            Xb,_,_ = generate_shortcut(n_steps, n_steps+5, seed=700+e,
                                       benign_step=benign_step, shortcut_gain=sg)
            fb=detect_stream(Xb, W, cfg, benign_step)
            benign_fa += 1 if (fb is not None and fb>=benign_step) else 0
        TP=jam_det; FP=benign_fa; FN=n_seeds-jam_det
        det_P = TP/(TP+FP) if TP+FP else 1.0
        det_R = TP/(TP+FN) if TP+FN else 1.0
        det_F1= 2*det_P*det_R/(det_P+det_R) if det_P+det_R else 0.0
        gm,gh=mean_ci(gF1)
        rows.append([sg, gm, gh, float(np.mean(ctrl_ok)), det_P, det_R, det_F1])
        print("shortcut_gain=%.1f | global F1=%.2f+/-%.2f | ctrl-edge P_tx->g recovered=%.0f%% | "
              "detector P=%.2f R=%.2f F1=%.2f"
              %(sg, gm, gh, 100*np.mean(ctrl_ok), det_P, det_R, det_F1))
    with open(f"{OUT}/table_exp7_violation.csv","w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["shortcut_gain","global_F1","global_F1_ci",
            "ctrl_edge_recovery_rate","detector_precision","detector_recall","detector_F1"])
        w.writerows([[r[0]]+[round(v,3) for v in r[1:]] for r in rows])
    print("EXP7 done.")

if __name__=="__main__":
    import time; t0=time.time(); run_exp(); print("wall %.1fs"%(time.time()-t0))

def make_fig():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    IEEE={'font.family':'serif','font.size':10,'axes.labelsize':10,'axes.titlesize':10,
          'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8}
    rows=list(csv.reader(open(f"{OUT}/table_exp7_violation.csv")))[1:]
    sg=[float(r[0]) for r in rows]; gF1=[float(r[1]) for r in rows]; gci=[float(r[2]) for r in rows]
    ctrl=[float(r[3]) for r in rows]; detF1=[float(r[6]) for r in rows]
    with plt.rc_context(IEEE):
        fig,ax=plt.subplots(figsize=(3.5,2.6))
        ax.errorbar(sg,gF1,yerr=gci,fmt="-o",color="C3",capsize=3,label="global structure F1")
        ax.plot(sg,ctrl,"-^",color="C0",label="control edge $P_{tx}\\!\\to\\!\\gamma$ recovery")
        ax.plot(sg,detF1,"-s",color="C2",label="jamming detector F1")
        ax.set_xlabel("cross-layer shortcut strength (L1$\\to$L3)"); ax.set_ylabel("score / rate")
        ax.set_ylim(0.6,1.05); ax.grid(alpha=0.3)
        ax.set_title("Robustness to Unmodeled Cross-Layer Coupling")
        ax.legend(fontsize=7,loc="lower left",framealpha=0.85); fig.tight_layout()
        fig.savefig(f"{OUT}/fig_exp7_violation.png",dpi=200,bbox_inches='tight')
        fig.savefig(f"{OUT}/fig_exp7_violation.pdf",bbox_inches='tight'); plt.close(fig)
    print("fig written")

if __name__=="__main__":
    make_fig()