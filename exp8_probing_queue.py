"""
Exp 8 (Task 3) -- Probing-interval vs queue stability / energy tradeoff.

The causal gate yields to P_safe under jamming but PROBES periodically (briefly
restoring power) to detect cessation. A short probe interval serves the queue more
often (low backlog) but spends energy; a long interval saves more energy but lets
the app-layer queue accumulate. We sweep probing_interval_TTI and map peak/mean
queue against total energy saved to locate the stability sweet-spot.
"""
import os, csv, time, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import scm  # load first: avoids estimator<->scm circular import
from config import SimConfig, EstimatorConfig, IDX
from network_sim import run, make_jammer_schedule
from controller import CorrelationControl, CausalGatedControl

OUT="results"; os.makedirs(OUT, exist_ok=True)
IEEE={'font.family':'serif','font.size':10,'axes.labelsize':10,'axes.titlesize':10,
      'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8}

def run_exp(probes=(10,20,30,50,70,100), n_seeds=3, n_steps=3000, duty=0.4, W=90):
    peakQ={p:[] for p in probes}; meanQ={p:[] for p in probes}
    sav={p:[] for p in probes}; ret={p:[] for p in probes}
    for s in range(n_seeds):
        cfg=SimConfig(n_steps=n_steps, seed=s)
        cfg.traffic.Q_max_bits=1e8  # large buffer: expose true queue accumulation vs probe interval
        jam=make_jammer_schedule(n_steps, duty, np.random.default_rng(1000+s))
        base=run(cfg, CorrelationControl(cfg.power), jammer_on=jam, seed=s)
        Eb=base["energy_J"]; Tb=base["mean_thr_mbps"]
        for p in probes:
            cfg.estimator=EstimatorConfig(window_W=W, cd_iters=45)
            ctrl=CausalGatedControl(cfg.power, cfg.estimator, probe_period=p)
            rc=run(cfg, ctrl, jammer_on=jam, seed=s)
            Qj=rc["X"][jam, IDX["Q"]]/1e6            # queue [Mbits] during jamming
            peakQ[p].append(float(Qj.max())); meanQ[p].append(float(Qj.mean()))
            sav[p].append(100*(Eb-rc["energy_J"])/Eb)
            ret[p].append(100*rc["mean_thr_mbps"]/Tb)
        print("seed %d done"%s)
    probes=list(probes)
    def ms(d): return [np.mean(d[p]) for p in probes]
    pk, mn, sv, rt = ms(peakQ), ms(meanQ), ms(sav), ms(ret)
    with open(f"{OUT}/table_exp8_probing.csv","w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["probe_interval_TTI","peak_queue_Mbit","mean_queue_Mbit",
                                      "energy_saved_pct","throughput_retention_pct"])
        for i,p in enumerate(probes):
            w.writerow([p,round(pk[i],3),round(mn[i],3),round(sv[i],2),round(rt[i],2)])
    for i,p in enumerate(probes):
        print("probe=%3d TTI | peakQ=%.2f Mbit meanQ=%.2f | energy saved=%.1f%% | thr retained=%.1f%%"
              %(p,pk[i],mn[i],sv[i],rt[i]))
    # dual-axis figure
    with plt.rc_context(IEEE):
        fig,ax=plt.subplots(figsize=(3.6,2.7))
        l1=ax.plot(probes,pk,"-o",color="C3",label="peak queue")[0]
        l2=ax.plot(probes,mn,"--o",color="C1",label="mean queue")[0]
        ax.set_xlabel("probing interval [TTI]"); ax.set_ylabel("queue backlog [Mbit]",color="C3")
        ax.tick_params(axis='y',colors="C3"); ax.grid(alpha=0.3)
        ax2=ax.twinx()
        l3=ax2.plot(probes,sv,"-s",color="C0",label="energy saved")[0]
        ax2.set_ylabel("energy saved [%]",color="C0"); ax2.tick_params(axis='y',colors="C0")
        ax.legend(handles=[l1,l2,l3],fontsize=7,loc="center right",framealpha=0.85)
        ax.set_title("Probing Interval: Queue vs Energy")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_exp8_probing.png",dpi=200,bbox_inches='tight')
        fig.savefig(f"{OUT}/fig_exp8_probing.pdf",bbox_inches='tight'); plt.close(fig)
    print("EXP8 done.")

if __name__=="__main__":
    t0=time.time(); run_exp(); print("wall %.1fs"%(time.time()-t0))