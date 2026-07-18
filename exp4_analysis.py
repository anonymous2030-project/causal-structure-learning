import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metrics import mean_ci
OUT="results"
IEEE={'font.family':'serif','font.size':10,'axes.labelsize':10,'axes.titlesize':10,
      'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8}

def load(path):
    with open(path) as f:
        rows=list(csv.reader(f))
    return rows[0], rows[1:]

# ---------- seed convergence ----------
def seed_conv(duty_tag):
    hdr,rows=load(f"{OUT}/table_sens_seeds_d{duty_tag}.csv")
    sav=np.array([float(r[2]) for r in rows]); ret=np.array([float(r[3]) for r in rows])
    ks=[2,4,6,8,12,16,20,len(sav)]
    out=[]
    for k in ks:
        sm,sh=mean_ci(sav[:k]); rm,rh=mean_ci(ret[:k])
        out.append((k,sm,sh,rm,rh))
    return sav,ret,out

conv={}
with open(f"{OUT}/table_sens_seeds_convergence.csv","w",newline="") as fh:
    w=csv.writer(fh)
    w.writerow(["duty","n_seeds","savings_mean","savings_ci95_halfwidth",
                "retention_mean","retention_ci95_halfwidth"])
    for tag,duty in [("30",0.3),("50",0.5)]:
        sav,ret,out=seed_conv(tag); conv[tag]=(sav,ret,out)
        for k,sm,sh,rm,rh in out:
            w.writerow([duty,k,round(sm,2),round(sh,2),round(rm,2),round(rh,2)])
        print(f"duty {duty}: 4-seed savings {out[1][1]:.1f}+/-{out[1][2]:.1f}  ->  "
              f"24-seed {out[-1][1]:.1f}+/-{out[-1][2]:.1f}   |  "
              f"4-seed retention {out[1][3]:.1f}+/-{out[1][4]:.1f} -> "
              f"24-seed {out[-1][3]:.1f}+/-{out[-1][4]:.1f}")

# ---------- Fig A: window sweep ----------
hdr,rows=load(f"{OUT}/table_sens_W.csv")
Wv=np.array([float(r[0]) for r in rows]); F1=np.array([float(r[1]) for r in rows])
F1c=np.array([float(r[2]) for r in rows]); det=np.array([float(r[3]) for r in rows])
dly=np.array([float(r[4]) for r in rows]); dlyc=np.array([float(r[5]) for r in rows])
with plt.rc_context(IEEE):
    fig,ax=plt.subplots(figsize=(3.5,2.6))
    ax.errorbar(Wv,F1,yerr=F1c,fmt="-o",color="C0",capsize=3,label="structure F1")
    ax.plot(Wv,det,"-^",color="C2",label="jamming detect rate")
    ax.set_xlabel("window length $W$ [TTIs]"); ax.set_ylabel("F1 / detection rate")
    ax.set_ylim(-0.03,1.08); ax.grid(alpha=0.3)
    ax2=ax.twinx()
    ax2.errorbar(Wv,dly,yerr=dlyc,fmt="--s",color="C3",capsize=3,label="detect delay")
    ax2.set_ylabel("detection delay [TTIs]",color="C3"); ax2.tick_params(axis='y',colors="C3")
    ax.axvspan(60,120,color="green",alpha=0.08)
    l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
    ax.legend(l1+l2,la1+la2,loc="lower center",framealpha=0.85,fontsize=6)
    ax.set_title("Sensitivity to window length $W$")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_sens_W.png",dpi=200,bbox_inches='tight')
    fig.savefig(f"{OUT}/fig_sens_W.pdf",dpi=600,bbox_inches='tight'); plt.close(fig)

# ---------- Fig B: delta sweep ----------
hdr,rows=load(f"{OUT}/table_sens_delta.csv")
dv=np.array([float(r[0]) for r in rows]); f1=np.array([float(r[1]) for r in rows])
f1c=np.array([float(r[2]) for r in rows]); pr=np.array([float(r[3]) for r in rows])
rc=np.array([float(r[4]) for r in rows])
with plt.rc_context(IEEE):
    fig,ax=plt.subplots(figsize=(3.5,2.6))
    ax.errorbar(dv,f1,yerr=f1c,fmt="-o",color="C0",capsize=3,label="F1")
    ax.plot(dv,pr,"--s",color="C2",label="precision")
    ax.plot(dv,rc,"--^",color="C3",label="recall")
    ax.axvspan(0.06,0.18,color="green",alpha=0.08,label="stable plateau")
    ax.set_xlabel(r"binarization threshold $\delta$"); ax.set_ylabel("score")
    ax.set_ylim(0.78,1.03); ax.grid(alpha=0.3); ax.legend(loc="lower center",fontsize=7,framealpha=0.85)
    ax.set_title(r"Sensitivity to threshold $\delta$")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_sens_delta.png",dpi=200,bbox_inches='tight')
    fig.savefig(f"{OUT}/fig_sens_delta.pdf", dpi=600,bbox_inches='tight'); plt.close(fig)

# ---------- Fig C: seed convergence ----------
with plt.rc_context(IEEE):
    fig,ax=plt.subplots(figsize=(3.5,2.6))
    for tag,duty,col in [("30",0.3,"C0"),("50",0.5,"C3")]:
        out=conv[tag][2]; ks=[o[0] for o in out]; hw=[o[2] for o in out]
        ax.plot(ks,hw,"-o",color=col,label=f"duty {duty}")
    kk=np.linspace(2,24,50); ref=hw[0]* (np.sqrt(ks[0])/np.sqrt(kk))
    ax.plot(kk,ref,"k:",lw=1,label=r"$\propto 1/\sqrt{k}$")
    #ax.axvline(4,color="gray",ls="--",lw=1)
    ax.set_xlabel("number of seeds $k$"); ax.set_ylabel("95% CI half-width [pp]\n(energy savings)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7,framealpha=0.85)
    ax.set_title("Seed-count vs estimate precision")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_sens_seeds.png",dpi=200,bbox_inches='tight')
    fig.savefig(f"{OUT}/fig_sens_seeds.pdf",dpi=600,bbox_inches='tight'); plt.close(fig)
print("figures written")