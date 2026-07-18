"""
Experiment 2 -- Energy efficiency under jamming (closed-loop system simulation).

Sweeps the jammer duty cycle. For each (duty, seed) the SAME jammer schedule is
run under three controllers:
  * base   : correlation power control (inverse-error; ramps to Pmax under jamming)
  * causal : our causal-gated controller (yields to P_safe when Ptx->gamma breaks,
             probes periodically to detect jammer cessation)
  * oracle : knows the true jammer schedule (lower bound on energy)
Reports energy savings vs duty cycle and throughput retention.
"""
import os, csv, time, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import SimConfig, EstimatorConfig, IDX
from network_sim import run, make_jammer_schedule
from controller import CorrelationControl, CausalGatedControl, OracleControl
from metrics import mean_ci

IEEE_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'text.usetex': False 
}

OUT = "results"; os.makedirs(OUT, exist_ok=True)


def run_exp(duties=(0.1, 0.2, 0.3, 0.4, 0.5), n_seeds=4, n_steps=3000, W=90):
    rows = []
    sav_c = {d: [] for d in duties}; sav_o = {d: [] for d in duties}
    thr_ret = {d: [] for d in duties}; real_duty = {d: [] for d in duties}
    for d in duties:
        for s in range(n_seeds):
            cfg = SimConfig(n_steps=n_steps, seed=s)
            cfg.estimator = EstimatorConfig(window_W=W, cd_iters=45)
            rng = np.random.default_rng(1000 + s)
            jam = make_jammer_schedule(n_steps, d, rng)
            rb = run(cfg, CorrelationControl(cfg.power), jammer_on=jam, seed=s)
            rc = run(cfg, CausalGatedControl(cfg.power, cfg.estimator), jammer_on=jam, seed=s)
            ro = run(cfg, OracleControl(jam, cfg.power), jammer_on=jam, seed=s)
            sav_c[d].append(100 * (rb["energy_J"] - rc["energy_J"]) / rb["energy_J"])
            sav_o[d].append(100 * (rb["energy_J"] - ro["energy_J"]) / rb["energy_J"])
            thr_ret[d].append(100 * rc["mean_thr_mbps"] / rb["mean_thr_mbps"])
            real_duty[d].append(100 * jam.mean())

    with plt.rc_context(IEEE_STYLE):

        # ---------- savings vs duty ----------
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        dd = np.array([np.mean(real_duty[d]) for d in duties]) / 100.0
        mc = np.array([mean_ci(sav_c[d]) for d in duties]); mo = np.array([mean_ci(sav_o[d]) for d in duties])
        ax.errorbar(dd, mc[:, 0], yerr=mc[:, 1], fmt="-o", capsize=3, label="Causal-gated (ours)", color="C0")
        ax.errorbar(dd, mo[:, 0], yerr=mo[:, 1], fmt="--s", capsize=3, label="Oracle (knows jammer)", color="C2")
        ax.axhline(45, color="gray", ls=":", lw=1, label="reference 45%")
        ax.set_xlabel("jammer duty cycle"); ax.set_ylabel("energy savings [%]")
        ax.set_title("Energy Savings vs Jamming Intensity")
        ax.legend(loc="best", framealpha=0.8); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(f"{OUT}/fig_exp2_savings_vs_duty.pdf", dpi=600, bbox_inches='tight'); plt.close(fig)

        # ---------- representative time series (duty 0.3, seed 0) ----------
        cfg = SimConfig(n_steps=n_steps, seed=0); cfg.estimator = EstimatorConfig(window_W=W, cd_iters=45)
        rng = np.random.default_rng(1000)
        jam = make_jammer_schedule(n_steps, 0.3, rng)
        rb = run(cfg, CorrelationControl(cfg.power), jammer_on=jam, seed=0)
        rc = run(cfg, CausalGatedControl(cfg.power, cfg.estimator), jammer_on=jam, seed=0)
        fig, axs = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True)
        tt = np.arange(n_steps)
        axs[0].plot(tt, rb["X"][:, IDX["Ptx"]], color="C3", lw=0.8, label="base (correlation)")
        axs[0].plot(tt, rc["X"][:, IDX["Ptx"]], color="C0", lw=0.8, label="causal-gated")
        axs[0].set_ylabel("Tx power [dBm]"); axs[0].legend(fontsize=8, loc="upper right")
        axs[1].plot(tt, rb["X"][:, IDX["gamma"]], color="C3", lw=0.8, label="base SINR")
        axs[1].plot(tt, rc["X"][:, IDX["gamma"]], color="C0", lw=0.8, label="causal SINR")
        axs[1].set_ylabel("SINR [dB]"); axs[1].set_xlabel("TTI"); axs[1].legend(fontsize=8, loc="upper right")
        for a in axs:
            ymin, ymax = a.get_ylim()
            a.fill_between(tt, ymin, ymax, where=jam, color="orange", alpha=0.15)
            a.set_ylim(ymin, ymax)
        axs[0].set_title("Tx Power (Correlation vs. Causal-Gated) under Jamming (shaded)")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_exp2_timeseries.pdf", dpi=600, bbox_inches='tight'); plt.close(fig)

    # ---------- table ----------
    with open(f"{OUT}/table_exp2.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"EXP2 -- energy efficiency under jamming (mean +/- 95pct CI over {n_seeds} seeds)"])
        w.writerow(["duty_cycle", "causal_savings_%", "oracle_savings_%", "throughput_retention_%"])
        for d in duties:
            mc_ = mean_ci(sav_c[d]); mo_ = mean_ci(sav_o[d]); tr_ = mean_ci(thr_ret[d])
            w.writerow([d, f"{mc_[0]:.1f}+/-{mc_[1]:.1f}", f"{mo_[0]:.1f}+/-{mo_[1]:.1f}",
                        f"{tr_[0]:.1f}+/-{tr_[1]:.1f}"])
    print("EXP2 done.")
    for d in duties:
        mc_ = mean_ci(sav_c[d]); tr_ = mean_ci(thr_ret[d])
        print("  duty %.1f: causal savings %.1f+/-%.1f%%  throughput retained %.1f%%"
              % (d, mc_[0], mc_[1], tr_[0]))


if __name__ == "__main__":
    t0 = time.time(); run_exp(); print("wall %.1fs" % (time.time() - t0))