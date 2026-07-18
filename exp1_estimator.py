"""
Experiment 1 -- Structure recovery + change-detection SPECIFICITY.

Protocol (standard in detection theory): every detector is first CALIBRATED to a
common false-alarm rate (alpha) on purely STATIONARY data, then evaluated on
  * benign-only episodes (a legitimate path-loss surge; mechanism intact), and
  * jamming-only episodes (the Ptx->gamma mechanism breaks).
A good detector rejects the benign event yet detects jamming. Marginal detectors
(CUSUM / correlation / neural-residual) must false-alarm on the benign swing to be
sensitive enough to catch jamming; the causal detector separates them by design.
"""
import os, csv, time, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from config import EstimatorConfig, IDX
from scm import generate
from estimator import HierarchicalCausalDiscovery
from baselines import CusumDetector, CorrelationDetector, NeuralResidualDetector, _HAS_TORCH
from metrics import edge_prf, mean_ci

IEEE_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'text.usetex': False  # True if LaTeX is installed
}

OUT = "results"; os.makedirs(OUT, exist_ok=True)
NEURAL = "LSTM" if _HAS_TORCH else "MLP"
METHODS = ["Causal (ours)", "CUSUM", "Correlation", NEURAL]


def causal_scores(X, W, cfg):
    """Per-step anomaly score for the causal detector = -w (w=|A[gamma,Ptx]|)."""
    est = HierarchicalCausalDiscovery(cfg)
    sc = np.full(len(X), np.nan)
    for t in range(W + 1, len(X)):
        sc[t] = -est.gamma_weight(X[t - W - 1:t])["w_ptx_gamma"]
    return sc


def baseline_scores(kind, X, W):
    det = {"CUSUM": CusumDetector(warmup=W),
           "Correlation": CorrelationDetector(),
           NEURAL: NeuralResidualDetector(use_lstm=_HAS_TORCH, calib=W)}[kind]
    sc = np.full(len(X), np.nan)
    for t in range(W + 1, len(X)):
        det.update(X[t - W:t]); sc[t] = det.stat
    return sc


def scores_for(method, X, W, cfg):
    return causal_scores(X, W, cfg) if method == METHODS[0] else baseline_scores(method, X, W)


def first_cross(sc, thr, dwell, start):
    run = 0
    for t in range(start, len(sc)):
        if not np.isnan(sc[t]) and sc[t] > thr:
            run += 1
            if run >= dwell:
                return t - dwell + 1
        else:
            run = 0
    return None


def run(n_cal=4, n_eval=8, n_steps=800, benign_step=350, break_step=520,
        W=150, H=80, alpha=0.01, dwell=5):
    cfg = EstimatorConfig(window_W=W, cd_iters=45, delta_edge=0.10)
    settle = W + 20

    # ---------- calibrate thresholds on stationary data ----------
    thr = {m: [] for m in METHODS}
    for e in range(n_cal):
        Xs, *_ = generate(n_steps, n_steps + 5, seed=900 + e, benign_step=n_steps + 5)
        for m in METHODS:
            sc = scores_for(m, Xs, W, cfg)
            vals = sc[settle:]; vals = vals[~np.isnan(vals)]
            thr[m].append(np.quantile(vals, 1 - alpha))
    thr = {m: float(np.mean(v)) for m, v in thr.items()}

    # ---------- evaluate ----------
    benign_fa = {m: [] for m in METHODS}
    jam_det = {m: [] for m in METHODS}
    jam_delay = {m: [] for m in METHODS}
    f1_list = []
    ex = {}  # store one example episode for the events figure
    sc_ben = {m: [] for m in METHODS}; sc_jam = {m: [] for m in METHODS}

    for e in range(n_eval):
        Xb, _, ben, _ = generate(n_steps, n_steps + 5, seed=200 + e, benign_step=benign_step)
        Xj, jam, _, tg = generate(n_steps, break_step, seed=300 + e, benign_step=n_steps + 5)
        for m in METHODS:
            scb = scores_for(m, Xb, W, cfg)
            cb = first_cross(scb, thr[m], dwell, benign_step)
            benign_fa[m].append(1.0 if (cb is not None and cb < benign_step + H) else 0.0)
            scj = scores_for(m, Xj, W, cfg)
            cj = first_cross(scj, thr[m], dwell, break_step)
            det = cj is not None and cj < break_step + H
            jam_det[m].append(1.0 if det else 0.0)
            jam_delay[m].append((cj - break_step) if det else H)
            sc_ben[m].append(scb); sc_jam[m].append(scj)
            if e == 0:
                ex[m] = (scb, scj)
        # structure recovery on the pre-jamming segment
        est = HierarchicalCausalDiscovery(cfg)
        for t in range(break_step - 60, break_step):
            est.update(Xj[t - W - 1:t])
        f1_list.append(edge_prf(tg(False), est.G_prev)[2])

    with plt.rc_context(IEEE_STYLE):

        # ---------- weight-collapse figure ----------
        ws = np.full((n_eval, n_steps), np.nan)
        for e in range(n_eval):
            Xj, jam, _, tg = generate(n_steps, break_step, seed=300 + e, benign_step=benign_step)
            est = HierarchicalCausalDiscovery(cfg)
            for t in range(W + 1, n_steps):
                ws[e, t] = est.gamma_weight(Xj[t - W - 1:t])["w_ptx_gamma"]
        rel = np.arange(n_steps) - break_step
        m_ = np.nanmean(ws, 0); hw = 1.96 * np.nanstd(ws, 0) / np.sqrt(n_eval)
        sl = (rel > -(break_step - settle)) & (rel < n_steps - break_step)
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.plot(rel[sl], m_[sl], "C0", lw=2, label=r"$w_t=\hat A[P_{tx}\!\to\!\gamma]$")
        ax.fill_between(rel[sl], (m_ - hw)[sl], (m_ + hw)[sl], color="C0", alpha=0.25)
        ax.axvline(benign_step - break_step, color="orange", ls="-.", label="benign surge")
        ax.axvline(0, color="r", ls="--", label="jamming onset")
        ax.axhline(cfg.w_break_thresh, color="k", ls=":", label="break threshold")
        ax.set_xlabel("TTI relative to jamming onset"); ax.set_ylabel("causal edge weight")
        ax.set_title("Control Edge Collapse", fontsize=10)
        ax.legend(loc="best", framealpha=0.8); fig.tight_layout()
        fig.savefig(f"{OUT}/fig_exp1_weight_collapse.pdf", dpi=600, bbox_inches='tight'); plt.close(fig)

        # ---------- operating curve: jamming detection rate vs benign false-alarm ----------
        def rate(score_list, thr_, start):
            hits = [1.0 if first_cross(sc, thr_, dwell, start) is not None
                    and first_cross(sc, thr_, dwell, start) < start + H else 0.0
                    for sc in score_list]
            return float(np.mean(hits))
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        for m in METHODS:
            pooled = np.concatenate([np.concatenate([sc_ben[m][e], sc_jam[m][e]])
                                    for e in range(n_eval)])
            pooled = pooled[~np.isnan(pooled)]
            lo, hi = np.percentile(pooled, 1), np.percentile(pooled, 99.5)
            ths = np.linspace(lo, hi, 45)
            bfa = np.array([rate(sc_ben[m], th, benign_step) for th in ths])
            jdr = np.array([rate(sc_jam[m], th, break_step) for th in ths])
            order = np.argsort(bfa)
            ax.plot(bfa[order], jdr[order], "-o", ms=3, label=m)
        ax.set_xlabel("benign false-alarm rate"); ax.set_ylabel("jamming detection rate")
        ax.set_title("Detection vs Benign False Alarms")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05); ax.legend(loc="best", framealpha=0.8)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_exp1_roc.pdf", dpi=600, bbox_inches='tight'); plt.close(fig)

        # ---------- specificity bar chart ----------
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        x = np.arange(len(METHODS)); wbar = 0.38
        bfa = [np.mean(benign_fa[m]) for m in METHODS]
        jdr = [np.mean(jam_det[m]) for m in METHODS]
        ax.bar(x - wbar/2, bfa, wbar, label="benign false-alarm rate", color="C3")
        ax.bar(x + wbar/2, jdr, wbar, label="jamming detection rate", color="C2")
        ax.set_xticks(x); ax.set_xticklabels(METHODS, fontsize=8, rotation=15)
        ax.set_ylim(0, 1.05); ax.set_ylabel("rate")
        ax.set_title(f"Specificity at Matched Stationary FAR={alpha:.0%}")
        ax.legend(loc="best", framealpha=0.8); fig.tight_layout()
        fig.savefig(f"{OUT}/fig_exp1_specificity.pdf", dpi=600, bbox_inches='tight'); plt.close(fig)

        # ---------- events timeline (one seed) ----------
        fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True)
        for m in METHODS:
            scb, scj = ex[m]
            axs[0].plot(scb - thr[m], label=m, lw=1)
            axs[1].plot(scj - thr[m], label=m, lw=1)
        for a, title, ev in [(axs[0], "Benign episode (should NOT fire)", benign_step),
                            (axs[1], "Jamming episode (should fire)", break_step)]:
            a.axhline(0, color="k", ls=":", lw=1); a.axvline(ev, color="r", ls="--", lw=1)
            a.set_ylabel("score - threshold", fontsize=8); a.set_title(title, fontsize=9)
        axs[1].set_xlabel("TTI"); axs[0].legend(fontsize=7, ncol=4)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_exp1_events.pdf", dpi=600, bbox_inches='tight'); plt.close(fig)

    # ---------- table ---------
    f1m, f1h = mean_ci(f1_list)
    with open(f"{OUT}/table_exp1.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["EXP1 -- structure recovery + detection specificity"])
        w.writerow(["causal structure F1 (pre-jamming)", f"{f1m:.3f} +/- {f1h:.3f}"])
        w.writerow(["calibration stationary FAR (alpha)", alpha])
        w.writerow([])
        w.writerow(["method", "benign_false_alarm_rate", "jamming_detection_rate",
                    "median_jamming_delay_TTI"])
        for m in METHODS:
            w.writerow([m, round(np.mean(benign_fa[m]), 3),
                        round(np.mean(jam_det[m]), 3),
                        round(float(np.median(jam_delay[m])), 1)])
    print("EXP1 done. structure F1 = %.3f +/- %.3f | thresholds calibrated @ FAR=%.0f%%"
          % (f1m, f1h, alpha * 100))
    print("  %-15s benignFA  jamDet  delay" % "method")
    for m in METHODS:
        print("  %-15s  %.2f     %.2f    %.0f" % (m, np.mean(benign_fa[m]),
              np.mean(jam_det[m]), np.median(jam_delay[m])))


if __name__ == "__main__":
    t0 = time.time(); run(); print("wall %.1fs" % (time.time() - t0))