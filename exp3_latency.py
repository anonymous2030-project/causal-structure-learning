"""
Experiment 3 -- Real-time latency & complexity scaling.

(a) Per-TTI inference latency of the actual 7-variable estimator: p50/p95/p99,
    demonstrating sub-10 ms operation within the RIC budget.
(b) Latency vs number of KPIs N for the masked estimator (linear in N, since the
    per-target candidate set d_eff is constant) versus an unmasked Lasso
    (quadratic in N) -- empirically backing the complexity claim.
"""
import os, csv, time, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import SimConfig, EstimatorConfig
from network_sim import run, make_jammer_schedule
from controller import CorrelationControl
from estimator import HierarchicalCausalDiscovery, _lasso_cd, _detrend
from metrics import percentiles

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


def real_latency(W=120, n=2500):
    """Time the true 7-variable online estimator on realistic network-sim data."""
    cfg = SimConfig(n_steps=n); cfg.estimator = EstimatorConfig(window_W=W, cd_iters=45)
    rng = np.random.default_rng(0); jam = make_jammer_schedule(n, 0.3, rng)
    X = run(cfg, CorrelationControl(cfg.power), jammer_on=jam, seed=0)["X"]
    est = HierarchicalCausalDiscovery(cfg.estimator)
    lat = []
    for t in range(W + 1, n):
        t0 = time.perf_counter(); est.update(X[t - W - 1:t]); lat.append((time.perf_counter() - t0) * 1e3)
    return np.array(lat)


def masked_sweep(Ns, W=120, d_eff=5, reps=120):
    """Latency vs N for masked (constant d_eff) vs unmasked (all-N) per-TTI sweep."""
    rng = np.random.default_rng(1)
    res = {"masked": {}, "full": {}}
    for N in Ns:
        n_targets = max(1, round(N * 4 / 7))   # endogenous fraction as in the 7-var model
        masked_t, full_t = [], []
        for _ in range(reps):
            win = rng.standard_normal((W + 1, N))
            Xl = _detrend(win[:-1]); sd = Xl.std(0); sd[sd < 1e-9] = 1; Xs = Xl / sd
            Yc = _detrend(win[1:])
            # masked: each target regresses on d_eff predictors
            t0 = time.perf_counter()
            for j in range(n_targets):
                par = rng.choice(N, size=min(d_eff, N), replace=False)
                y = Yc[:, j]; ys = y.std() or 1.0
                _lasso_cd(Xs[:, par], y / ys, 0.05, 45, 1e-4)
            masked_t.append((time.perf_counter() - t0) * 1e3)
            # full (unmasked): each target regresses on all N predictors
            t0 = time.perf_counter()
            for j in range(n_targets):
                y = Yc[:, j]; ys = y.std() or 1.0
                _lasso_cd(Xs, y / ys, 0.05, 45, 1e-4)
            full_t.append((time.perf_counter() - t0) * 1e3)
        res["masked"][N] = percentiles(masked_t); res["full"][N] = percentiles(full_t)
    return res


def main():
    lat = real_latency()
    p = percentiles(lat)
    with plt.rc_context(IEEE_STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.hist(lat, bins=50, color="C0", alpha=0.8)
        for q, c in [("p50", "k"), ("p95", "C1"), ("p99", "C3")]:
            ax.axvline(p[q], color=c, ls="--", lw=1.2, label=f"{q}={p[q]:.2f} ms")
        ax.axvline(10, color="green", ls="-", lw=1.5, label="10 ms RIC budget")
        ax.set_xlabel("per-TTI inference latency [ms]"); ax.set_ylabel("count")
        ax.set_title("Online Estimator Latency (7-var model)"); ax.legend(loc="best", framealpha=0.8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_exp3_latency_hist.pdf", dpi=600, bbox_inches='tight'); plt.close(fig)

        Ns = [7, 14, 21, 28, 42, 56, 70]
        sw = masked_sweep(Ns)
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        mm = [sw["masked"][N]["p50"] for N in Ns]; m95 = [sw["masked"][N]["p95"] for N in Ns]
        ff = [sw["full"][N]["p50"] for N in Ns]; f95 = [sw["full"][N]["p95"] for N in Ns]
        ax.plot(Ns, mm, "-o", color="C0", ms=4, lw=1.5, label="masked p50 (ours, ~linear)")
        ax.plot(Ns, m95, "--o", color="C0", ms=3, lw=1, alpha=0.8, label="masked p95")
        ax.plot(Ns, ff, "-s", color="C3", ms=4, lw=1.5, label="unmasked p50 (~quadratic)")
        ax.plot(Ns, f95, "--s", color="C3", ms=3, lw=1, alpha=0.8, label="unmasked p95")
        ax.axhline(10, color="green", ls="-", lw=1.2, label="10 ms budget")
        ax.set_xlabel("number of network KPIs  N"); ax.set_ylabel("per-TTI latency [ms]")
        ax.set_title("Inference Latency Scaling")
        ax.legend(fontsize=7, loc="upper left", ncol=1, framealpha=0.8); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(f"{OUT}/fig_exp3_scaling.pdf", dpi=600, bbox_inches='tight'); plt.close(fig)

    with open(f"{OUT}/table_exp3.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["EXP3 -- latency"])
        w.writerow(["7-var estimator p50_ms", "p95_ms", "p99_ms", "max_ms"])
        w.writerow([f"{p['p50']:.3f}", f"{p['p95']:.3f}", f"{p['p99']:.3f}", f"{p['max']:.3f}"])
        w.writerow([])
        w.writerow(["N", "masked_p50_ms", "masked_p95_ms", "full_p50_ms", "full_p95_ms"])
        for N in Ns:
            w.writerow([N, f"{sw['masked'][N]['p50']:.3f}", f"{sw['masked'][N]['p95']:.3f}",
                        f"{sw['full'][N]['p50']:.3f}", f"{sw['full'][N]['p95']:.3f}"])
    print("EXP3 done. 7-var estimator: p50=%.2f p95=%.2f p99=%.2f ms (budget 10 ms)"
          % (p["p50"], p["p95"], p["p99"]))
    print("  masked N=70 p95=%.2f ms | unmasked N=70 p95=%.2f ms"
          % (sw["masked"][70]["p95"], sw["full"][70]["p95"]))


if __name__ == "__main__":
    t0 = time.time(); main(); print("wall %.1fs" % (time.time() - t0))