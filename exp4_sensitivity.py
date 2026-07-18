"""
Experiment 4 -- Sensitivity analysis.

Sweeps three knobs on the REAL estimator/simulator:
  (A) window length W        -> structure F1, jamming detection delay, benign FAR
  (B) binarization delta_edge-> structure F1 / SHD / precision / recall
  (C) number of seeds        -> 95% CI half-width of energy savings & throughput

Usage:
  python exp4_sensitivity.py wdelta
  python exp4_sensitivity.py seeds <duty> <s0> <s1>   # append seeds [s0,s1)
  python exp4_sensitivity.py plots
"""
import os, csv, sys, time, numpy as np
from config import SimConfig, EstimatorConfig, IDX
from scm import generate
from estimator import HierarchicalCausalDiscovery
from metrics import edge_prf, shd, mean_ci
from network_sim import run, make_jammer_schedule
from controller import CorrelationControl, CausalGatedControl

OUT = "results"; os.makedirs(OUT, exist_ok=True)
N_STEPS_SCM = 800
BREAK = 520
BENIGN = 350
H = 100                      # detection horizon (TTIs)

# ---------- helpers on the synthetic H-SCM (Exp-1 style) ----------
def causal_stream(X, W, est_cfg):
    """Online gamma-weight detector over a stream. Returns (flags, w)."""
    est = HierarchicalCausalDiscovery(est_cfg)
    flags = np.zeros(len(X), bool)
    for t in range(W + 1, len(X)):
        out = est.gamma_weight(X[t - W - 1:t])
        flags[t] = out["link_broken"]
    return flags

def structure_A(X, W, est_cfg, k_last=40, upto=BREAK):
    """Average continuous Jacobian over the last k_last pre-event windows."""
    est = HierarchicalCausalDiscovery(est_cfg)
    acc = None; cnt = 0
    for t in range(W + 1, upto):
        A = est.estimate(X[t - W - 1:t])
        if t >= upto - k_last:
            acc = A.copy() if acc is None else acc + A
            cnt += 1
    return acc / max(cnt, 1)

# =========================================================
# (A) window-length sweep
# =========================================================
def run_W_sweep(Ws=(45, 60, 90, 120, 150, 200), n_seeds=6):
    from scm import true_graph
    rows = []
    for W in Ws:
        cfg = EstimatorConfig(window_W=W, cd_iters=45, delta_edge=0.10)
        f1s, delays, dets, fars = [], [], [], []
        for e in range(n_seeds):
            # jamming episode (no benign): detect + structure
            Xj, jam, _, tg = generate(N_STEPS_SCM, BREAK, seed=300 + e,
                                      benign_step=N_STEPS_SCM + 5)
            fl = causal_stream(Xj, W, cfg)
            post = np.where(fl[BREAK:BREAK + H])[0]
            if post.size:
                delays.append(int(post[0])); dets.append(1.0)
            else:
                delays.append(H); dets.append(0.0)
            A = structure_A(Xj, W, cfg)
            G = (np.abs(A) > cfg.delta_edge).astype(int)
            f1s.append(edge_prf(tg(False), G)[2])
            # benign episode (surge, no jam): false-alarm rate
            Xb, _, ben, _ = generate(N_STEPS_SCM, N_STEPS_SCM + 5, seed=200 + e,
                                     benign_step=BENIGN)
            flb = causal_stream(Xb, W, cfg)
            fars.append(float(flb[W + 20:].mean()))
        f1m, f1h = mean_ci(f1s); dm, dh = mean_ci(delays); fm, fh = mean_ci(fars)
        rows.append([W, f1m, f1h, float(np.mean(dets)), dm, dh, fm, fh])
        print("W=%3d  F1=%.2f+/-%.2f  det=%.2f  delay=%4.1f+/-%.1f  benignFAR=%.3f+/-%.3f"
              % (W, f1m, f1h, np.mean(dets), dm, dh, fm, fh))
    with open(f"{OUT}/table_sens_W.csv", "w", newline="") as fh_:
        w = csv.writer(fh_)
        w.writerow(["W", "F1", "F1_ci", "detect_rate", "delay_TTI", "delay_ci",
                    "benign_FAR", "benign_FAR_ci"])
        w.writerows([[r[0]] + [round(v, 4) for v in r[1:]] for r in rows])

# =========================================================
# (B) delta_edge sweep  (re-binarize a shared averaged Jacobian)
# =========================================================
def run_delta_sweep(deltas=(0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28),
                    W=120, n_seeds=6):
    from scm import true_graph
    cfg = EstimatorConfig(window_W=W, cd_iters=45, delta_edge=0.10)
    per_seed_A, tgs = [], None
    for e in range(n_seeds):
        Xj, jam, _, tg = generate(N_STEPS_SCM, BREAK, seed=300 + e,
                                  benign_step=N_STEPS_SCM + 5)
        per_seed_A.append(structure_A(Xj, W, cfg)); tgs = tg
    rows = []
    for d in deltas:
        f1s, shds, ps, rs = [], [], [], []
        for A in per_seed_A:
            G = (np.abs(A) > d).astype(int)
            p, r, f1 = edge_prf(tgs(False), G)
            f1s.append(f1); ps.append(p); rs.append(r)
            shds.append(shd(tgs(False), G))
        f1m, f1h = mean_ci(f1s)
        rows.append([d, f1m, f1h, float(np.mean(ps)), float(np.mean(rs)),
                     float(np.mean(shds))])
        print("delta=%.2f  F1=%.2f+/-%.2f  P=%.2f R=%.2f SHD=%.1f"
              % (d, f1m, f1h, np.mean(ps), np.mean(rs), np.mean(shds)))
    with open(f"{OUT}/table_sens_delta.csv", "w", newline="") as fh_:
        w = csv.writer(fh_)
        w.writerow(["delta_edge", "F1", "F1_ci", "precision", "recall", "SHD"])
        w.writerows([[r[0]] + [round(v, 4) for v in r[1:]] for r in rows])

# =========================================================
# (C) seed-convergence sweep (append mode)
# =========================================================
def run_seeds(duty, s0, s1, n_steps=3000, W=90):
    path = f"{OUT}/table_sens_seeds_d{int(duty*100):02d}.csv"
    new = not os.path.exists(path)
    fh_ = open(path, "a", newline=""); w = csv.writer(fh_)
    if new:
        w.writerow(["seed", "duty", "causal_savings_%", "throughput_retention_%"])
    for s in range(s0, s1):
        cfg = SimConfig(n_steps=n_steps, seed=s)
        cfg.estimator = EstimatorConfig(window_W=W, cd_iters=45)
        rng = np.random.default_rng(1000 + s)
        jam = make_jammer_schedule(n_steps, duty, rng)
        rb = run(cfg, CorrelationControl(cfg.power), jammer_on=jam, seed=s)
        rc = run(cfg, CausalGatedControl(cfg.power, cfg.estimator), jammer_on=jam, seed=s)
        sav = 100 * (rb["energy_J"] - rc["energy_J"]) / rb["energy_J"]
        ret = 100 * rc["mean_thr_mbps"] / rb["mean_thr_mbps"]
        w.writerow([s, duty, round(sav, 3), round(ret, 3)]); fh_.flush()
        print("  duty=%.1f seed=%2d  savings=%.1f%%  retention=%.1f%%" % (duty, s, sav, ret))
    fh_.close()

if __name__ == "__main__":
    t0 = time.time(); cmd = sys.argv[1] if len(sys.argv) > 1 else "wdelta"
    if cmd == "wdelta":
        run_W_sweep(); run_delta_sweep()
    elif cmd == "seeds":
        run_seeds(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    print("wall %.1fs" % (time.time() - t0))