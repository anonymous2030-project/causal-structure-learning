"""
Synthetic Hierarchical Structural Causal Model (H-SCM) generator (Experiment 1).

Dimensionless units (each variable O(1)) so the linear-Gaussian TV-VAR(1)
coefficients are directly meaningful and every true edge is recoverable -- this
is the estimator-correctness benchmark; physical 3GPP units live in network_sim.

Two events are injected:
  * benign_step  : a LEGITIMATE sustained path-loss surge. It drags gamma down
                   (a level/variance shift) but the Ptx->gamma MECHANISM is
                   intact -- power still helps. Correlation/CUSUM/level detectors
                   misfire here; the causal detector should NOT, because the
                   Ptx->gamma coefficient is unchanged.
  * break_step   : JAMMING onset. The Ptx->gamma coefficient is set to 0 and
                   high-variance jammer noise dominates gamma -> the edge
                   collapses and the causal detector fires.
"""
import numpy as np
from config import IDX, N_VARS
from metrics import edge_prf, shd
from config import EstimatorConfig
from estimator import HierarchicalCausalDiscovery


def true_graph(jam_on):
    """Binary ground-truth adjacency G[j, i] = 1 iff edge x_i -> x_j present."""
    G = np.zeros((N_VARS, N_VARS), dtype=int)
    g, lp, re, T, lam, Q, ptx = (IDX["gamma"], IDX["Lpath"], IDX["Reff"],
                                 IDX["T"], IDX["lam"], IDX["Q"], IDX["Ptx"])
    G[g, g] = 1; G[g, lp] = 1
    if not jam_on:
        G[g, ptx] = 1
    G[re, re] = 1; G[re, g] = 1
    G[T, T] = 1; G[T, re] = 1
    G[Q, Q] = 1; G[Q, lam] = 1; G[Q, T] = 1
    return G


def generate(n_steps, break_step, seed=0, benign_step=None, jam_std=1.2,
             benign_offset=2.5):
    """Return X (n_steps x N), event labels, and true_graph().

    benign_step : optional step for a legitimate path-loss surge (default ~halfway
                  between settle and the jamming break).
    """
    rng = np.random.default_rng(seed)
    g, lp, re, T, lam, Q, ptx = (IDX["gamma"], IDX["Lpath"], IDX["Reff"],
                                 IDX["T"], IDX["lam"], IDX["Q"], IDX["Ptx"])
    if benign_step is None:
        benign_step = max(0, break_step - 200)
    X = np.zeros((n_steps, N_VARS))
    jam_on = np.zeros(n_steps, dtype=bool); jam_on[break_step:] = True
    benign = np.zeros(n_steps, dtype=bool); benign[benign_step:] = True

    lp_level = 0.0
    lam_state = 0.0
    for t in range(1, n_steps):
        xp = X[t - 1]
        # exogenous path loss: AR(1) + sustained benign surge after benign_step
        lp_level = 0.6 * lp_level + rng.normal(0, 0.6)  # stationary AR(1)
        Lpath = lp_level
        # exogenous arrivals: mean-reverting two-level
        if rng.random() < 0.02:
            lam_state = rng.choice([-1.0, 1.0])
        lam_val = 0.95 * lam_state + rng.normal(0, 0.5)
        # control input (AR(1), carries variance so Ptx->gamma is identifiable)
        Ptx = rng.normal(0, 1.0)  # white (dithered) excitation for identifiability
        # --- pure VAR(1): every endogenous var depends on PARENTS AT t-1 ---
        # (matches the TV-VAR x_t = A_t x_{t-1}; the L1->L2->L3 chain propagates
        #  one layer per TTI, avoiding instantaneous-causation / estimator mismatch)
        a_ptx = 0.0 if jam_on[t] else 0.6
        noise_g = rng.normal(0, jam_std if jam_on[t] else 0.6)
        # benign interference ramps in gradually (~60 TTIs): a sustained mean
        # shift with NO within-window variance spike, so the standardized Ptx
        # coefficient (partial correlation) is unaffected while marginal detectors
        # that track gamma's level still fire.
        if benign[t] and not jam_on[t]:
            ramp = min(1.0, (t - benign_step) / 200.0)
            b_off = benign_offset * ramp
        else:
            b_off = 0.0
        gamma = 0.45 * xp[g] + a_ptx * xp[ptx] - 0.5 * xp[lp] + b_off + noise_g
        Reff = 0.40 * xp[re] + 0.60 * xp[g] + rng.normal(0, 0.4)
        Tn = 0.45 * xp[T] + 0.60 * xp[re] + rng.normal(0, 0.4)
        Qn = 0.50 * xp[Q] + 0.60 * xp[lam] - 0.50 * xp[T] + rng.normal(0, 0.4)
        X[t] = [gamma, Lpath, Reff, Tn, lam_val, Qn, Ptx]
    return X, jam_on, benign, true_graph


if __name__ == "__main__":
    X, jam, benign, tg = generate(1200, 700, seed=1, benign_step=450)
    print("X shape", X.shape, "| benign@450 jam@700")
    print("gamma std  pre-benign/benign/jam: %.2f / %.2f / %.2f"
          % (X[100:450, 0].std(), X[450:700, 0].std(), X[700:, 0].std()))
    for dlt in [0.06, 0.08, 0.10, 0.12]:
        est = HierarchicalCausalDiscovery(EstimatorConfig(window_W=120, cd_iters=50,
                                                          delta_edge=dlt))
        for t in range(121, 690):
            est.update(X[t - 121:t])
        p, r, f1 = edge_prf(tg(False), est.G_prev)
        print("delta=%.2f  F1=%.2f P=%.2f R=%.2f SHD=%d"
              % (dlt, f1, p, r, shd(tg(False), est.G_prev)))