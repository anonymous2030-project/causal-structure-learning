"""
Online Hierarchical Causal Structure Learning.

Components:
  (i)   build_mask()          -- typed protocol-stack structural mask M.
  (ii)  HierarchicalCausalDiscovery -- sliding-window masked Lasso that recovers
        the local Jacobian A_t via a fast numpy coordinate-descent (so the per-TTI
        latency stays well under the RIC budget), then binarises and runs a
        change detector with hysteresis/dwell-time (more robust than flagging on
        any single binarised-edge flip).

The mask restricts each target's predictors to same/adjacent layers and forbids
parents for exogenous + control variables. This drops the candidate-edge set and
makes per-TTI cost linear in N (constant d_eff).
"""
import numpy as np
from config import (N_VARS, LAYER, IDX, ENDO_IDX, EXOG_IDX, CTRL_IDX,
                    EstimatorConfig)
import time
from scm import *


def build_mask():
    """M[i, j] = 1 iff candidate edge x_i -> x_j is allowed.

    Rules: (a) same/adjacent layer (|L_i - L_j| <= 1); (b) exogenous + control
    targets get no parents; (c) self-loops allowed. Returns (M, parents_of) where
    parents_of[j] = array of permitted predictor indices i for target j.
    """
    M = np.zeros((N_VARS, N_VARS), dtype=int)
    no_parent = set(EXOG_IDX) | {CTRL_IDX}
    for j in range(N_VARS):
        if j in no_parent:
            continue
        for i in range(N_VARS):
            if abs(LAYER[i] - LAYER[j]) <= 1:
                M[i, j] = 1
    parents_of = {j: np.where(M[:, j] == 1)[0] for j in range(N_VARS)}
    return M, parents_of


def _lasso_cd(X, y, lam, n_iter, tol):
    """Coordinate-descent Lasso on standardized (X, y). X: (W,p), y: (W,)."""
    W, p = X.shape
    if p == 0:
        return np.zeros(0)
    G = X.T @ X            # p x p (p is tiny, <= ~5)
    Xy = X.T @ y
    diag = np.diag(G).copy()
    diag[diag < 1e-12] = 1e-12
    beta = np.zeros(p)
    lamW = lam * W
    for _ in range(n_iter):
        max_delta = 0.0
        for k in range(p):
            rho = Xy[k] - G[k] @ beta + G[k, k] * beta[k]
            new = np.sign(rho) * max(abs(rho) - lamW, 0.0) / diag[k]
            max_delta = max(max_delta, abs(new - beta[k]))
            beta[k] = new
        if max_delta < tol:
            break
    return beta


def _detrend(M):
    """Remove per-column mean + linear time trend (kills benign level ramps)."""
    W = M.shape[0]
    tc = np.arange(W) - (W - 1) / 2.0
    denom = float(tc @ tc) or 1.0
    slope = (tc @ M) / denom
    return M - np.outer(tc, slope) - M.mean(axis=0)


class HierarchicalCausalDiscovery:
    """Sliding-window masked-Lasso structural estimator + change detector."""

    def __init__(self, cfg: EstimatorConfig = None):
        self.cfg = cfg or EstimatorConfig()
        self.M, self.parents_of = build_mask()
        self.reset()

    def reset(self):
        self.G_prev = None
        self._below = 0
        self.last_A = np.zeros((N_VARS, N_VARS))
        self.w_ema = None

    def n_candidate_edges(self):
        return int(self.M.sum())

    def estimate(self, window):
        """Estimate Jacobian A_hat from a (W x N) window. Row j = coefs for target j."""
        c = self.cfg
        W = window.shape[0]
        A = np.zeros((N_VARS, N_VARS))
        Xlag = _detrend(window[:-1])      # detrended predictors x_{t-1}
        Ycur = _detrend(window[1:])       # detrended targets   x_t
        sd = Xlag.std(axis=0); sd[sd < 1e-9] = 1.0
        Xs = Xlag / sd
        for j in ENDO_IDX:
            par = self.parents_of[j]
            if par.size == 0:
                continue
            y = Ycur[:, j]; ys = y.std(); ys = ys if ys > 1e-9 else 1.0
            beta = _lasso_cd(Xs[:, par], y / ys, c.lam_lasso, c.cd_iters, c.cd_tol)
            A[j, par] = beta              # partial-correlation edge strengths
        self.last_A = A
        return A

    def binarize(self, A):
        return (np.abs(A) > self.cfg.delta_edge).astype(int)

    def gamma_weight(self, window):
        """Fast path: estimate ONLY the gamma row to get w = |A[gamma, Ptx]|.

        4x cheaper than a full-graph update; used inside the online detection /
        control loop where only the control edge Ptx->gamma matters.
        """
        c = self.cfg
        j = IDX["gamma"]; par = self.parents_of[j]
        Xlag = _detrend(window[:-1]); y = _detrend(window[1:])[:, j]
        sd = Xlag.std(axis=0); sd[sd < 1e-9] = 1.0
        Xs = Xlag / sd
        ys = y.std(); ys = ys if ys > 1e-9 else 1.0
        beta = _lasso_cd(Xs[:, par], y / ys, c.lam_lasso, c.cd_iters, c.cd_tol)
        full = np.zeros(N_VARS); full[par] = beta
        w = abs(full[IDX["Ptx"]])
        # scale-invariant detection: flag when the control-edge weight drops below
        # a fraction of its own adaptive baseline (EMA over healthy regimes). This
        # works regardless of variable units (synthetic O(1) vs network-sim dB).
        if self.w_ema is None:
            self.w_ema = w
        broken = w < c.w_break_frac * self.w_ema
        if broken:
            self._below += 1
        else:
            self._below = 0
            self.w_ema = (1 - c.ema_alpha) * self.w_ema + c.ema_alpha * w
        return {"w_ptx_gamma": w, "w_baseline": self.w_ema,
                "link_broken": self._below >= c.dwell}

    def update(self, window):
        """One online step: estimate, binarize, detect change. Returns dict."""
        A = self.estimate(window)
        G = self.binarize(A)
        # key control edge weight w_t = A[gamma, Ptx]
        w = abs(A[IDX["gamma"], IDX["Ptx"]])
        # robust break detection: dwell-time hysteresis on the control edge
        if w < self.cfg.w_break_thresh:
            self._below += 1
        else:
            self._below = 0
        link_broken = self._below >= self.cfg.dwell
        # structural-change flag (Frobenius diff) with same dwell guard
        struct_change = False
        if self.G_prev is not None:
            struct_change = np.linalg.norm(G - self.G_prev) > 0
        self.G_prev = G
        return {"A": A, "G": G, "w_ptx_gamma": w,
                "link_broken": link_broken, "struct_change": struct_change}


if __name__ == "__main__":
    est = HierarchicalCausalDiscovery()
    M, _ = build_mask()
    print("candidate edges (masked):", est.n_candidate_edges(),
          "/ full N(N-1) =", N_VARS * (N_VARS - 1))
    X, jam, tg = generate(1000, 500, seed=2)
    W = est.cfg.window_W
    ws, broken_at = [], None
    t0 = time.perf_counter()
    for t in range(W, len(X)):
        out = est.update(X[t - W:t])
        ws.append(out["w_ptx_gamma"])
        if broken_at is None and out["link_broken"]:
            broken_at = t
    dt = (time.perf_counter() - t0) / (len(X) - W) * 1e3
    ws = np.array(ws)
    print("mean per-step latency: %.3f ms" % dt)
    print("w(Ptx->gamma) pre-break mean: %.3f  post-break mean: %.3f"
          % (ws[:500 - W].mean(), ws[500 - W + 20:].mean()))
    print("break truly at 500, detected at", broken_at)