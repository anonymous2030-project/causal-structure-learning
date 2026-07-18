"""Evaluation metrics: structure recovery + change detection + energy."""
import numpy as np


def shd(G_true, G_hat):
    """Structural Hamming Distance (count of differing adjacency entries)."""
    return int(np.sum(G_true != G_hat))


def edge_prf(G_true, G_hat, ignore_diag=True):
    """Directed-edge precision / recall / F1 (optionally excluding self-loops)."""
    A, B = G_true.copy(), G_hat.copy()
    if ignore_diag:
        np.fill_diagonal(A, 0); np.fill_diagonal(B, 0)
    tp = int(np.sum((A == 1) & (B == 1)))
    fp = int(np.sum((A == 0) & (B == 1)))
    fn = int(np.sum((A == 1) & (B == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def detection_delay_and_far(flags, break_step, settle=0):
    """Given a boolean flag stream and the true break, return (delay, far).

    delay : steps from break to first post-break flag (np.nan if never).
    far   : fraction of pre-break (post-settle) steps that flagged (false alarms).
    """
    flags = np.asarray(flags, bool)
    pre = flags[settle:break_step]
    far = float(pre.mean()) if pre.size else 0.0
    post = np.where(flags[break_step:])[0]
    delay = int(post[0]) if post.size else np.nan
    return delay, far


def mean_ci(x, z=1.96):
    """Mean and half-width of a 95% CI."""
    x = np.asarray(x, float)
    m = x.mean()
    hw = z * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, hw


def percentiles(x):
    x = np.asarray(x, float)
    return {"p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)),
            "mean": float(x.mean()), "max": float(x.max())}