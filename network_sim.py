"""
Closed-loop system-level simulator (Experiment 2: energy study).

Pulls large-scale statistics from the 3GPP UMa channel model, runs AMC + a
finite-buffer Lindley queue with MMPP arrivals, and lets an unobserved,
high-variance jammer corrupt the SINR on a scheduled on/off pattern. A pluggable
controller sets Tx power each TTI; the simulator returns the observed 7-dim KPI
stream, an explicit energy ledger, and ground-truth jammer state for evaluation.
"""
import numpy as np
from config import (SimConfig, VAR_NAMES, N_VARS, IDX, db2lin, dbm2watt)
from channel import ChannelState
from controller import FixedPower

# NR CQI spectral-efficiency grid (bps/Hz) and approx SINR thresholds (dB).
CQI_EFF = np.array([0.0, 0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758,
                    1.4766, 1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234,
                    5.1152, 5.5547, 6.2266, 7.4063])
CQI_SINR = np.array([-30, -6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3,
                     11.7, 14.1, 16.3, 18.7, 21.0, 22.7, 24.2, 26.0])


def amc_efficiency(sinr_dB):
    """Map SINR -> spectral efficiency via the NR CQI table (with margin)."""
    idx = np.searchsorted(CQI_SINR, sinr_dB, side="right") - 1
    idx = int(np.clip(idx, 0, len(CQI_EFF) - 1))
    return CQI_EFF[idx]


def make_jammer_schedule(n_steps, duty_cycle, rng, burst_len=300, settle=250):
    """Boolean array: jammer ON during sustained bursts of `burst_len` TTIs.

    Sustained (seconds-scale) jamming is realistic for a barrage jammer and lets
    the estimator observe the jammed, power-saturated regime. Non-overlapping
    bursts are placed to reach the target overall duty cycle.
    """
    on = np.zeros(n_steps, dtype=bool)
    if duty_cycle <= 0:
        return on
    usable = n_steps - settle
    total_on = int(duty_cycle * usable)
    n_bursts = max(1, int(round(total_on / burst_len)))
    # partition the timeline into n_bursts slots, place one burst per slot
    slots = np.linspace(settle, n_steps - burst_len, n_bursts).astype(int)
    jitter = rng.integers(-40, 40, size=n_bursts)
    for s0, j in zip(slots, jitter):
        s = int(np.clip(s0 + j, settle, n_steps - burst_len))
        on[s:s + burst_len] = True
    return on


class TrafficState:
    def __init__(self, cfg, rng):
        self.cfg = cfg; self.rng = rng
        self.high = False
        self.Q_bits = 0.0

    def arrivals_bits(self, dt_s):
        c = self.cfg
        if self.high and self.rng.random() < c.p_high_to_low:
            self.high = False
        elif (not self.high) and self.rng.random() < c.p_low_to_high:
            self.high = True
        lam_mbps = c.lam_high_mbps if self.high else c.lam_low_mbps
        lam_bits = lam_mbps * 1e6 * dt_s
        return self.rng.poisson(max(lam_bits, 0.0)), lam_mbps


def run(cfg: SimConfig, controller, jammer_on=None, seed=None):
    """Run one closed-loop episode. `controller` exposes reset()/act(obs, ctx)."""
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    ch = ChannelState(cfg.channel, rng)
    tr = TrafficState(cfg.traffic, rng)
    pw = cfg.power
    dt_s = cfg.dt_ms * 1e-3
    bw = cfg.channel.bandwidth_Hz
    n = cfg.n_steps
    if jammer_on is None:
        jammer_on = make_jammer_schedule(n, cfg.jammer.duty_cycle, rng)

    X = np.zeros((n, N_VARS))
    energy_J = 0.0
    served_bits_total = 0.0
    ptx = 0.5 * (pw.p_min_dBm + pw.p_max_dBm)
    flags = np.zeros(n, dtype=bool)
    controller.reset()
    prev_gamma = cfg.power.gamma_target_dB

    for t in range(n):
        pl, fading = ch.step()
        # unobserved high-variance jammer
        if jammer_on[t]:
            jam_dBm = cfg.jammer.power_dBm + rng.normal(0, cfg.jammer.sigma_dB)
        else:
            jam_dBm = None
        # controller chooses Tx power from observed history (not from jam state)
        ctx = {"t": t, "gamma_target": pw.gamma_target_dB, "prev_gamma": prev_gamma}
        ptx, flag = controller.act(X[:t], ctx)
        ptx = float(np.clip(ptx, pw.p_min_dBm, pw.p_max_dBm))
        flags[t] = flag

        gamma = ch.sinr_dB(ptx, pl, fading, jammer_dBm=jam_dBm)
        reff = amc_efficiency(gamma)
        arr_bits, lam_mbps = tr.arrivals_bits(dt_s)
        servable = reff * bw * dt_s
        backlog = tr.Q_bits + arr_bits
        served = min(servable, backlog)
        tr.Q_bits = float(np.clip(backlog - served, 0.0, cfg.traffic.Q_max_bits))
        thr_mbps = served / dt_s / 1e6

        # explicit energy ledger: PA + circuit
        p_tx_W = dbm2watt(ptx)
        energy_J += (p_tx_W / pw.pa_efficiency + pw.p_circuit_W) * dt_s
        served_bits_total += served

        X[t] = [gamma, pl, reff, thr_mbps, lam_mbps, tr.Q_bits, ptx]
        prev_gamma = gamma

    cee = served_bits_total / max(energy_J, 1e-9)  # bits per Joule
    return {
        "X": X, "energy_J": energy_J, "served_bits": served_bits_total,
        "cee_bpj": cee, "jammer_on": jammer_on, "flags": flags,
        "mean_ptx_dBm": float(X[:, IDX["Ptx"]].mean()),
        "mean_thr_mbps": float(X[:, IDX["T"]].mean()),
    }


if __name__ == "__main__":
    cfg = SimConfig(n_steps=1500)
    cfg.jammer.duty_cycle = 0.3
    res = run(cfg, FixedPower(40.0), seed=0)
    print("energy(J):", round(res["energy_J"], 1),
          "CEE(bpj):", round(res["cee_bpj"], 1),
          "mean Tx(dBm):", round(res["mean_ptx_dBm"], 1),
          "mean thr(Mbps):", round(res["mean_thr_mbps"], 2),
          "jam steps:", int(res["jammer_on"].sum()))