"""
Global configuration for the Causal-O-RAN experiments.

All physical constants follow 3GPP TR 38.901 (UMa) conventions where applicable.
Everything is a plain dataclass so experiments can override fields per run and the
exact configuration can be serialized into the results for reproducibility.
"""
from dataclasses import dataclass, field, asdict
import numpy as np

# --- Canonical observed state vector (N = 7, matching Table I of the paper) ---
# Index order is fixed and used everywhere (mask, estimator, plots).
#   0: gamma   (SINR)            endogenous   L1 (PHY)
#   1: Lpath   (path loss)       exogenous    L1 (PHY)
#   2: Reff    (spectral eff.)   endogenous   L2 (MAC)
#   3: T       (throughput)      endogenous   L2 (MAC)
#   4: lam     (arrival rate)    exogenous    L3 (APP)
#   5: Q       (queue length)    endogenous   L3 (APP)
#   6: Ptx     (transmit power)  control      L1 (PHY)
#  I_jam is an UNOBSERVED exogenous driver. It is NOT
# part of the 7-dim observed state: the method detects the jamming regime through the *collapse* of
# the Ptx->gamma coefficient, it does not claim to identify I_jam.
VAR_NAMES = ["gamma", "Lpath", "Reff", "T", "lam", "Q", "Ptx"]
N_VARS = len(VAR_NAMES)
IDX = {name: i for i, name in enumerate(VAR_NAMES)}

# OSI-stack layer of each observed variable (1=PHY, 2=MAC, 3=APP).
LAYER = np.array([1, 1, 2, 2, 3, 3, 1])

# Endogenous targets that get a structural equation / regression.
ENDO = ["gamma", "Reff", "T", "Q"]
ENDO_IDX = [IDX[v] for v in ENDO]
# Exogenous observed variables (cannot have parents).
EXOG = ["Lpath", "lam"]
EXOG_IDX = [IDX[v] for v in EXOG]
# Control input (cannot have parents; not a regression target).
CTRL_IDX = IDX["Ptx"]


@dataclass
class ChannelConfig:
    """3GPP TR 38.901 Urban-Macro (UMa) parameters (subset)."""
    fc_GHz: float = 3.5            # carrier frequency
    h_BS: float = 25.0            # base-station height [m]
    h_UT: float = 1.5            # user height [m]
    bandwidth_Hz: float = 20e6   # system bandwidth
    noise_figure_dB: float = 7.0
    thermal_noise_dBm_Hz: float = -174.0
    shadow_sigma_dB: float = 6.0  # UMa NLOS log-normal shadowing std
    antenna_gain_dB: float = 8.0
    d0_m: float = 35.0           # initial 2D UE distance
    d_min_m: float = 10.0
    d_max_m: float = 500.0
    mobility_sigma_m: float = 2.0  # random-walk step std for UE distance


@dataclass
class TrafficConfig:
    """Markov-modulated Poisson arrivals (two states) + buffer."""
    lam_low_mbps: float = 5.0
    lam_high_mbps: float = 40.0
    p_low_to_high: float = 0.02
    p_high_to_low: float = 0.05
    Q_max_bits: float = 5.0e6     # finite buffer -> saturation breaks T->Q link


@dataclass
class JammerConfig:
    """Abrupt on/off barrage jammer (unobserved exogenous driver).

    Modelled as a *received* interference power that is (a) well above the noise
    floor and (b) highly variable per step. The jammer's variance dominates the SINR, so the smooth
    Ptx variations can no longer explain gamma and the Ptx->gamma Lasso
    coefficient collapses toward zero.
    """
    power_dBm: float = -28.0      # mean received jammer power when ON (dominates max signal)
    sigma_dB: float = 9.0         # per-step std of received jammer power (high variance)
    on_intervals: tuple = ()      # list of (start_step, end_step); set per experiment
    duty_cycle: float = 0.3       # used when intervals are auto-generated


@dataclass
class PowerConfig:
    """Tx power limits and power-amplifier / circuit energy model."""
    p_min_dBm: float = 0.0
    p_max_dBm: float = 46.0       # 3GPP macro BS max
    p_safe_dBm: float = 5.0       # fallback ("yield to jammer") power
    pa_efficiency: float = 0.30   # power-amplifier efficiency
    p_circuit_W: float = 20.0     # static circuit power per TTI
    gamma_target_dB: float = 12.0
    ctrl_gain: float = 0.5        # proportional power-control gain
    dither_dB: float = 1.0        # persistent excitation for identifiability


@dataclass
class EstimatorConfig:
    window_W: int = 120
    lam_lasso: float = 0.05       # selected via BIC in practice; default for speed
    delta_edge: float = 0.15      # binarization threshold on |coef|
    cd_iters: int = 60
    cd_tol: float = 1e-4
    dwell: int = 3                # consecutive steps below threshold to confirm break
    w_break_thresh: float = 0.18  # absolute level 
    w_break_frac: float = 0.50    # flag when w < frac * adaptive baseline (scale-free)
    ema_alpha: float = 0.03       # baseline EMA rate 


@dataclass
class SimConfig:
    dt_ms: float = 10.0           # control interval (near-RT/real-time RIC boundary)
    n_steps: int = 2000
    seed: int = 0
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    jammer: JammerConfig = field(default_factory=JammerConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)

    def to_dict(self):
        return asdict(self)


# --- dB / linear helpers ---
def db2lin(x_db):
    return 10.0 ** (np.asarray(x_db) / 10.0)

def lin2db(x_lin):
    return 10.0 * np.log10(np.maximum(np.asarray(x_lin), 1e-30))

def dbm2watt(p_dbm):
    return 10.0 ** ((np.asarray(p_dbm) - 30.0) / 10.0)