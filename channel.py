"""
3GPP TR 38.901 Urban-Macro (UMa) channel model (tractable subset).

Implements the UMa path-loss equations, log-normal shadowing, and Rayleigh
small-scale fading, plus SINR computation given Tx power and (possibly) a jammer.
This is intentionally a *system-level* model: it gives 3GPP-compliant large-scale
statistics without the full spatial-consistency / cluster machinery of the
ray-level model.
"""
import numpy as np
from config import db2lin, lin2db, dbm2watt, SimConfig


def uma_pathloss_dB(d2d_m, cfg):
    """UMa NLOS path loss (TR 38.901 Table 7.4.1-1), with LOS floor.

    d2d_m : 2D distance BS<->UE in metres.
    Returns large-scale path loss in dB (no shadowing).
    """
    fc = cfg.fc_GHz
    h_bs, h_ut = cfg.h_BS, cfg.h_UT
    d3d = np.sqrt(d2d_m ** 2 + (h_bs - h_ut) ** 2)
    # LOS path loss (PL1 branch), used as a floor.
    pl_los = 28.0 + 22.0 * np.log10(np.maximum(d3d, 1.0)) + 20.0 * np.log10(fc)
    # NLOS path loss.
    pl_nlos = (13.54 + 39.08 * np.log10(np.maximum(d3d, 1.0))
               + 20.0 * np.log10(fc) - 0.6 * (h_ut - 1.5))
    return np.maximum(pl_los, pl_nlos)


def noise_power_dBm(cfg):
    """Thermal noise over the system bandwidth + receiver noise figure."""
    return (cfg.thermal_noise_dBm_Hz + 10.0 * np.log10(cfg.bandwidth_Hz)
            + cfg.noise_figure_dB)


class ChannelState:
    """Slowly-varying UE distance (mobility) -> path loss + shadowing + fading."""

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.d2d = cfg.d0_m
        self.shadow_dB = rng.normal(0.0, cfg.shadow_sigma_dB)
        self.noise_dBm = noise_power_dBm(cfg)

    def step(self):
        """Advance mobility one TTI; return (pathloss_dB incl. shadowing, fading_lin)."""
        c = self.cfg
        # Correlated random walk in distance (UE mobility).
        self.d2d = float(np.clip(self.d2d + self.rng.normal(0, c.mobility_sigma_m),
                                 c.d_min_m, c.d_max_m))
        # AR(1) shadowing for temporal correlation.
        rho = 0.95
        self.shadow_dB = (rho * self.shadow_dB
                          + np.sqrt(1 - rho ** 2) * self.rng.normal(0, c.shadow_sigma_dB))
        pl = uma_pathloss_dB(self.d2d, c) + self.shadow_dB - c.antenna_gain_dB
        # Rayleigh fast fading power gain (exp(1)), lightly smoothed.
        fading_lin = self.rng.exponential(1.0)
        return pl, fading_lin

    def sinr_dB(self, ptx_dBm, pathloss_dB, fading_lin, jammer_dBm=None):
        """Linear-domain SINR -> dB. Jammer (if present) adds to the denominator."""
        rx_dBm = ptx_dBm - pathloss_dB + lin2db(fading_lin)
        rx_lin = dbm2watt(rx_dBm)
        noise_lin = dbm2watt(self.noise_dBm)
        denom = noise_lin
        if jammer_dBm is not None:
            denom = denom + dbm2watt(jammer_dBm)
        return float(lin2db(rx_lin / denom))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cfg = SimConfig()
    ch = ChannelState(cfg.channel, rng)
    pl, fad = ch.step()
    print("pathloss dB:", round(pl, 2), "noise dBm:", round(ch.noise_dBm, 2))
    print("SINR no-jam dB:", round(ch.sinr_dB(40, pl, fad), 2))
    print("SINR jam dB:   ", round(ch.sinr_dB(40, pl, fad, jammer_dBm=30), 2))