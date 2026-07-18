"""
Power-control policies for the closed-loop energy study.

  FixedPower            -- constant Tx power (reference).
  CorrelationControl    -- standard inverse-error proportional control (pi_base).
                           Under jamming it keeps pushing toward P_max -> energy waste.
  CausalGatedControl    -- pi_base gated by the online causal estimator: when the
                           Ptx->gamma link breaks it backs off to P_safe.
  OracleControl         -- knows the true jammer schedule (upper bound on savings).

All expose reset() and act(history, ctx) -> (ptx_dBm, flag).
`history` is the (t x N) array of past observations; `ctx` carries gamma_target etc.
"""
import numpy as np
from config import IDX, EstimatorConfig, PowerConfig
from estimator import HierarchicalCausalDiscovery


class FixedPower:
    def __init__(self, ptx_dBm, power: PowerConfig = None):
        self.ptx = ptx_dBm

    def reset(self):
        pass

    def act(self, history, ctx):
        return self.ptx, False


class CorrelationControl:
    """Proportional power control toward an SINR target (the correlation baseline)."""

    def __init__(self, power: PowerConfig = None):
        self.pw = power or PowerConfig()
        self.reset()

    def reset(self):
        self.ptx = 0.5 * (self.pw.p_min_dBm + self.pw.p_max_dBm)
        self.rng = np.random.default_rng(12345)

    def _dither(self):
        return self.rng.uniform(-self.pw.dither_dB, self.pw.dither_dB)

    def _step_power(self, ctx):
        err = ctx["gamma_target"] - ctx["prev_gamma"]
        self.ptx = float(np.clip(self.ptx + self.pw.ctrl_gain * err,
                                 self.pw.p_min_dBm, self.pw.p_max_dBm))
        return float(np.clip(self.ptx + self._dither(),
                             self.pw.p_min_dBm, self.pw.p_max_dBm))

    def act(self, history, ctx):
        return self._step_power(ctx), False


class CausalGatedControl(CorrelationControl):
    """pi_base normally; yields to P_safe when the Ptx->gamma link breaks.

    Decoupled gate logic (avoids chatter from the gating action itself):
      * ENTER gate when the causal control-edge weight collapses (est.link_broken).
      * EXIT gate on an action-independent signal -- gamma's variance returning to
        its healthy baseline (the jammer's high-variance noise has subsided) --
        after a minimum hold time.
    """

    def __init__(self, power: PowerConfig = None, est_cfg: EstimatorConfig = None,
                 min_hold=30, probe_period=50, probe_margin_dB=4.0):
        super().__init__(power)
        self.est = HierarchicalCausalDiscovery(est_cfg)
        self.W = self.est.cfg.window_W
        self.min_hold, self.probe_period, self.probe_margin = (
            min_hold, probe_period, probe_margin_dB)

    def reset(self):
        super().reset()
        if hasattr(self, "est"):
            self.est.reset()
        self._gated = False; self._hold = 0; self._refr = 0; self._probing = False

    def act(self, history, ctx):
        base = self._step_power(ctx)
        t = history.shape[0]
        if t < self.W + 1:
            return base, False
        win = history[-(self.W + 1):]
        if not self._gated:
            broken = self.est.gamma_weight(win)["link_broken"]
            if self._refr > 0:               # refractory: let window refill with live data
                self._refr -= 1
            elif broken:
                self._gated = True; self._hold = 0; self._probing = False
            return (base, False) if not self._gated else \
                   (float(self.pw.p_safe_dBm + self._dither()), True)
        # --- gated: yield power, but probe periodically to test if power helps again ---
        self._hold += 1
        if self._probing:
            self._probing = False
            # did the full-power probe restore SINR? (causal test: power works again)
            if history[-1, IDX["gamma"]] > ctx["gamma_target"] - self.probe_margin:
                self._gated = False; self._refr = self.W
                return base, False
        elif self._hold > self.min_hold and (self._hold % self.probe_period == 0):
            self._probing = True
            return base, True                # 1-step probe at full power
        return float(self.pw.p_safe_dBm + self._dither()), True


class OracleControl(CorrelationControl):
    """Knows the ground-truth jammer schedule; lower-bounds achievable energy."""

    def __init__(self, jammer_on, power: PowerConfig = None):
        super().__init__(power)
        self.jammer_on = jammer_on

    def act(self, history, ctx):
        base = self._step_power(ctx)
        if self.jammer_on[ctx["t"]]:
            return float(self.pw.p_safe_dBm + self._dither()), True
        return base, False