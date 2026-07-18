"""
Change-detection baselines used to benchmark the causal detector.

  CusumDetector        -- classic two-sided CUSUM on standardized gamma innovations.
  CorrelationDetector  -- sliding-window Pearson corr(Ptx, gamma); flags when the
                          correlation magnitude collapses.
  NeuralResidualDetector -- LSTM (PyTorch) or MLP (sklearn) next-step predictor of
                          gamma trained on a calibration window; CUSUM on residuals.

All expose reset() and update(window)->bool (change flagged this step), matching
the causal estimator's online interface for apples-to-apples ROC comparison.
"""
import numpy as np
from config import IDX, EstimatorConfig

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
from sklearn.neural_network import MLPRegressor


class CusumDetector:
    def __init__(self, k=0.5, h=6.0, warmup=120):
        self.k, self.h, self.warmup = k, h, warmup
        self.reset()

    def reset(self):
        self.mu = None; self.sd = None; self.gp = self.gn = 0.0; self.stat = 0.0

    def update(self, window):
        g = window[:, IDX["gamma"]]
        if self.mu is None:
            self.mu, self.sd = g.mean(), g.std() + 1e-6
        z = (g[-1] - self.mu) / self.sd
        self.gp = max(0.0, self.gp + z - self.k)
        self.gn = max(0.0, self.gn - z - self.k)
        self.stat = max(self.gp, self.gn)
        return self.stat > self.h


class CorrelationDetector:
    def __init__(self, thresh=0.2, dwell=3):
        self.thresh, self.dwell = thresh, dwell
        self.reset()

    def reset(self):
        self._below = 0; self.stat = 0.0

    def update(self, window):
        a = window[:-1, IDX["Ptx"]]; b = window[1:, IDX["gamma"]]  # lag-1
        if a.std() < 1e-9 or b.std() < 1e-9:
            r = 0.0
        else:
            r = abs(np.corrcoef(a, b)[0, 1])
        self.stat = 1.0 - r   # higher => more anomalous (corr collapsed)
        self._below = self._below + 1 if r < self.thresh else 0
        return self._below >= self.dwell


if _HAS_TORCH:
    class _LSTM(nn.Module):
        def __init__(self, n_in, hid=16):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hid, batch_first=True)
            self.fc = nn.Linear(hid, 1)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.fc(o[:, -1, :])


class NeuralResidualDetector:
    """Predict gamma_t from a short lag window; CUSUM on the prediction residual."""

    def __init__(self, lag=8, k=0.5, h=6.0, calib=300, use_lstm=True):
        self.lag, self.k, self.h, self.calib = lag, k, h, calib
        self.use_lstm = use_lstm and _HAS_TORCH
        self.reset()

    def reset(self):
        self.model = None; self.res_mu = self.res_sd = None
        self.gp = self.gn = 0.0; self._buf = []; self.stat = 0.0

    def _features(self, window):
        g = window[:, IDX["gamma"]]; p = window[:, IDX["Ptx"]]
        Xs, ys = [], []
        for t in range(self.lag, len(window)):
            Xs.append(np.concatenate([g[t - self.lag:t], p[t - self.lag:t]]))
            ys.append(g[t])
        return np.array(Xs), np.array(ys)

    def _fit(self, window):
        X, y = self._features(window)
        if self.use_lstm:
            seq = X.reshape(X.shape[0], 2, self.lag).transpose(0, 2, 1)
            xt = torch.tensor(seq, dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            self.model = _LSTM(2)
            opt = torch.optim.Adam(self.model.parameters(), lr=0.02)
            lossf = nn.MSELoss()
            for _ in range(60):
                opt.zero_grad(); loss = lossf(self.model(xt), yt)
                loss.backward(); opt.step()
            pred = self.model(xt).detach().numpy().ravel()
        else:
            self.model = MLPRegressor(hidden_layer_sizes=(16,), max_iter=300,
                                      random_state=0)
            self.model.fit(X, y); pred = self.model.predict(X)
        res = y - pred
        self.res_mu, self.res_sd = res.mean(), res.std() + 1e-6

    def _predict_last(self, window):
        X, _ = self._features(window)
        x = X[-1:]
        if self.use_lstm:
            seq = x.reshape(1, 2, self.lag).transpose(0, 2, 1)
            return float(self.model(torch.tensor(seq, dtype=torch.float32)).item())
        return float(self.model.predict(x)[0])

    def update(self, window):
        if self.model is None:
            if window.shape[0] < self.calib:
                return False
            self._fit(window[:self.calib]); return False
        pred = self._predict_last(window)
        res = (window[-1, IDX["gamma"]] - pred - self.res_mu) / self.res_sd
        self.gp = max(0.0, self.gp + res - self.k)
        self.gn = max(0.0, self.gn - res - self.k)
        self.stat = max(self.gp, self.gn)
        return self.stat > self.h


def make_baselines(window_W):
    bl = {"CUSUM": CusumDetector(warmup=window_W),
          "Correlation": CorrelationDetector()}
    name = "LSTM" if _HAS_TORCH else "MLP"
    bl[name] = NeuralResidualDetector(use_lstm=_HAS_TORCH)
    return bl


if __name__ == "__main__":
    print("torch available:", _HAS_TORCH)