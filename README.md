# Real-Time Causal Discovery for Energy-Efficient 6G O-RAN — Experiment Code


Everything is pure Python (NumPy / SciPy / scikit-learn / Matplotlib). No GPU,
no external simulator. A custom system-level simulator uses 3GPP TR 38.901 (UMa)
large-scale propagation statistics.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py          # runs all three experiments, writes results/
# or individually:
python exp1_estimator.py   # structure recovery + detection specificity
python exp2_energy.py       # energy savings vs jamming duty cycle
python exp3_latency.py      # per-TTI latency + complexity scaling
python exp4_analysis.py
python exp4_sensitivity.py
python exp6_multiuser_scaling.py
python exp7_violation_shortcut.py
python exp8_probing_queue.py
```

Figures (`.pdf`) and tables (`.csv`) land in `results/`.

## Code map (to paper sections)

| File | Paper section | Role |
|------|---------------|------|
| `config.py` | Table I, §II | Variables, layers, 3GPP/power/estimator parameters (dataclasses) |
| `channel.py` | §II | TR 38.901 UMa path loss, shadowing, fading, SINR |
| `scm.py` | §II-B | Synthetic Hierarchical SCM (known ground-truth DAG) for Exp 1 |
| `network_sim.py` | §II, §IV | Closed-loop system simulator: AMC, Lindley queue, MMPP arrivals, jammer, energy ledger |
| `estimator.py` | §III, Alg. 1 | Structural mask + sliding-window masked Lasso + change detection |
| `controller.py` | §IV | Correlation baseline, causal-gated controller, oracle |
| `baselines.py` | §V | CUSUM, correlation, neural (MLP/LSTM) residual detectors |
| `metrics.py` | §V | SHD, edge-F1, detection delay/FAR, energy, latency percentiles |
| `exp1/2/3_*.py` | §V | The three experiments |

## Method summary

State `x_t ∈ R^7` over three OSI layers (PHY/MAC/APP). Within a sliding window the
nonlinear dynamics are locally linear (manifold hypothesis), giving a TV-VAR(1)
`x_t = A_t x_{t-1} + η`. We estimate `A_t` row-by-row with a **mask-constrained
Lasso** (coordinate descent), where the protocol-stack mask forbids layer-skipping
and exogenous parents — making per-TTI cost linear in N. Each window is **linearly
detrended** (removing benign level shifts) and predictors standardized, so the
recovered coefficients are partial correlations that are invariant to benign mean
changes but collapse under jamming. The control edge `w_t = Â_t[P_tx→γ]` drives a
**causal gate**: when it collapses below an adaptive baseline the controller yields
to `P_safe`, and a periodic **probe** detects jammer cessation to resume.

## Results (default settings)

- **Exp 1 — structure & specificity.** Pre-jamming structure recovery **F1 = 1.00**.
  Calibrated to a common 1% stationary false-alarm rate, the causal detector is the
  only method with both low benign false alarms and high jamming detection; on the
  detection-rate-vs-benign-false-alarm operating curve it sits in the top-left while
  CUSUM (chases benign mean shifts), correlation (no specificity) and the neural
  detector (insensitive to subtle jamming) do not.
- **Exp 2 — energy.** Causal-gated control tracks the oracle and saves
  **~21% / 42% / 50% / 58% / 60%** energy at jammer duty cycles **0.1–0.5**
  (crossing the paper's headline ~45% near 20% duty), while retaining **87–97%**
  throughput. The savings are duty-dependent — report the curve, not a single number.
- **Exp 3 — latency.** Per-TTI inference **p50 ≈ 1.9 ms, p99 ≈ 3.8 ms** for the
  7-variable model — well inside the RIC budget. The mask keeps latency under 10 ms
  even at N = 70 (≈8 ms) where unmasked Lasso needs ≈150 ms (linear vs quadratic).


## Reproducibility

All randomness is seeded (`numpy.random.default_rng`). Each experiment runs multiple
seeds and reports 95% confidence intervals. Edit the dataclasses in `config.py` to
change physical parameters; the full configuration is serializable via
`SimConfig.to_dict()`.
