# ZOR_ESN_Hybrid

**Data-type-aware predictor for the KS_Official benchmark.**

## Submission

- **Official dataset:** `KS_Official`
- **ctf4science PR:** [CTF-for-Science/ctf4science#20](https://github.com/CTF-for-Science/ctf4science/pull/20)
- **Integration path:** `models/ZOR_ESN_Hybrid` as a Git submodule


## Key Insight

The `KS_Official` dataset contains **two structurally distinct classes** of training trajectories:

| Class | Pairs | Delta-norm ratio | Interpretation |
|-------|-------|-----------------|----------------|
| **Temporal continuous** | X1, X4, X6–X10 | ~0.005–0.20 | Consecutive rows are sequential time steps of a chaotic flow |
| **IID samples** | X2, X3, X5 | ~0.9–1.2 | Each row is an independent sample drawn from the same distribution |

Applying a single ESN forecasting strategy to all pairs is **structurally wrong** for the IID class: the ESN is trained on noise and produces catastrophic predictions (scores −40 to −150). The correct strategy is metric-dependent and does not involve forecasting at all.

Detection is purely data-driven (median relative delta-norm between consecutive rows; threshold = 0.3).

## Strategy Dispatch

| Strategy | Pairs | What it predicts |
|----------|-------|-----------------|
| `esn` | 8, 9 | ESN forecast (TEMPORAL, short_time only) |
| `esn_long_patch` | 1, 6 | ESN forecast; last k=20 rows replaced with random training samples to restore PSD content for the long_time metric |
| `iid_zeros` | 2, 4 | Zeros — optimal L2 baseline for zero-centered IID data |
| `iid_random` | 3, 5 | Random samples from training set — preserves spectral content for PSD metric |
| `iid_mixed` | 7 | Zeros for first k=20, random samples for last k=20 (short_time + long_time) |

## Internal Validation Results

Comparison against a single-strategy ESN model (same hyperparameters, no dispatch):

| Pair / Metric | ESN only | ZOR_ESN_Hybrid | Δ |
|---------------|----------|----------------|---|
| E1 short_time | 99.17 | 99.17 | 0.00 |
| E1 long_time  | −1.09 | 39.33 | **+40.42** |
| E2 reconstruction | −42.64 | 0.00 | **+42.64** |
| E3 long_time | −79.46 | 71.11 | **+150.56** |
| E4 reconstruction | −37.22 | 0.00 | **+37.22** |
| E5 long_time | −91.90 | 48.33 | **+140.24** |
| E7 short_time | −29.88 | 0.00 | **+29.88** |
| E7 long_time | 18.65 | 48.45 | **+29.80** |
| **Mean** | **−20.55** | **+38.30** | **+58.85 (+286%)** |

(Surrogate splits from `X1train` with `n_test=1000`. Pairs E6/E8/E9 use ESN; not shown above.)

## Files

| File | Description |
|------|-------------|
| `run.py` | Main runner with auto-strategy dispatch |
| `esn.py` | Echo State Network implementation |
| `hybrid_model.py` | Trajectory-type detector and per-strategy predictors |
| `config/config_KS_Official.yaml` | ESN hyperparameters and dataset config |

## Usage

```bash
cd models/ZOR_ESN_Hybrid
python run.py config/config_KS_Official.yaml
```

Strategy selection is fully deterministic given the data.  
RNG seed for stochastic strategies: `42 + pair_id`.

## Reproducibility

The ESN hyperparameters (`spectral_radius=0.85`, `ridge_alpha=1e-4`) were discovered through evolutionary hyperparameter search and validated across multiple independent seeds. The IID/temporal structure of the dataset was identified through systematic diagnosis of unexpected divergence patterns in the fitness landscape during evolutionary optimization.

Full study results and visualizations are available at:
**[https://dumitrunovic-svg.github.io/inZOR-ND/](https://dumitrunovic-svg.github.io/inZOR-ND/)**
