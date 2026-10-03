# EXP-1 model family

## Naming

Experiment family: `exp-1`

Models in this family are named sequentially:

- `exp-1-m1`
- `exp-1-m2`
- `exp-1-m3`
- ...

A model number represents a frozen hypothesis. Once its 2026 result has been observed, do not silently change that model and rerun it under the same name. A materially changed hypothesis becomes the next model number.

## Why EXP-1 exists

The LR/GB sequence repeatedly found some gross directional edge, but simulated commission + slippage often erased it. EXP-1 therefore changes both the model family and the decision architecture rather than merely retuning gradient boosting.

---

## EXP-1-M1

`exp-1-m1` used a dual Extra Trees consensus architecture:

1. **ExtraTreesClassifier** estimated probability that the exact simulated 60-minute trade was profitable after costs.
2. **ExtraTreesRegressor** estimated the magnitude of the exact simulated 60-minute net return.
3. Regression-tree disagreement was used as an uncertainty filter.

### M1 chronology

- Train architecture: 2023-2024
- Tune execution gates: 2025 only
- Freeze configuration
- Historical shadow backtest: 2026

### M1 result

The 2025 tuner selected a configuration that passed its deployment gate:

- 2025 after-cost return: +3.35%
- profit factor: 1.63
- average trade: +0.410%
- completed trades: 41

Frozen 2026 test:

- after-cost return: -0.91%
- gross return: -0.14%
- after-cost profit factor: 0.33
- gross profit factor: 0.84
- completed trades: 13

M1 is therefore rejected.

### Three lessons carried into M2

1. **Exact-return regression did not generalize.** The M1 regression head had effectively zero ranking correlation in 2025 and remained weak in 2026. M2 removes direct return regression entirely.
2. **The uncertainty filter was not useful.** The M1 tuner selected the 100th-percentile uncertainty cap, effectively disabling the filter. M2 removes that head rather than pretending it adds value.
3. **The 2025 threshold result was too easy to win by selection luck.** M1 searched a 3-D execution grid and passed with only 41 trades. M2 reduces execution selection to one parameter, increases minimum trade support to 75, and requires the entire local threshold neighborhood to remain profitable.

M1 files remain frozen:

- `exp1_core.py`
- `train_exp_1_m1.py`
- `tune_exp_1_m1.py`
- `backtest_exp_1_m1.py`

---

## EXP-1-M2 hypothesis

`exp-1-m2` is a **three-head multi-horizon Extra Trees classifier consensus model**.

It deliberately predicts easier categorical questions instead of an exact noisy return:

1. `p30_positive`: probability the exact 30-minute simulated trade has positive net return after costs.
2. `p60_positive`: probability the exact 60-minute simulated trade has positive net return after costs.
3. `p60_large`: probability the exact 60-minute simulated trade has net return of at least +0.30% after costs.

The final trade still holds for 60 minutes. A signal is eligible only when **all three heads simultaneously enter the same high tail of their own recent probability distributions**.

### Causal rolling consensus

For every symbol and every head, M2 calculates a rolling probability threshold from the previous 1,000 predictions only. The current row is excluded from its own threshold.

Five predefined consensus percentiles are tested on 2025:

- 90.0%
- 92.5%
- 95.0%
- 97.5%
- 99.0%

For example, at the 95% setting, all three heads must each be above their own trailing 95th-percentile probability threshold.

This has several advantages:

- absolute probability calibration may drift without automatically destroying execution,
- all three economic questions must agree,
- only one execution parameter is tuned,
- the tuner cannot mix-and-match independent thresholds for every head.

### M2 anti-selection-luck gate

A selected 2025 configuration is considered deployable only if it has:

- at least 75 completed trades,
- positive after-cost return,
- after-cost profit factor >= 1.10,
- positive average trade,
- positive daily Sharpe,
- gross profit factor >= 1.20,
- a full three-point local percentile neighborhood,
- positive median local return,
- **positive worst local-neighbor return**, and
- local median profit factor >= 1.05.

The selected threshold is frozen before the 2026 historical test.

### M2 chronology

- Train all three heads: 2023-2024
- Tune the single consensus percentile: 2025 only
- Freeze configuration
- Historical shadow backtest: 2026

For the 2026 backtest, trailing probability thresholds are seeded with the final 1,000 2025 predictions per symbol. No 2026 future information is used to form a threshold.

### M2 files

- `exp1_m2_core.py` - architecture, causal consensus, execution and robust selection logic
- `train_exp_1_m2.py` - train three classifier heads
- `tune_exp_1_m2.py` - select one consensus percentile on 2025 only
- `backtest_exp_1_m2.py` - frozen 2026 historical shadow backtest

### M2 workflow

```bash
python train_exp_1_m2.py
python tune_exp_1_m2.py
python backtest_exp_1_m2.py
```

As with M1, do not modify M2 after seeing its frozen 2026 result. Any materially changed hypothesis becomes `exp-1-m3`.

---

## Test-status limitation

2026 has already been observed in earlier model research. These experiments are therefore retrospective architecture comparisons, not pristine final tests. Truly new evidence must come from future/paper data.
