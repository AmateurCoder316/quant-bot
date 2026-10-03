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

## EXP-1-M1 hypothesis

`exp-1-m1` is a **dual Extra Trees consensus model** using the existing execution-aligned 60-minute net-return target.

It has two independent prediction heads:

1. **ExtraTreesClassifier** estimates the probability that the exact simulated 60-minute trade is profitable after costs.
2. **ExtraTreesRegressor** estimates the magnitude of the exact simulated 60-minute net return.

The regression forest also provides an ensemble-disagreement estimate. This is used as an uncertainty filter.

A trade is eligible only when all three conditions pass:

- minimum predicted probability of positive net return,
- minimum predicted net-return magnitude,
- maximum allowed model disagreement.

When multiple signals compete for portfolio slots, they are ranked by a confidence-adjusted expected-edge score.

## Chronology for M1

- Train architecture: 2023-2024
- Tune execution gates: 2025 only
- Freeze configuration
- Historical shadow backtest: 2026

2026 is already known from earlier model research, so this is not a pristine test of the entire research process. It remains useful for apples-to-apples comparison with LR/GB models. Future live paper data is required for genuinely new confirmation.

## Files

- `exp1_core.py` - shared EXP-1-M1 architecture, prediction, uncertainty, execution and selection logic
- `train_exp_1_m1.py` - train classifier + regressor
- `tune_exp_1_m1.py` - tune consensus execution gates on 2025 only
- `backtest_exp_1_m1.py` - frozen 2026 historical shadow backtest

## M1 workflow

```bash
python train_exp_1_m1.py
python tune_exp_1_m1.py
python backtest_exp_1_m1.py
```

Do not use 2026 results to modify M1. If the architecture or thresholds need a materially different hypothesis after seeing 2026, create `exp-1-m2`.
