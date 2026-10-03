# EXP-2 model family

## Purpose

EXP-2 is a clean break from the LR / GB / EXP-1 trees.

The repeated earlier result was not simply "the classifier is bad". Several models showed directional ranking skill, but fixed-horizon trades captured moves too small to survive the configured commission + slippage model. EXP-1-M1 also showed that direct return regression and tree-disagreement uncertainty did not generalize, while EXP-1-M2 showed that increasingly selective short-horizon consensus still produced only a small gross edge.

EXP-2 therefore changes the problem definition, the sampling, the label, the validation method, the model architecture and the execution gate.

## EXP-2-M1 hypothesis

`exp-2-m1` predicts **path-dependent, economically meaningful trade events** instead of asking only whether a fixed future close is higher.

### 1. Information-event sampling

Calendar bars are highly redundant. M1 uses a symmetric CUSUM event filter whose threshold scales with trailing five-minute realized volatility. The model is trained only when price movement has accumulated enough information to trigger an event.

The filter uses current/past returns and volatility only.

### 2. Volatility-scaled triple barrier

Signal at five-minute bar `t`:

- enter long at the next bar open,
- profit barrier: `3.0 * ATR14%`, floored at `+0.60%`, capped at `+1.80%`,
- stop barrier: `1.5 * ATR14%`, floored at `-0.30%`, capped at `-0.90%`,
- vertical barrier: 24 five-minute bars / 120 minutes.

Whichever horizontal barrier is hit first determines the path outcome. If neither is hit, the trade exits at the vertical barrier close.

If both horizontal barriers appear inside the same OHLC candle, the simulator assumes the **stop was hit first**. This is deliberately conservative because intrabar ordering is unknown from five-minute OHLC data.

Targets include the exact project commission/slippage assumptions.

### 3. Economic labels

Two related questions are modeled:

- `barrier_success`: did the profit barrier beat the stop/time barrier?
- `economic_success`: did the realized trade return finish at least `+0.15%` **after modeled costs**?

The second target forces the stack to distinguish merely correct direction from economically useful movement.

### 4. Richer point-in-time feature space

M1 keeps the original 31 features and adds regime / cross-sectional context without using future data:

- returns scaled by trailing volatility,
- short/long volatility ratio,
- trend strength normalized by ATR,
- MA-stack strength normalized by ATR,
- volume acceleration,
- breakout pressure,
- session progress,
- equal-weight five-stock market/breadth context,
- relative returns versus the five-stock basket,
- contemporaneous cross-sectional ranks for return, trend, volume and ATR.

This allows the model to distinguish "AAPL is rising" from "everything is rising" and to condition signals on volatility/liquidity regime.

### 5. Overlap-aware sample weights

Triple-barrier events overlap in time and therefore are not IID. Each event receives an average uniqueness weight derived from the number of concurrent labels across its outcome path. A mild clipped return-magnitude component additionally gives economically informative outcomes more influence without letting outliers dominate.

Class balancing is applied on top of these base weights inside each classifier fit.

### 6. Diverse base stack

Four base heads are used:

1. HistGradientBoosting -> `barrier_success`
2. ExtraTrees -> `barrier_success`
3. HistGradientBoosting -> `economic_success`
4. regularized Logistic Regression -> `economic_success`

The purpose is model diversity, not a large hyperparameter search. Hyperparameters are fixed before 2025/2026 execution evaluation.

HistGradientBoosting internal early stopping is disabled because a normal random/internal validation split is not an acceptable finance validation scheme for overlapping forward labels.

### 7. Purged expanding OOF stack

The meta-model is not trained on in-sample base predictions.

During 2024, each calendar month is predicted by fresh base models fitted only on earlier events. Training events whose label windows could overlap the next validation block are purged before fitting that fold.

Those 2024 out-of-fold base probabilities become the meta-model training set.

### 8. Meta-labeling

A regularized logistic meta-model receives:

- all four OOF base probabilities,
- base-model mean/min/max/disagreement,
- barrier-head consensus,
- economic-head consensus,
- a compact set of volatility, trend, breadth and relative-return context features.

Its target is `economic_success`.

After meta training, the final base heads are refit on all 2023-2024 events. The first forward evaluation of the fitted meta-model is 2025.

### 9. One-dimensional execution tuning

M1 does **not** search a multidimensional threshold grid.

Only five predeclared causal rolling percentiles of the final meta-score are tested on 2025:

- 90.0%
- 92.5%
- 95.0%
- 97.5%
- 99.0%

The rolling threshold uses only earlier event scores for that symbol. 2026 is seeded with score history from 2025 so the threshold is available immediately without consuming 2026 outcomes.

At a timestamp, simultaneous correlated candidates compete for one new entry slot.

### 10. Deployment gate

Passing 2025 requires all of the following:

- at least 75 completed trades,
- positive return after costs,
- profit factor >= 1.15,
- positive average trade,
- positive daily Sharpe,
- gross profit factor >= 1.25,
- more than 55% positive calendar months,
- positive median month,
- an interior percentile with both neighboring settings,
- positive median and worst return across that three-threshold local neighborhood,
- local median profit factor >= 1.05.

A good annual result sitting on one lucky threshold does not pass.

## Chronology

- Build experiment-specific event data from historical 5m bars.
- Train base/meta architecture: 2023-2024.
- Base-model meta-training predictions: purged expanding monthly OOF during 2024.
- Tune the single execution percentile axis: 2025 only.
- Freeze.
- Historical shadow backtest: 2026.

2026 has already been inspected by prior model families, and EXP-2 itself was designed after seeing those failures. Therefore the 2026 result is **not globally pristine evidence**. A future paper-trading period is required for genuinely untouched confirmation.

## Files

- `exp2_config.py` - immutable M1 experiment assumptions and feature list
- `build_exp_2_dataset.py` - CUSUM events, cross-sectional features, triple barriers and uniqueness weights
- `exp2_core.py` - base stack, purged OOF construction, meta-labeling and score generation
- `exp2_execution.py` - exact barrier execution, same-bar handling, portfolio simulation and deployment gate
- `train_exp_2_m1.py` - train stack and evaluate the first forward meta diagnostics on 2025
- `tune_exp_2_m1.py` - tune only the five predeclared causal percentiles on 2025
- `backtest_exp_2_m1.py` - frozen 2026 shadow backtest

## Workflow

```bash
python build_exp_2_dataset.py
python train_exp_2_m1.py
python tune_exp_2_m1.py
python backtest_exp_2_m1.py
```

Do not modify M1 after observing its 2026 result. A material change becomes `exp-2-m2`.

## Research basis

The design draws on established financial-ML ideas rather than treating ordinary IID tabular ML as sufficient:

- path-dependent triple-barrier labeling,
- CUSUM information-event sampling,
- meta-labeling,
- average uniqueness / overlap-aware sample weighting,
- purged chronological validation for overlapping labels,
- diverse stacked ensembles built from out-of-fold predictions,
- probability diagnostics plus economically aligned execution/backtesting.

These methods reduce specific failure modes; they do not guarantee profitability or eliminate regime risk, data-mining risk, execution-model error or the need for genuinely new paper-trading evidence.
