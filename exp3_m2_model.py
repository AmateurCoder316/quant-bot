from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from exp3_m2_config import (
    EARLY_STOPPING_PATIENCE,
    ECONOMIC_RETURN_SCALE,
    EXPANSION,
    FEATURE_COLUMNS,
    FINAL_MAX_EPOCHS,
    FINAL_MIN_EPOCHS,
    HEAD_WEIGHTS,
    MARKET_TIMEZONE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    MODEL_FEATURE_COLUMNS,
    PRIMARY_TAIL_FRACTION,
    PURGE_GAP_MINUTES,
    RANK_MIN_TARGET_GAP_STD,
    RANK_TEMPERATURE,
    SEARCH_SEED,
    SECONDARY_TAIL_FRACTION,
    SYMBOLS,
    TARGET_COLUMNS,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "EXP-3-M2 requires PyTorch. Install requirements-exp3-m2.txt first."
    ) from exc


@dataclass
class TargetScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, targets: np.ndarray) -> "TargetScaler":
        means: list[float] = []
        scales: list[float] = []
        for column in range(targets.shape[1]):
            values = targets[:, column]
            values = values[np.isfinite(values)]
            if len(values) == 0:
                means.append(0.0)
                scales.append(1.0)
                continue
            means.append(float(np.mean(values)))
            std = float(np.std(values))
            scales.append(std if std > 1e-8 else 1.0)
        return cls(np.asarray(means, dtype=np.float32), np.asarray(scales, dtype=np.float32))

    def transform(self, targets: np.ndarray) -> np.ndarray:
        return (targets - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TargetScaler":
        return cls(
            np.asarray(payload["mean"], dtype=np.float32),
            np.asarray(payload["scale"], dtype=np.float32),
        )


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        expanded = width * EXPANSION
        self.norm = nn.LayerNorm(width)
        self.network = nn.Sequential(
            nn.Linear(width, expanded),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.network(self.norm(x))


class LargeReturnMLP(nn.Module):
    """Large residual MLP for tabular 30-minute return forecasting."""

    def __init__(self, input_dim: int, width: int, blocks: int, dropout: float) -> None:
        super().__init__()
        neck = max(128, width // 2)
        self.stem = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(width, dropout) for _ in range(blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, neck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(neck, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


def seed_everything(seed: int = SEARCH_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dataframe_arrays(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (
        data[MODEL_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        data[TARGET_COLUMNS].to_numpy(dtype=np.float32),
    )


def make_cv_folds(
    data: pd.DataFrame,
    validation_year: int,
    block_months: int,
    min_train_rows: int,
) -> list[dict[str, Any]]:
    local_index = pd.DatetimeIndex(data.index).tz_convert(MARKET_TIMEZONE)
    label_end = pd.to_datetime(data["label_end_time"], utc=True, errors="coerce")
    folds: list[dict[str, Any]] = []

    for start_month in range(1, 13, block_months):
        end_month = min(12, start_month + block_months - 1)
        validation_mask = (
            (local_index.year == validation_year)
            & (local_index.month >= start_month)
            & (local_index.month <= end_month)
        )
        if not validation_mask.any():
            continue

        validation_indices = np.flatnonzero(validation_mask)
        validation_start = pd.Timestamp(data.index[validation_indices].min())
        purge_cutoff = validation_start - pd.Timedelta(minutes=PURGE_GAP_MINUTES)
        train_mask = (data.index < purge_cutoff) & (label_end < validation_start)
        train_indices = np.flatnonzero(train_mask)
        if len(train_indices) < min_train_rows:
            continue

        folds.append(
            {
                "name": f"{validation_year}-{start_month:02d}..{validation_year}-{end_month:02d}",
                "train_indices": train_indices,
                "validation_indices": validation_indices,
                "validation_start": str(validation_start),
            }
        )
    return folds


def rank_correlation(prediction: np.ndarray, actual: np.ndarray) -> float:
    finite = np.isfinite(prediction) & np.isfinite(actual)
    prediction = np.asarray(prediction, dtype=float)[finite]
    actual = np.asarray(actual, dtype=float)[finite]
    if len(prediction) < 3:
        return float("nan")
    pred_rank = pd.Series(prediction).rank(method="average").to_numpy()
    actual_rank = pd.Series(actual).rank(method="average").to_numpy()
    if np.std(pred_rank) == 0 or np.std(actual_rank) == 0:
        return 0.0
    return float(np.corrcoef(pred_rank, actual_rank)[0, 1])


def tail_metrics(prediction: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=float)
    actual = np.asarray(actual, dtype=float)
    finite = np.isfinite(prediction) & np.isfinite(actual)
    prediction = prediction[finite]
    actual = actual[finite]

    if len(actual) == 0:
        return {
            "rows": 0,
            "spearman": float("nan"),
            "mae": float("nan"),
            "top10_mean": float("nan"),
            "top10_median": float("nan"),
            "top05_mean": float("nan"),
            "top05_median": float("nan"),
            "top10_positive": float("nan"),
            "top05_positive": float("nan"),
        }

    order = np.argsort(prediction)[::-1]
    top10_n = max(1, int(math.ceil(len(actual) * PRIMARY_TAIL_FRACTION)))
    top05_n = max(1, int(math.ceil(len(actual) * SECONDARY_TAIL_FRACTION)))
    top10 = actual[order[:top10_n]]
    top05 = actual[order[:top05_n]]
    return {
        "rows": int(len(actual)),
        "spearman": rank_correlation(prediction, actual),
        "mae": float(np.mean(np.abs(prediction - actual))),
        "top10_mean": float(np.mean(top10)),
        "top10_median": float(np.median(top10)),
        "top05_mean": float(np.mean(top05)),
        "top05_median": float(np.median(top05)),
        "top10_positive": float(np.mean(top10 > 0)),
        "top05_positive": float(np.mean(top05 > 0)),
    }


def economic_checkpoint_score(metrics: dict[str, float]) -> float:
    top10 = np.clip(metrics["top10_mean"] / ECONOMIC_RETURN_SCALE, -3.0, 3.0)
    top05 = np.clip(metrics["top05_mean"] / ECONOMIC_RETURN_SCALE, -3.0, 3.0)
    rank = np.clip(metrics["spearman"] / 0.10, -3.0, 3.0)
    return float(0.45 * top10 + 0.35 * top05 + 0.20 * rank)


def summarize_fold_metrics(metrics: list[dict[str, float]]) -> dict[str, Any]:
    top10 = np.asarray([item["top10_mean"] for item in metrics], dtype=float)
    top05 = np.asarray([item["top05_mean"] for item in metrics], dtype=float)
    spearman = np.asarray([item["spearman"] for item in metrics], dtype=float)
    mae = np.asarray([item["mae"] for item in metrics], dtype=float)

    mean_top10 = float(np.nanmean(top10))
    mean_top05 = float(np.nanmean(top05))
    mean_spearman = float(np.nanmean(spearman))
    median_top10 = float(np.nanmedian(top10))
    worst_top10 = float(np.nanmin(top10))
    top10_std = float(np.nanstd(top10))
    positive_fold_fraction = float(np.mean(top10 > 0))

    eligible = bool(
        mean_top10 > 0
        and mean_top05 > 0
        and median_top10 > 0
        and mean_spearman > 0
        and positive_fold_fraction >= (4 / 6)
    )

    top10_component = np.clip(mean_top10 / ECONOMIC_RETURN_SCALE, -3.0, 3.0)
    top05_component = np.clip(mean_top05 / ECONOMIC_RETURN_SCALE, -3.0, 3.0)
    rank_component = np.clip(mean_spearman / 0.10, -3.0, 3.0)
    worst_component = np.clip(worst_top10 / ECONOMIC_RETURN_SCALE, -3.0, 3.0)
    stability_penalty = 0.10 * np.clip(top10_std / ECONOMIC_RETURN_SCALE, 0.0, 3.0)

    score = (
        0.30 * top10_component
        + 0.25 * top05_component
        + 0.20 * rank_component
        + 0.15 * worst_component
        + 0.10 * positive_fold_fraction
        - stability_penalty
    )
    if not eligible:
        score -= 2.0

    return {
        "objective": float(score),
        "eligible": eligible,
        "mean_spearman": mean_spearman,
        "mean_mae": float(np.nanmean(mae)),
        "mean_top10": mean_top10,
        "median_top10": median_top10,
        "worst_top10": worst_top10,
        "mean_top05": mean_top05,
        "top10_std": top10_std,
        "positive_fold_fraction": positive_fold_fraction,
    }


def _masked_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    losses = torch.nn.functional.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=0.5,
    )
    weights = torch.as_tensor(HEAD_WEIGHTS, device=prediction.device).view(1, -1)
    weighted_mask = mask * weights
    denominator = weighted_mask.sum().clamp_min(1.0)
    return (losses * weighted_mask).sum() / denominator


def _ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred = prediction[:, 1]
    actual = target[:, 1]
    valid = mask[:, 1] > 0
    if int(valid.sum()) < 4:
        return pred.sum() * 0.0

    pred = pred[valid]
    actual = actual[valid]
    permutation = torch.randperm(len(pred), device=pred.device)
    paired_pred = pred[permutation]
    paired_actual = actual[permutation]
    target_difference = actual - paired_actual
    usable = target_difference.abs() >= RANK_MIN_TARGET_GAP_STD
    if int(usable.sum()) < 2:
        return pred.sum() * 0.0

    direction = torch.sign(target_difference[usable])
    score_difference = pred[usable] - paired_pred[usable]
    margin = direction * score_difference / RANK_TEMPERATURE
    return torch.nn.functional.softplus(-margin).mean()


def _total_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    rank_weight: float,
) -> torch.Tensor:
    regression = _masked_regression_loss(prediction, target, mask)
    ranking = _ranking_loss(prediction, target, mask)
    return (1.0 - rank_weight) * regression + rank_weight * ranking


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    mask = np.isfinite(y).astype(np.float32)
    filled = np.nan_to_num(y, nan=0.0).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(filled),
        torch.from_numpy(mask),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def predict_array(
    model: LargeReturnMLP,
    x_scaled: np.ndarray,
    target_scaler: TargetScaler,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    tensor = torch.from_numpy(x_scaled.astype(np.float32))
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(device)
            outputs.append(model(batch).cpu().numpy())
    if not outputs:
        return np.empty((0, 3), dtype=np.float32)
    return target_scaler.inverse(np.concatenate(outputs, axis=0))


def train_one_model(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    params: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> tuple[LargeReturnMLP, StandardScaler, TargetScaler, int, dict[str, float]]:
    seed_everything(seed)
    x_train_raw, y_train_raw = dataframe_arrays(train_data)
    x_valid_raw, y_valid_raw = dataframe_arrays(validation_data)

    feature_scaler = StandardScaler().fit(x_train_raw)
    x_train = feature_scaler.transform(x_train_raw).astype(np.float32)
    x_valid = feature_scaler.transform(x_valid_raw).astype(np.float32)

    target_scaler = TargetScaler.fit(y_train_raw)
    y_train = target_scaler.transform(y_train_raw).astype(np.float32)

    model = LargeReturnMLP(
        input_dim=x_train.shape[1],
        width=int(params["width"]),
        blocks=int(params["blocks"]),
        dropout=float(params["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=float(params["learning_rate"]) * 0.05
    )
    loader = _loader(x_train, y_train, int(params["batch_size"]), True)

    best_score = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = MIN_EPOCHS
    best_metrics: dict[str, float] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb, mb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = _total_loss(model(xb), yb, mb, float(params["rank_weight"]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

        if epoch < MIN_EPOCHS:
            continue

        model.eval()
        prediction = predict_array(model, x_valid, target_scaler, device)
        metrics = tail_metrics(prediction[:, 1], y_valid_raw[:, 1])
        checkpoint_score = economic_checkpoint_score(metrics)
        metrics["checkpoint_score"] = checkpoint_score
        metrics["epoch"] = int(epoch)

        if checkpoint_score > best_score + 1e-4:
            best_score = checkpoint_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metrics = metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            break

    model.load_state_dict(best_state)
    if best_metrics is None:
        prediction = predict_array(model, x_valid, target_scaler, device)
        best_metrics = tail_metrics(prediction[:, 1], y_valid_raw[:, 1])
        best_metrics["checkpoint_score"] = economic_checkpoint_score(best_metrics)
        best_metrics["epoch"] = int(best_epoch)

    return model, feature_scaler, target_scaler, best_epoch, best_metrics


def choose_final_epochs(best_epochs: list[int]) -> int:
    if not best_epochs:
        return FINAL_MIN_EPOCHS
    median = float(np.median(np.asarray(best_epochs, dtype=float)))
    # Full-data training has more examples than any individual fold. Give it a
    # little extra optimization time, while guaranteeing M1's 2-epoch failure
    # can never recur.
    proposed = int(round(median * 1.15))
    return int(np.clip(proposed, FINAL_MIN_EPOCHS, FINAL_MAX_EPOCHS))


def fit_full_model(
    data: pd.DataFrame,
    params: dict[str, Any],
    epochs: int,
    *,
    seed: int,
    device: torch.device,
) -> tuple[LargeReturnMLP, StandardScaler, TargetScaler]:
    seed_everything(seed)
    x_raw, y_raw = dataframe_arrays(data)
    feature_scaler = StandardScaler().fit(x_raw)
    x = feature_scaler.transform(x_raw).astype(np.float32)
    target_scaler = TargetScaler.fit(y_raw)
    y = target_scaler.transform(y_raw).astype(np.float32)

    model = LargeReturnMLP(
        input_dim=x.shape[1],
        width=int(params["width"]),
        blocks=int(params["blocks"]),
        dropout=float(params["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=float(params["learning_rate"]) * 0.05
    )
    loader = _loader(x, y, int(params["batch_size"]), True)

    for _ in range(max(1, int(epochs))):
        model.train()
        for xb, yb, mb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = _total_loss(model(xb), yb, mb, float(params["rank_weight"]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    return model, feature_scaler, target_scaler


def predict_dataframe(
    model: LargeReturnMLP,
    feature_scaler: StandardScaler,
    target_scaler: TargetScaler,
    data: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    x = feature_scaler.transform(
        data[MODEL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    prediction = predict_array(model, x, target_scaler, device)
    result = data.copy()
    result["pred_net_1h"] = prediction[:, 0]
    result["pred_net_2h"] = prediction[:, 1]
    result["pred_net_4h"] = prediction[:, 2]
    return result


def save_model_bundle(
    model: LargeReturnMLP,
    feature_scaler: StandardScaler,
    target_scaler: TargetScaler,
    params: dict[str, Any],
    model_path: Path,
    scaler_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": len(MODEL_FEATURE_COLUMNS),
        "width": int(params["width"]),
        "blocks": int(params["blocks"]),
        "dropout": float(params["dropout"]),
    }
    torch.save(checkpoint, model_path)
    joblib.dump(feature_scaler, scaler_path)

    payload = dict(metadata)
    payload.update(
        {
            "model_features": MODEL_FEATURE_COLUMNS,
            "numeric_features": FEATURE_COLUMNS,
            "symbols": SYMBOLS,
            "target_scaler": target_scaler.to_dict(),
            "params": params,
        }
    )
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model_bundle(
    model_path: Path,
    scaler_path: Path,
    metadata_path: Path,
    device: torch.device | None = None,
) -> tuple[LargeReturnMLP, StandardScaler, TargetScaler, dict[str, Any], torch.device]:
    device = device or choose_device()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = LargeReturnMLP(
        input_dim=int(checkpoint["input_dim"]),
        width=int(checkpoint["width"]),
        blocks=int(checkpoint["blocks"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    feature_scaler = joblib.load(scaler_path)
    target_scaler = TargetScaler.from_dict(metadata["target_scaler"])
    return model, feature_scaler, target_scaler, metadata, device
