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

from exp3_config import (
    EARLY_STOPPING_PATIENCE,
    ECONOMIC_RETURN_SCALE,
    FEATURE_COLUMNS,
    MARKET_TIMEZONE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    MODEL_FEATURE_COLUMNS,
    PRIMARY_TAIL_FRACTION,
    PURGE_GAP_MINUTES,
    SECONDARY_TAIL_FRACTION,
    SEARCH_SEED,
    SYMBOLS,
    TARGET_1H,
    TARGET_2H,
    TARGET_4H,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - user-facing dependency error
    raise RuntimeError(
        "EXP-3 requires PyTorch. Install requirements-exp3-m1.txt first."
    ) from exc


TARGET_COLUMNS = [TARGET_1H, TARGET_2H, TARGET_4H]
HEAD_WEIGHTS = np.array([0.25, 1.0, 0.25], dtype=np.float32)


@dataclass
class TargetScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, targets: np.ndarray) -> "TargetScaler":
        means = []
        scales = []
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
    def from_dict(cls, value: dict[str, Any]) -> "TargetScaler":
        return cls(
            np.asarray(value["mean"], dtype=np.float32),
            np.asarray(value["scale"], dtype=np.float32),
        )


class ReturnMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layout: list[int],
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()

        activation_cls = nn.GELU if activation == "gelu" else nn.ReLU
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_layout:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.LayerNorm(width),
                    activation_cls(),
                    nn.Dropout(dropout),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


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
    x = data[MODEL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = data[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    return x, y


def make_cv_folds(
    data: pd.DataFrame,
    validation_year: int,
    block_months: int,
    min_train_rows: int,
) -> list[dict[str, Any]]:
    local_index = pd.DatetimeIndex(data.index).tz_convert(MARKET_TIMEZONE)
    label_end = pd.to_datetime(data["label_end_time"], utc=True, errors="coerce")
    folds = []

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
    instability = float(np.nanstd(top10))

    eligible = bool(
        mean_top10 > 0
        and mean_top05 > 0
        and median_top10 > 0
        and mean_spearman > 0
    )

    top10_component = np.clip(mean_top10 / ECONOMIC_RETURN_SCALE, -3.0, 3.0)
    top05_component = np.clip(mean_top05 / ECONOMIC_RETURN_SCALE, -3.0, 3.0)
    rank_component = np.clip(mean_spearman / 0.15, -2.0, 3.0)
    worst_component = np.clip(worst_top10 / ECONOMIC_RETURN_SCALE, -3.0, 3.0)
    instability_penalty = 0.10 * np.clip(instability / ECONOMIC_RETURN_SCALE, 0.0, 3.0)

    score = (
        0.35 * top10_component
        + 0.30 * top05_component
        + 0.20 * rank_component
        + 0.15 * worst_component
        - instability_penalty
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
        "top10_std": instability,
    }


def _masked_loss(
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


def train_one_model(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    params: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> tuple[ReturnMLP, StandardScaler, TargetScaler, int, dict[str, float]]:
    seed_everything(seed)

    x_train_raw, y_train_raw = dataframe_arrays(train_data)
    x_valid_raw, y_valid_raw = dataframe_arrays(validation_data)

    feature_scaler = StandardScaler().fit(x_train_raw)
    x_train = feature_scaler.transform(x_train_raw).astype(np.float32)
    x_valid = feature_scaler.transform(x_valid_raw).astype(np.float32)

    target_scaler = TargetScaler.fit(y_train_raw)
    y_train = target_scaler.transform(y_train_raw).astype(np.float32)
    y_valid = target_scaler.transform(y_valid_raw).astype(np.float32)

    model = ReturnMLP(
        input_dim=x_train.shape[1],
        hidden_layout=list(params["hidden_layout"]),
        dropout=float(params["dropout"]),
        activation=str(params["activation"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )

    train_loader = _loader(x_train, y_train, int(params["batch_size"]), True)
    valid_loader = _loader(x_valid, y_valid, int(params["batch_size"]), False)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb, mb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_loss(model(xb), yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for xb, yb, mb in valid_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                mb = mb.to(device, non_blocking=True)
                validation_losses.append(float(_masked_loss(model(xb), yb, mb).item()))

        validation_loss = float(np.mean(validation_losses))
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1

        if epoch >= MIN_EPOCHS and patience >= EARLY_STOPPING_PATIENCE:
            break

    model.load_state_dict(best_state)
    prediction = predict_array(model, x_valid, target_scaler, device)
    metrics = tail_metrics(prediction[:, 1], y_valid_raw[:, 1])
    metrics["validation_loss"] = best_loss
    metrics["best_epoch"] = int(best_epoch)
    return model, feature_scaler, target_scaler, best_epoch, metrics


def fit_full_model(
    data: pd.DataFrame,
    params: dict[str, Any],
    epochs: int,
    *,
    seed: int,
    device: torch.device,
) -> tuple[ReturnMLP, StandardScaler, TargetScaler]:
    seed_everything(seed)
    x_raw, y_raw = dataframe_arrays(data)
    feature_scaler = StandardScaler().fit(x_raw)
    x = feature_scaler.transform(x_raw).astype(np.float32)
    target_scaler = TargetScaler.fit(y_raw)
    y = target_scaler.transform(y_raw).astype(np.float32)

    model = ReturnMLP(
        input_dim=x.shape[1],
        hidden_layout=list(params["hidden_layout"]),
        dropout=float(params["dropout"]),
        activation=str(params["activation"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    loader = _loader(x, y, int(params["batch_size"]), True)

    model.train()
    for _ in range(max(1, int(epochs))):
        for xb, yb, mb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_loss(model(xb), yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    model.eval()
    return model, feature_scaler, target_scaler


def predict_array(
    model: ReturnMLP,
    x_scaled: np.ndarray,
    target_scaler: TargetScaler,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    outputs = []
    tensor = torch.from_numpy(x_scaled.astype(np.float32))
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(device)
            outputs.append(model(batch).cpu().numpy())
    scaled = np.concatenate(outputs, axis=0) if outputs else np.empty((0, 3), dtype=np.float32)
    return target_scaler.inverse(scaled)


def predict_dataframe(
    model: ReturnMLP,
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
    model: ReturnMLP,
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
        "hidden_layout": list(params["hidden_layout"]),
        "dropout": float(params["dropout"]),
        "activation": str(params["activation"]),
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
) -> tuple[ReturnMLP, StandardScaler, TargetScaler, dict[str, Any], torch.device]:
    device = device or choose_device()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = ReturnMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_layout=list(checkpoint["hidden_layout"]),
        dropout=float(checkpoint["dropout"]),
        activation=str(checkpoint["activation"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    feature_scaler = joblib.load(scaler_path)
    target_scaler = TargetScaler.from_dict(metadata["target_scaler"])
    return model, feature_scaler, target_scaler, metadata, device
