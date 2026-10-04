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

from exp4_config import (
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
    TARGET_COLUMNS,
    TARGET_SYMBOLS,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise RuntimeError("EXP-4 requires PyTorch. Install requirements-exp4-m1.txt first.") from exc


@dataclass
class TargetScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, y: np.ndarray) -> "TargetScaler":
        mean = np.nanmean(y, axis=0).astype(np.float32)
        scale = np.nanstd(y, axis=0).astype(np.float32)
        scale[scale < 1e-8] = 1.0
        return cls(mean, scale)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.scale

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return y * self.scale + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.astype(float).tolist(), "scale": self.scale.astype(float).tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TargetScaler":
        return cls(np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["scale"], dtype=np.float32))


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        expanded = width * EXPANSION
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, expanded),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class RegimeReturnMLP(nn.Module):
    def __init__(self, input_dim: int, width: int, blocks: int, dropout: float) -> None:
        super().__init__()
        neck = max(192, width // 2)
        self.stem = nn.Sequential(
            nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, neck), nn.GELU(), nn.Dropout(dropout), nn.Linear(neck, 2)
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


def make_cv_folds(data: pd.DataFrame, validation_year: int, block_months: int, min_train_rows: int) -> list[dict[str, Any]]:
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
        valid_indices = np.flatnonzero(validation_mask)
        valid_start = pd.Timestamp(data.index[valid_indices].min())
        purge_cutoff = valid_start - pd.Timedelta(minutes=PURGE_GAP_MINUTES)
        train_mask = (data.index < purge_cutoff) & (label_end < valid_start)
        train_indices = np.flatnonzero(train_mask)
        if len(train_indices) < min_train_rows:
            continue
        folds.append({
            "name": f"{validation_year}-{start_month:02d}..{validation_year}-{end_month:02d}",
            "train_indices": train_indices,
            "validation_indices": valid_indices,
        })
    return folds


def rank_correlation(prediction: np.ndarray, actual: np.ndarray) -> float:
    finite = np.isfinite(prediction) & np.isfinite(actual)
    prediction, actual = prediction[finite], actual[finite]
    if len(prediction) < 3:
        return float("nan")
    p = pd.Series(prediction).rank().to_numpy()
    a = pd.Series(actual).rank().to_numpy()
    if np.std(p) == 0 or np.std(a) == 0:
        return 0.0
    return float(np.corrcoef(p, a)[0, 1])


def tail_metrics(prediction: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=float)
    actual = np.asarray(actual, dtype=float)
    finite = np.isfinite(prediction) & np.isfinite(actual)
    prediction, actual = prediction[finite], actual[finite]
    if len(actual) == 0:
        return {k: float("nan") for k in ["spearman", "mae", "top_primary_mean", "top_primary_median", "top_secondary_mean", "top_secondary_median", "top_primary_positive", "top_secondary_positive"]} | {"rows": 0}
    order = np.argsort(prediction)[::-1]
    n1 = max(1, int(math.ceil(len(actual) * PRIMARY_TAIL_FRACTION)))
    n2 = max(1, int(math.ceil(len(actual) * SECONDARY_TAIL_FRACTION)))
    top1, top2 = actual[order[:n1]], actual[order[:n2]]
    return {
        "rows": int(len(actual)),
        "spearman": rank_correlation(prediction, actual),
        "mae": float(np.mean(np.abs(prediction - actual))),
        "top_primary_mean": float(np.mean(top1)),
        "top_primary_median": float(np.median(top1)),
        "top_secondary_mean": float(np.mean(top2)),
        "top_secondary_median": float(np.median(top2)),
        "top_primary_positive": float(np.mean(top1 > 0)),
        "top_secondary_positive": float(np.mean(top2 > 0)),
    }


def economic_checkpoint_score(metrics: dict[str, float]) -> float:
    top1 = np.clip(metrics["top_primary_mean"] / ECONOMIC_RETURN_SCALE, -3, 3)
    top2 = np.clip(metrics["top_secondary_mean"] / ECONOMIC_RETURN_SCALE, -3, 3)
    rank = np.clip(metrics["spearman"] / 0.08, -3, 3)
    return float(0.42 * top1 + 0.38 * top2 + 0.20 * rank)


def summarize_fold_metrics(metrics: list[dict[str, float]]) -> dict[str, Any]:
    top1 = np.asarray([m["top_primary_mean"] for m in metrics], dtype=float)
    top2 = np.asarray([m["top_secondary_mean"] for m in metrics], dtype=float)
    rank = np.asarray([m["spearman"] for m in metrics], dtype=float)
    mean1, mean2, mean_rank = float(np.nanmean(top1)), float(np.nanmean(top2)), float(np.nanmean(rank))
    worst1 = float(np.nanmin(top1))
    positive_folds = float(np.mean(top1 > 0))
    eligible = bool(mean1 > 0 and mean2 > 0 and mean_rank > 0 and positive_folds >= 4 / 6)
    score = (
        0.30 * np.clip(mean1 / ECONOMIC_RETURN_SCALE, -3, 3)
        + 0.30 * np.clip(mean2 / ECONOMIC_RETURN_SCALE, -3, 3)
        + 0.20 * np.clip(mean_rank / 0.08, -3, 3)
        + 0.10 * np.clip(worst1 / ECONOMIC_RETURN_SCALE, -3, 3)
        + 0.10 * positive_folds
        - 0.08 * np.clip(np.nanstd(top1) / ECONOMIC_RETURN_SCALE, 0, 3)
    )
    if not eligible:
        score -= 2.0
    return {
        "objective": float(score), "eligible": eligible,
        "mean_spearman": mean_rank, "mean_top_primary": mean1, "mean_top_secondary": mean2,
        "worst_top_primary": worst1, "positive_fold_fraction": positive_folds,
        "top_primary_std": float(np.nanstd(top1)),
    }


def _masked_regression_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=0.5)
    weights = torch.as_tensor(HEAD_WEIGHTS, dtype=pred.dtype, device=pred.device).view(1, -1)
    weighted = mask * weights
    return (losses * weighted).sum() / weighted.sum().clamp_min(1.0)


def _ranking_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    score, actual = pred[:, 1], target[:, 1]
    valid = mask[:, 1] > 0
    if int(valid.sum()) < 4:
        return score.sum() * 0
    score, actual = score[valid], actual[valid]
    perm = torch.randperm(len(score), device=score.device)
    diff = actual - actual[perm]
    usable = diff.abs() >= RANK_MIN_TARGET_GAP_STD
    if int(usable.sum()) < 2:
        return score.sum() * 0
    direction = torch.sign(diff[usable])
    margin = direction * (score[usable] - score[perm][usable]) / RANK_TEMPERATURE
    return torch.nn.functional.softplus(-margin).mean()


def _total_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, rank_weight: float) -> torch.Tensor:
    return (1 - rank_weight) * _masked_regression_loss(pred, target, mask) + rank_weight * _ranking_loss(pred, target, mask)


def _loader(x, y, batch_size: int, shuffle: bool) -> DataLoader:
    mask = np.isfinite(y).astype(np.float32)
    filled = np.nan_to_num(y, nan=0.0).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(filled), torch.from_numpy(mask))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def predict_array(model, x_scaled, target_scaler, device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(x_scaled.astype(np.float32))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            chunks.append(model(tensor[start:start+batch_size].to(device)).cpu().numpy())
    return target_scaler.inverse(np.concatenate(chunks)) if chunks else np.empty((0, 2), dtype=np.float32)


def train_one_model(train_data, validation_data, params, *, seed: int, device):
    seed_everything(seed)
    x_train_raw, y_train_raw = dataframe_arrays(train_data)
    x_valid_raw, y_valid_raw = dataframe_arrays(validation_data)
    scaler = StandardScaler().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw).astype(np.float32)
    x_valid = scaler.transform(x_valid_raw).astype(np.float32)
    target_scaler = TargetScaler.fit(y_train_raw)
    y_train = target_scaler.transform(y_train_raw).astype(np.float32)

    model = RegimeReturnMLP(x_train.shape[1], int(params["width"]), int(params["blocks"]), float(params["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
    loader = _loader(x_train, y_train, int(params["batch_size"]), True)

    best_score = -float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = MIN_EPOCHS
    stale = 0
    best_metrics = None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _total_loss(model(xb), yb, mb, float(params["rank_weight"]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if epoch < MIN_EPOCHS or epoch % 2:
            continue
        prediction = predict_array(model, x_valid, target_scaler, device)
        metrics = tail_metrics(prediction[:, 1], y_valid_raw[:, 1])
        score = economic_checkpoint_score(metrics)
        if score > best_score + 1e-4:
            best_score, best_epoch, best_metrics = score, epoch, metrics
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 2
        if stale >= EARLY_STOPPING_PATIENCE:
            break

    model.load_state_dict(best_state)
    if best_metrics is None:
        prediction = predict_array(model, x_valid, target_scaler, device)
        best_metrics = tail_metrics(prediction[:, 1], y_valid_raw[:, 1])
    best_metrics["economic_checkpoint_score"] = float(best_score)
    best_metrics["best_epoch"] = int(best_epoch)
    return model, scaler, target_scaler, best_epoch, best_metrics


def fit_full_model(data, params, epochs: int, *, seed: int, device):
    seed_everything(seed)
    x_raw, y_raw = dataframe_arrays(data)
    scaler = StandardScaler().fit(x_raw)
    x = scaler.transform(x_raw).astype(np.float32)
    target_scaler = TargetScaler.fit(y_raw)
    y = target_scaler.transform(y_raw).astype(np.float32)
    model = RegimeReturnMLP(x.shape[1], int(params["width"]), int(params["blocks"]), float(params["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
    loader = _loader(x, y, int(params["batch_size"]), True)
    final_epochs = max(FINAL_MIN_EPOCHS, min(FINAL_MAX_EPOCHS, int(epochs)))
    for _ in range(final_epochs):
        model.train()
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _total_loss(model(xb), yb, mb, float(params["rank_weight"]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    return model, scaler, target_scaler, final_epochs


def predict_dataframe(model, scaler, target_scaler, data, device):
    x = scaler.transform(data[MODEL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)).astype(np.float32)
    prediction = predict_array(model, x, target_scaler, device)
    result = data.copy()
    result["pred_net_2h"] = prediction[:, 0]
    result["pred_net_4h"] = prediction[:, 1]
    return result


def save_model_bundle(model, scaler, target_scaler, params, model_path: Path, scaler_path: Path, metadata_path: Path, metadata: dict[str, Any]) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "input_dim": len(MODEL_FEATURE_COLUMNS),
        "width": int(params["width"]), "blocks": int(params["blocks"]), "dropout": float(params["dropout"]),
    }, model_path)
    joblib.dump(scaler, scaler_path)
    payload = dict(metadata)
    payload.update({
        "model_features": MODEL_FEATURE_COLUMNS, "numeric_features": FEATURE_COLUMNS,
        "symbols": TARGET_SYMBOLS, "target_scaler": target_scaler.to_dict(), "params": params,
    })
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model_bundle(model_path: Path, scaler_path: Path, metadata_path: Path, device=None):
    device = device or choose_device()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = RegimeReturnMLP(int(checkpoint["input_dim"]), int(checkpoint["width"]), int(checkpoint["blocks"]), float(checkpoint["dropout"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, joblib.load(scaler_path), TargetScaler.from_dict(metadata["target_scaler"]), metadata, device
