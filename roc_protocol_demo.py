#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROC/AUC protocol for binary classifiers with Youden's J threshold selection.
Implements:
  - Reproducible stratified splits
  - Random Forest (scikit-learn) evaluation from class-1 probabilities
  - PyTorch CNN and LSTM evaluation from logits -> probabilities
  - Validation-set ROC to choose a single tau* (Youden's J)
  - Test-set AUC from scores and point metrics at fixed tau*
This script is intentionally verbose (≈250+ lines) to serve as a reviewer-facing
artifact. Replace the toy data loaders with your real dataset.
"""

from __future__ import annotations
import math
import os
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# PyTorch (CPU is fine for evaluation; switch to CUDA if available in your environment)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set RNG seeds for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def stratified_splits(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> SplitData:
    """
    Produce stratified train/val/test splits with a fixed random seed.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=seed
    )
    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


# -----------------------------------------------------------------------------
# Metrics and ROC helpers
# -----------------------------------------------------------------------------

@dataclass
class RocResult:
    fpr: np.ndarray
    tpr: np.ndarray
    thr: np.ndarray
    auc_val: float


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> RocResult:
    """
    Compute ROC and AUC from probability scores.
    Returns FPR, TPR, thresholds, and AUC value.
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    auc_val = auc(fpr, tpr)
    return RocResult(fpr=fpr, tpr=tpr, thr=thr, auc_val=auc_val)


def select_threshold_youden(fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray) -> float:
    """
    Choose tau* that maximizes Youden's J = TPR - FPR on the validation ROC.
    """
    J = tpr - fpr
    idx = np.argmax(J)
    tau_star = thr[idx]
    return float(tau_star)


@dataclass
class PointMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


def evaluate_at_threshold(y_true: np.ndarray, y_score: np.ndarray, tau_star: float) -> PointMetrics:
    """
    Threshold scores at tau* to compute accuracy, precision, recall, F1.
    """
    y_pred = (y_score >= tau_star).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return PointMetrics(acc, prec, rec, f1)


# -----------------------------------------------------------------------------
# Random Forest pipeline
# -----------------------------------------------------------------------------

@dataclass
class RFEvaluation:
    tau_star: float
    val_auc: float
    test_auc: float
    test_points: PointMetrics


def evaluate_random_forest(split: SplitData, n_estimators: int = 300, seed: int = 42) -> RFEvaluation:
    """
    Train RF on train, determine tau* on validation ROC from class-1 probabilities,
    compute AUC on test from scores, then point metrics at fixed tau*.
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(split.X_train)
    X_val = scaler.transform(split.X_val)
    X_test = scaler.transform(split.X_test)

    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, class_weight="balanced_subsample"
    )
    rf.fit(X_train, split.y_train)

    # Probabilities for positive class
    val_scores = rf.predict_proba(X_val)[:, 1]
    test_scores = rf.predict_proba(X_test)[:, 1]

    # Validation ROC -> tau*
    roc_val = compute_roc_auc(split.y_val, val_scores)
    tau_star = select_threshold_youden(roc_val.fpr, roc_val.tpr, roc_val.thr)

    # Test-set AUC from scores
    test_auc = roc_auc_score(split.y_test, test_scores)

    # Point metrics at fixed tau*
    points = evaluate_at_threshold(split.y_test, test_scores, tau_star)

    return RFEvaluation(
        tau_star=tau_star,
        val_auc=roc_val.auc_val,
        test_auc=test_auc,
        test_points=points,
    )


# -----------------------------------------------------------------------------
# Torch models: simple CNN & LSTM for binary classification
# -----------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    Minimal 1D CNN for sequence-like inputs [N, C, L]. Outputs raw logits of shape [N].
    """
    def __init__(self, in_channels: int = 1, num_filters: int = 16, hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, num_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_filters, hidden)
        self.fc_out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, L]
        z = F.relu(self.conv1(x))
        z = F.relu(self.conv2(z))
        z = self.pool(z).squeeze(-1)    # [N, num_filters]
        z = F.relu(self.fc1(z))
        logits = self.fc_out(z).squeeze(-1)  # [N]
        return logits


class SimpleLSTM(nn.Module):
    """
    Minimal LSTM for sequence inputs [N, L, D]. Outputs raw logits of shape [N].
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, L, D]
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]          # [N, hidden_dim]
        logits = self.fc_out(h_last).squeeze(-1)  # [N]
        return logits


# -----------------------------------------------------------------------------
# Generic evaluation for torch models (logits -> probabilities -> ROC/AUC)
# -----------------------------------------------------------------------------

@dataclass
class TorchEvalResult:
    tau_star: float
    val_auc: float
    test_auc: float
    test_points: PointMetrics


def torch_predict_scores(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Run model to obtain class-1 probabilities from raw logits via sigmoid.
    Accepts DataLoader batches of the form (xb, yb) and ignores labels.
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            # Support either (xb,) or (xb, yb)
            xb = batch[0].to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).squeeze(-1)  # convert logits -> class-1 probability
            scores.append(probs.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def evaluate_torch_model(
    model: nn.Module,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    test_ds: TensorDataset,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> TorchEvalResult:
    """
    Compute validation ROC to pick tau*, test AUC from scores, and point metrics at fixed tau*.
    Assumes the model outputs raw logits for the positive class.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Scores
    val_scores = torch_predict_scores(model, val_loader, device=device)
    test_scores = torch_predict_scores(model, test_loader, device=device)

    # ROC on validation -> tau*
    y_val = val_ds.tensors[1].cpu().numpy().astype(int)
    roc_val = compute_roc_auc(y_val, val_scores)
    tau_star = select_threshold_youden(roc_val.fpr, roc_val.tpr, roc_val.thr)

    # Test AUC from scores (not labels)
    y_test = test_ds.tensors[1].cpu().numpy().astype(int)
    test_auc = roc_auc_score(y_test, test_scores)

    # Test point metrics at fixed tau*
    points = evaluate_at_threshold(y_test, test_scores, tau_star)

    return TorchEvalResult(
        tau_star=tau_star,
        val_auc=roc_val.auc_val,
        test_auc=test_auc,
        test_points=points,
    )


# -----------------------------------------------------------------------------
# Demo data helpers (toy) -- replace with your real loaders
# -----------------------------------------------------------------------------

def make_toy_tabular(n: int = 1000, d: int = 12, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a toy binary classification dataset with slight class imbalance.
    """
    rng = np.random.default_rng(seed)
    X_pos = rng.normal(loc=0.7, scale=1.0, size=(n // 2, d))
    X_neg = rng.normal(loc=-0.3, scale=1.0, size=(n - n // 2, d))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n // 2) + [0] * (n - n // 2), dtype=int)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def make_toy_sequences_cnn(n: int = 1200, L: int = 64, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Toy 1D signals for CNN: shape [N, 1, L]."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, L))
    signal = base + 0.6 * np.sin(np.linspace(0, 10, L))[None, :]
    y = rng.integers(0, 2, size=n)
    X = np.where(y[:, None] == 1, signal, base)  # class-dependent pattern
    X = X[:, None, :]  # add channel dim
    return X.astype(np.float32), y.astype(int)


def make_toy_sequences_lstm(n: int = 1200, L: int = 24, D: int = 8, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Toy sequences for LSTM: shape [N, L, D]."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, L, D))
    drift = np.linspace(0, 1, L)[None, :, None]
    y = rng.integers(0, 2, size=n)
    X = np.where(y[:, None, None] == 1, base + 0.5 * drift, base)
    return X.astype(np.float32), y.astype(int)


# -----------------------------------------------------------------------------
# Main (demonstration scaffold)
# -----------------------------------------------------------------------------

def main_demo() -> Dict[str, Any]:
    """
    Demonstrates the full protocol for RF, CNN, and LSTM on toy data.
    Replace toy makers with your real data loaders to reproduce paper numbers.
    """
    set_seed(123)

    # --- Random Forest on tabular data ---
    X_tab, y_tab = make_toy_tabular(n=2000, d=10, seed=123)
    splits_tab = stratified_splits(X_tab, y_tab, seed=123, val_size=0.2, test_size=0.2)
    rf_eval = evaluate_random_forest(splits_tab, n_estimators=400, seed=123)

    # --- CNN on 1D signals ---
    X_cnn, y_cnn = make_toy_sequences_cnn(n=1800, L=96, seed=321)
    split_cnn = stratified_splits(X_cnn.reshape(len(X_cnn), -1), y_cnn, seed=321)  # split by labels
    # Convert back to tensors with original shape
    tr = torch.tensor(X_cnn[0:len(split_cnn.X_train)], dtype=torch.float32)  # Not used; placeholder
    # Proper split tensors:
    Xtr = torch.tensor(split_cnn.X_train.reshape(-1, 1, 96), dtype=torch.float32)
    Xva = torch.tensor(split_cnn.X_val.reshape(-1, 1, 96), dtype=torch.float32)
    Xte = torch.tensor(split_cnn.X_test.reshape(-1, 1, 96), dtype=torch.float32)
    ytr = torch.tensor(split_cnn.y_train, dtype=torch.float32)
    yva = torch.tensor(split_cnn.y_val, dtype=torch.float32)
    yte = torch.tensor(split_cnn.y_test, dtype=torch.float32)

    cnn = SimpleCNN(in_channels=1, num_filters=16, hidden=32)
    # NOTE: For brevity we skip training and assume a trained model is loaded.
    # In your code, train on (X_train, y_train) and save the model; here we simulate random weights.
    cnn_eval = evaluate_torch_model(
        model=cnn,
        train_ds=TensorDataset(Xtr, ytr),
        val_ds=TensorDataset(Xva, yva),
        test_ds=TensorDataset(Xte, yte),
        batch_size=128,
    )

    # --- LSTM on multivariate sequences ---
    X_lstm, y_lstm = make_toy_sequences_lstm(n=1800, L=24, D=8, seed=999)
    split_lstm = stratified_splits(X_lstm.reshape(len(X_lstm), -1), y_lstm, seed=999)
    XtrL = torch.tensor(split_lstm.X_train.reshape(-1, 24, 8), dtype=torch.float32)
    XvaL = torch.tensor(split_lstm.X_val.reshape(-1, 24, 8), dtype=torch.float32)
    XteL = torch.tensor(split_lstm.X_test.reshape(-1, 24, 8), dtype=torch.float32)
    ytrL = torch.tensor(split_lstm.y_train, dtype=torch.float32)
    yvaL = torch.tensor(split_lstm.y_val, dtype=torch.float32)
    yteL = torch.tensor(split_lstm.y_test, dtype=torch.float32)

    lstm = SimpleLSTM(input_dim=8, hidden_dim=32, num_layers=1)
    lstm_eval = evaluate_torch_model(
        model=lstm,
        train_ds=TensorDataset(XtrL, ytrL),
        val_ds=TensorDataset(XvaL, yvaL),
        test_ds=TensorDataset(XteL, yteL),
        batch_size=128,
    )

    # Bundle results
    results = {
        "rf": {
            "tau_star": rf_eval.tau_star,
            "val_auc": rf_eval.val_auc,
            "test_auc": rf_eval.test_auc,
            "test_points": rf_eval.test_points.__dict__,
        },
        "cnn": {
            "tau_star": cnn_eval.tau_star,
            "val_auc": cnn_eval.val_auc,
            "test_auc": cnn_eval.test_auc,
            "test_points": cnn_eval.test_points.__dict__,
        },
        "lstm": {
            "tau_star": lstm_eval.tau_star,
            "val_auc": lstm_eval.val_auc,
            "test_auc": lstm_eval.test_auc,
            "test_points": lstm_eval.test_points.__dict__,
        },
    }
    return results


if __name__ == "__main__":
    set_seed(42)
    # Running this script as-is will execute a quick demo using toy data
    # (random initializations; results are not meaningful). Replace the
    # toy data with your dataset and trained models to reproduce paper numbers.
    res = main_demo()
    # Pretty-print
    import json
    print(json.dumps(res, indent=2))
