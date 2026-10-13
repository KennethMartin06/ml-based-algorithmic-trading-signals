"""LSTM sequence classifier (PyTorch) for next-period direction.

Pipeline
--------
1. Inputs are the same engineered features used by the tabular models, standardised with a
   ``StandardScaler`` fitted on training-period rows only.
2. For a decision row t the input window is rows ``t-L+1 .. t`` (L = ``seq_len``) - strictly past
   information. The window maps to the label of row t, i.e. the same target and dates as the
   tabular models, so all models are compared on identical test dates.
3. Network: LSTM -> dropout -> linear layer -> class logits; trained with (optionally
   class-weighted) cross-entropy and Adam. Early stopping monitors the UNWEIGHTED cross-entropy
   (= log loss, the selection metric used for every model) on a held-out *early-stopping slice*;
   the best epoch's weights are restored.
4. In the experiment the early-stopping slice is the final part of the TRAINING period (purged by
   the label horizon), so the validation period stays untouched by LSTM fitting - exactly as for
   the tabular models, whose hyperparameters are chosen by CV inside the training period.

Only the *order of mini-batches* of training windows is shuffled (``shuffle_train_batches``). Each
window is self-contained and all training windows precede the early-stopping slice, so this does
not leak future information; it only de-correlates consecutive gradient steps.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from .preprocessing import build_sequences

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except Exception:  # pragma: no cover - depends on environment
    torch = None
    nn = None
    TORCH_AVAILABLE = False
    TORCH_VERSION = None


if TORCH_AVAILABLE:

    class LSTMNet(nn.Module):
        def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float, n_classes: int):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size, n_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(self.dropout(out[:, -1, :]))


class LSTMSequenceClassifier:
    """Small, seeded LSTM classifier with early stopping and train-only normalisation."""

    def __init__(
        self,
        n_classes: int,
        seq_len: int = 20,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 60,
        patience: int = 8,
        class_weight: str | None = "balanced",
        shuffle_train_batches: bool = True,
        seed: int = 42,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for the LSTM model (pip install torch)")
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.class_weight = class_weight
        self.shuffle_train_batches = shuffle_train_batches
        self.seed = seed
        self.history: list[dict] = []
        self.best_epoch: int | None = None
        self.scaler: StandardScaler | None = None
        self.net = None

    # ------------------------------------------------------------------ helpers
    def get_params(self) -> dict:
        keys = ["seq_len", "hidden_size", "num_layers", "dropout", "learning_rate", "weight_decay",
                "batch_size", "max_epochs", "patience", "class_weight", "shuffle_train_batches", "seed"]
        return {k: getattr(self, k) for k in keys}

    def _windows(self, X_scaled: np.ndarray, positions: np.ndarray):
        return torch.tensor(build_sequences(X_scaled, positions, self.seq_len), dtype=torch.float32)

    def _class_weights(self, y: np.ndarray):
        if self.class_weight != "balanced":
            return None
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        counts[counts == 0] = 1.0
        w = len(y) / (self.n_classes * counts)
        return torch.tensor(w, dtype=torch.float32)

    # ------------------------------------------------------------------ API
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_pos: np.ndarray,
        val_pos: np.ndarray,
        scaler_rows: np.ndarray | None = None,
    ) -> "LSTMSequenceClassifier":
        """Fit on windows ending at ``train_pos``; early-stop on windows ending at ``val_pos``.

        ``X`` is the full chronological feature matrix (rows = dates); ``y`` holds labels aligned
        with X (NaN allowed where unused). ``scaler_rows`` are the row positions used to fit the
        scaler (default: every row up to the last training decision row).
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        train_pos = np.asarray(train_pos, dtype=int)
        val_pos = np.asarray(val_pos, dtype=int)
        train_pos = train_pos[train_pos >= self.seq_len - 1]
        if scaler_rows is None:
            scaler_rows = np.arange(0, train_pos.max() + 1)
        self.scaler = StandardScaler().fit(X[scaler_rows])
        Xs = self.scaler.transform(X).astype(np.float32)

        y_tr = np.asarray(y[train_pos], dtype=int)
        y_va = np.asarray(y[val_pos], dtype=int)
        Xtr, Xva = self._windows(Xs, train_pos), self._windows(Xs, val_pos)
        ytr_t, yva_t = torch.tensor(y_tr), torch.tensor(y_va)

        self.net = LSTMNet(X.shape[1], self.hidden_size, self.num_layers, self.dropout, self.n_classes)
        weights = self._class_weights(y_tr)
        loss_fn = nn.CrossEntropyLoss(weight=weights)  # training objective (weighted if 'balanced')
        eval_loss_fn = nn.CrossEntropyLoss()  # early stopping on plain log loss, the selection metric
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        gen = torch.Generator().manual_seed(self.seed)

        best_loss, best_state, bad_epochs = np.inf, None, 0
        self.history = []
        n = len(Xtr)
        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            order = torch.randperm(n, generator=gen) if self.shuffle_train_batches else torch.arange(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(Xtr[idx]), ytr_t[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx)
            self.net.eval()
            with torch.no_grad():
                logits = self.net(Xva)
                val_loss = eval_loss_fn(logits, yva_t).item()
                val_acc = (logits.argmax(1) == yva_t).float().mean().item()
            self.history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss, "val_accuracy": val_acc})
            if val_loss < best_loss - 1e-5:
                best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(self.net.state_dict()), 0
                self.best_epoch = epoch
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_loss = best_loss
        return self

    def predict_proba(self, X: np.ndarray, positions: np.ndarray) -> np.ndarray:
        if self.net is None or self.scaler is None:
            raise RuntimeError("Model is not fitted")
        Xs = self.scaler.transform(X).astype(np.float32)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(self._windows(Xs, np.asarray(positions, dtype=int)))
            # float64 like every other model, so prediction frames have one consistent dtype
            return torch.softmax(logits, dim=1).numpy().astype(np.float64)

    def save(self, directory: str | Path, stem: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), d / f"{stem}.pt")
        with open(d / f"{stem}_history.json", "w", encoding="utf-8") as fh:
            json.dump({"params": self.get_params(), "best_epoch": self.best_epoch, "history": self.history}, fh, indent=2)
