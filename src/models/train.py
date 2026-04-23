# src/models/train.py
# ─────────────────────────────────────────────────────────
# Trains and evaluates:
#   - LightGBM  (default, fast, interpretable)
#   - Logistic Regression (baseline)
#   - LSTM      (sequence model, optional)
# ─────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
import lightgbm as lgb

from src.utils.config import (
    DATA_PROCESSED, DATA_SEQ, MODELS_DIR,
    FEATURE_COLS, LABEL_COL,
    LGBM_PARAMS, LOGREG_PARAMS,
    MODEL_TYPE, LABEL_MAP, RANDOM_SEED, SEQUENCE_LEN,
)

logger = logging.getLogger(__name__)


# ── Data loading ───────────────────────────────────────────

def load_flat(split: str) -> Tuple[np.ndarray, np.ndarray]:
    path = DATA_PROCESSED / f"{split}.csv"
    df   = pd.read_csv(path)
    X    = df[FEATURE_COLS].values.astype(np.float32)
    y    = df[LABEL_COL].values.astype(np.int64)
    return X, y


def load_sequences(split: str) -> Tuple[np.ndarray, np.ndarray]:
    X = np.load(DATA_SEQ / f"{split}_X.npy")
    y = np.load(DATA_SEQ / f"{split}_y.npy")
    return X, y


# ── Evaluation helpers ─────────────────────────────────────

def evaluate(model, X_val, y_val, model_name: str, save_dir: Path = MODELS_DIR):
    preds = model.predict(X_val)
    proba = (
        model.predict_proba(X_val)
        if hasattr(model, "predict_proba")
        else None
    )

    acc = accuracy_score(y_val, preds)
    logger.info(f"\n{'='*50}")
    logger.info(f"[{model_name}] Accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(
        y_val, preds,
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    ))

    # Confusion matrix plot
    cm = confusion_matrix(y_val, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([LABEL_MAP[i] for i in range(3)])
    ax.set_yticklabels([LABEL_MAP[i] for i in range(3)])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{model_name} — Confusion Matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
    fig.colorbar(im)
    fig.tight_layout()
    out = save_dir / f"{model_name}_confusion.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"[{model_name}] Confusion matrix → {out}")

    # AUC (OvR)
    if proba is not None:
        try:
            auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro")
            logger.info(f"[{model_name}] Macro AUC (OvR): {auc:.4f}")
        except Exception:
            pass

    return acc


# ── LightGBM ──────────────────────────────────────────────

class LGBMTrainer:
    def __init__(self, params: dict = LGBM_PARAMS):
        self.params = params
        self.model: Optional[lgb.LGBMClassifier] = None

    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
    ) -> lgb.LGBMClassifier:
        logger.info("[LightGBM] Starting training …")
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set   = [(X_val, y_val)],
            callbacks  = [lgb.early_stopping(50, verbose=False),
                          lgb.log_evaluation(50)],
        )
        logger.info(f"[LightGBM] Best iteration: {self.model.best_iteration_}")
        return self.model

    def feature_importance_plot(self, save_dir: Path = MODELS_DIR):
        if self.model is None:
            return
        imp = pd.Series(
            self.model.feature_importances_,
            index=FEATURE_COLS
        ).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        imp.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title("LightGBM — Feature Importance (gain)")
        ax.set_xlabel("Feature"); ax.set_ylabel("Importance")
        fig.tight_layout()
        out = save_dir / "lgbm_feature_importance.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        logger.info(f"[LightGBM] Feature importance → {out}")

    def save(self, path: Optional[Path] = None):
        path = path or (MODELS_DIR / "lgbm_model.pkl")
        joblib.dump(self.model, path)
        logger.info(f"[LightGBM] Model saved → {path}")

    @staticmethod
    def load(path: Optional[Path] = None) -> lgb.LGBMClassifier:
        path = path or (MODELS_DIR / "lgbm_model.pkl")
        return joblib.load(path)


# ── Logistic Regression ────────────────────────────────────

class LogRegTrainer:
    def __init__(self, params: dict = LOGREG_PARAMS):
        self.params = params
        self.model: Optional[LogisticRegression] = None

    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
    ) -> LogisticRegression:
        logger.info("[LogReg] Training …")
        self.model = LogisticRegression(**self.params)
        self.model.fit(X_train, y_train)
        return self.model

    def save(self, path: Optional[Path] = None):
        path = path or (MODELS_DIR / "logreg_model.pkl")
        joblib.dump(self.model, path)
        logger.info(f"[LogReg] Model saved → {path}")

    @staticmethod
    def load(path: Optional[Path] = None) -> LogisticRegression:
        path = path or (MODELS_DIR / "logreg_model.pkl")
        return joblib.load(path)


# ── LSTM (PyTorch) ────────────────────────────────────────

class LSTMTrainer:
    """Optional PyTorch LSTM. Requires: pip install torch"""

    def __init__(
        self,
        input_size:  int = len(FEATURE_COLS),
        hidden_size: int = 64,
        num_layers:  int = 2,
        dropout:     float = 0.2,
        lr:          float = 1e-3,
        epochs:      int   = 20,
        batch_size:  int   = 512,
        device:      str   = "cpu",
    ):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.device      = device
        self.model       = None

    def _build_model(self):
        try:
            import torch
            import torch.nn as nn

            class _LSTM(nn.Module):
                def __init__(self, inp, hid, layers, drop):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        inp, hid, layers,
                        batch_first=True, dropout=drop
                    )
                    self.fc = nn.Linear(hid, 3)

                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])

            return _LSTM(
                self.input_size, self.hidden_size,
                self.num_layers, self.dropout
            ).to(self.device)
        except ImportError:
            raise ImportError("PyTorch not installed. Run: pip install torch")

    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
    ):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        self.model = self._build_model()
        opt    = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        crit   = nn.CrossEntropyLoss()

        def make_loader(X, y, shuffle):
            ds = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long)
            )
            return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

        train_loader = make_loader(X_train, y_train, shuffle=True)
        val_loader   = make_loader(X_val,   y_val,   shuffle=False)

        best_val_loss = float("inf")
        best_state    = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_loss += crit(self.model(xb), yb).item()

            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in self.model.state_dict().items()}

            if epoch % 5 == 0:
                logger.info(
                    f"[LSTM] Epoch {epoch:3d}/{self.epochs}  "
                    f"train_loss={total_loss/len(train_loader):.4f}  "
                    f"val_loss={val_loss:.4f}"
                )

        if best_state:
            self.model.load_state_dict(best_state)
        logger.info("[LSTM] Training complete.")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            xb   = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(xb)
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(self.device)
            return F.softmax(self.model(xb), dim=1).cpu().numpy()

    def save(self, path: Optional[Path] = None):
        import torch
        path = path or (MODELS_DIR / "lstm_model.pt")
        torch.save(self.model.state_dict(), path)
        logger.info(f"[LSTM] Model saved → {path}")


# ── Main training entry point ─────────────────────────────

def run_training(model_type: str = MODEL_TYPE):
    logging.basicConfig(level=logging.INFO)

    if model_type == "lstm":
        X_tr, y_tr = load_sequences("train")
        X_v,  y_v  = load_sequences("val")
        X_te, y_te = load_sequences("test")
        trainer = LSTMTrainer()
        trainer.train(X_tr, y_tr, X_v, y_v)
        # Wrap for evaluate()
        class _Wrapper:
            def predict(self, X): return trainer.predict(X)
            def predict_proba(self, X): return trainer.predict_proba(X)
        evaluate(_Wrapper(), X_te, y_te, "LSTM")
        trainer.save()
    else:
        X_tr, y_tr = load_flat("train")
        X_v,  y_v  = load_flat("val")
        X_te, y_te = load_flat("test")

        if model_type == "lgbm":
            trainer = LGBMTrainer()
            trainer.train(X_tr, y_tr, X_v, y_v)
            evaluate(trainer.model, X_te, y_te, "LightGBM")
            trainer.feature_importance_plot()
            trainer.save()
        elif model_type == "logreg":
            trainer = LogRegTrainer()
            trainer.train(X_tr, y_tr)
            evaluate(trainer.model, X_te, y_te, "LogisticRegression")
            trainer.save()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    run_training()