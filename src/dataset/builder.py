# src/dataset/builder.py

import sys
from pathlib import Path as _Path
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import json
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from src.engine.orderbook import OrderBook, BookSnapshot
from src.features.features import FeatureEngine, LabelGenerator
from src.utils.config import (
    DATA_RAW, DATA_PROCESSED, DATA_SEQ,
    FEATURE_COLS, LABEL_COL,
    LOOKAHEAD_TICKS, PRICE_THRESHOLD,
    SEQUENCE_LEN,
    TRAIN_RATIO, VAL_RATIO,
    SYMBOL, MODELS_DIR,
)

logger = logging.getLogger(__name__)


# ── Step 1: Replay raw JSONL into BookSnapshots ────────────

class TickReplayer:
    """
    Replays Binance depthUpdate JSONL events into BookSnapshots.

    Strategy: build a synthetic order book by accumulating all
    price-level updates from the JSONL file itself — no REST
    snapshot needed. This works perfectly for offline replay
    because we have the full history of updates in sequence.
    """

    def __init__(self, symbol: str = SYMBOL):
        self.symbol = symbol.upper()

    def replay_file(self, path: Path) -> List[BookSnapshot]:
        if path.stat().st_size == 0:
            logger.warning(f"[Replayer] Skipping empty file: {path.name}")
            return []

        logger.info(f"[Replayer] Replaying {path.name} ...")

        # Build order book purely from accumulated diff events.
        # We treat the first event as a warm-up seed by applying
        # every update unconditionally, growing the book organically.
        bids: dict = {}   # price -> qty
        asks: dict = {}

        snaps    = []
        n_lines  = 0
        n_errors = 0

        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    n_errors += 1
                    continue

                # Validate it's a Binance depthUpdate event
                if not ("b" in msg and "a" in msg and "u" in msg):
                    continue

                n_lines += 1

                # Apply bid updates
                for price_str, qty_str in msg["b"]:
                    p = float(price_str)
                    q = float(qty_str)
                    if q == 0.0:
                        bids.pop(p, None)
                    else:
                        bids[p] = q

                # Apply ask updates
                for price_str, qty_str in msg["a"]:
                    p = float(price_str)
                    q = float(qty_str)
                    if q == 0.0:
                        asks.pop(p, None)
                    else:
                        asks[p] = q

                # Only emit snapshot once book has enough levels
                if len(bids) < 5 or len(asks) < 5:
                    continue

                # Build sorted top-20 levels
                top_bids = sorted(bids.items(), key=lambda x: -x[0])[:20]
                top_asks = sorted(asks.items(), key=lambda x:  x[0])[:20]

                # Timestamp: use _ts if present, else E (ms) / 1000
                ts = msg.get("_ts")
                if ts is None:
                    ts = msg.get("E", time.time() * 1000)
                    if ts > 1e12:
                        ts = ts / 1000.0

                snap = BookSnapshot(
                    bids        = top_bids,
                    asks        = top_asks,
                    timestamp   = ts,
                    sequence_id = msg["u"],
                )
                snaps.append(snap)

        logger.info(
            f"[Replayer] {path.name}: "
            f"{n_lines} valid events → {len(snaps)} snapshots "
            f"({n_errors} parse errors)"
        )
        return snaps

    def replay_all(self, raw_dir: Path = DATA_RAW) -> List[BookSnapshot]:
        files = sorted(raw_dir.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(
                f"No JSONL files found in {raw_dir}\n"
                f"Run: python main.py collect"
            )

        logger.info(f"[Replayer] Found {len(files)} JSONL file(s)")
        all_snaps = []
        for f in files:
            all_snaps.extend(self.replay_file(f))

        logger.info(f"[Replayer] Total snapshots: {len(all_snaps)}")

        if len(all_snaps) == 0:
            raise RuntimeError(
                "Replayer produced 0 snapshots.\n"
                "Run the inspector to check your JSONL:\n"
                "  python -c \"import json; f=open('data/raw/YOUR_FILE.jsonl'); "
                "print(list(json.loads(f.readline()).keys()))\""
            )
        return all_snaps


# ── Step 2: Snapshots → Labeled Feature DataFrame ─────────

def build_feature_dataframe(
    snapshots:  List[BookSnapshot],
    lookahead:  int   = LOOKAHEAD_TICKS,
    threshold:  float = PRICE_THRESHOLD,
    save_path:  Optional[Path] = None,
) -> pd.DataFrame:
    logger.info(f"[Builder] Extracting features from {len(snapshots)} snapshots ...")
    engine  = FeatureEngine()
    labeler = LabelGenerator(lookahead=lookahead, threshold=threshold)
    df      = labeler.generate_from_snapshots(snapshots, engine)

    if save_path is None:
        save_path = DATA_PROCESSED / f"features_{int(time.time())}.parquet"

    df.to_parquet(save_path, index=False)
    logger.info(f"[Builder] Saved → {save_path}  shape={df.shape}")
    return df


def load_processed_dataframes(processed_dir: Path = DATA_PROCESSED) -> pd.DataFrame:
    files = sorted(processed_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {processed_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"[Builder] Loaded {len(files)} parquet(s). shape={df.shape}")
    return df


# ── Step 3: Chronological split ───────────────────────────

def chronological_split(
    df:          pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n  = len(df)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    train = df.iloc[:t1].copy()
    val   = df.iloc[t1:t2].copy()
    test  = df.iloc[t2:].copy()
    logger.info(f"[Builder] Split → train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


# ── Step 4: Feature scaling ────────────────────────────────

def fit_scaler(
    train_df:     pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
    save_path:    Optional[Path] = None,
) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    if save_path is None:
        save_path = MODELS_DIR / "scaler.pkl"
    joblib.dump(scaler, save_path)
    logger.info(f"[Builder] Scaler saved → {save_path}")
    return scaler


def apply_scaler(
    df:           pd.DataFrame,
    scaler:       StandardScaler,
    feature_cols: List[str] = FEATURE_COLS,
) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


# ── Step 5: Sequence builder (LSTM) ───────────────────────

class SequenceBuilder:
    def __init__(
        self,
        seq_len:      int       = SEQUENCE_LEN,
        feature_cols: List[str] = FEATURE_COLS,
        label_col:    str       = LABEL_COL,
        stride:       int       = 1,
    ):
        self.seq_len      = seq_len
        self.feature_cols = feature_cols
        self.label_col    = label_col
        self.stride       = stride

    def build(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_vals = df[self.feature_cols].values.astype(np.float32)
        y_vals = df[self.label_col].values.astype(np.int64)
        Xs, ys = [], []
        for i in range(0, len(df) - self.seq_len, self.stride):
            Xs.append(X_vals[i : i + self.seq_len])
            ys.append(y_vals[i + self.seq_len - 1])
        X = np.stack(Xs)
        y = np.array(ys)
        logger.info(f"[SequenceBuilder] X={X.shape}  y={y.shape}")
        return X, y

    def save(self, X, y, name, seq_dir=DATA_SEQ):
        np.save(seq_dir / f"{name}_X.npy", X)
        np.save(seq_dir / f"{name}_y.npy", y)
        logger.info(f"[SequenceBuilder] Saved {name} → {seq_dir}")

    @staticmethod
    def load(name, seq_dir=DATA_SEQ):
        return (np.load(seq_dir / f"{name}_X.npy"),
                np.load(seq_dir / f"{name}_y.npy"))


# ── Full pipeline ──────────────────────────────────────────

def run_full_pipeline(
    symbol:    str   = SYMBOL,
    lookahead: int   = LOOKAHEAD_TICKS,
    threshold: float = PRICE_THRESHOLD,
    seq_len:   int   = SEQUENCE_LEN,
):
    replayer  = TickReplayer(symbol=symbol)
    snapshots = replayer.replay_all()

    df = build_feature_dataframe(snapshots, lookahead, threshold)

    train_df, val_df, test_df = chronological_split(df)

    scaler   = fit_scaler(train_df)
    train_df = apply_scaler(train_df, scaler)
    val_df   = apply_scaler(val_df,   scaler)
    test_df  = apply_scaler(test_df,  scaler)

    for split, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = DATA_PROCESSED / f"{split}.csv"
        sdf.to_csv(out, index=False)
        logger.info(f"[Pipeline] {split}.csv  rows={len(sdf)}")

    sb = SequenceBuilder(seq_len=seq_len)
    for split, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X, y = sb.build(sdf)
        sb.save(X, y, split)

    logger.info("[Pipeline] Complete.")
    return train_df, val_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_full_pipeline()