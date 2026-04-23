# main.py
# ─────────────────────────────────────────────────────────
# Entry point for the OBI Predictor.
#
# Modes (pass as first CLI argument):
#   collect   — stream raw ticks to data/raw/
#   build     — build features + dataset from raw ticks
#   train     — train model from processed dataset
#   backtest  — run backtester on test split
#   live      — live prediction loop (default)
#
# Usage examples:
#   python main.py collect
#   python main.py build
#   python main.py train
#   python main.py backtest
#   python main.py live
# ─────────────────────────────────────────────────────────

import asyncio
import logging
import sys
import time
from pathlib import Path
import numpy as np 
import joblib
import pandas as pd
from src.utils.config import (
    SYMBOL, FEATURE_COLS, LABEL_MAP, MODEL_TYPE,
    MODELS_DIR, SIGNAL_THRESHOLD,
)

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("main")


# ── Live prediction loop ──────────────────────────────────

class LivePredictor:
    """
    Wires together:
      BinanceDepthStream  →  OrderBook  →  FeatureEngine  →  Model  →  Signal
    """

    def __init__(
        self,
        symbol:     str   = SYMBOL,
        model_type: str   = MODEL_TYPE,
        threshold:  float = SIGNAL_THRESHOLD,
    ):
        from src.engine.orderbook import OrderBook
        from src.features.features import FeatureEngine

        self.symbol    = symbol.upper()
        self.threshold = threshold

        # Load trained model
        model_path = MODELS_DIR / f"{model_type}_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run: python main.py train"
            )
        self.model = joblib.load(model_path)
        logger.info(f"[Live] Loaded model from {model_path}")

        # Load scaler
        scaler_path = MODELS_DIR / "scaler.pkl"
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        if self.scaler:
            logger.info(f"[Live] Loaded scaler from {scaler_path}")

        self.book   = OrderBook(symbol=symbol, auto_seed=True)
        self.engine = FeatureEngine()
        self._tick  = 0

    async def _on_update(self, msg: dict):
        applied = self.book.apply_update(msg)
        if not applied:
            return

        snap = self.book.snapshot(timestamp=time.time())
        fv   = self.engine.compute(snap)
        row_df = pd.DataFrame([fv.to_dict()])[FEATURE_COLS]

        if self.scaler is not None:
            row = self.scaler.transform(row_df)
        else:
            row = row_df.values

        proba = self.model.predict_proba(row)[0]
        pred  = int(proba.argmax())
        conf  = float(proba.max())

        signal = LABEL_MAP[pred]
        self._tick += 1

        # Print every 10 ticks to avoid flooding the console
        if self._tick % 10 == 0:
            bar    = self._sparkbar(proba)
            action = f"\033[92mBUY \033[0m" if signal == "BUY" else \
                     f"\033[91mSELL\033[0m" if signal == "SELL" else \
                     "\033[93mHOLD\033[0m"

            print(
                f"  [{self._tick:>7}]  "
                f"mid={snap.mid_price:>10.2f}  "
                f"OBI={fv.obi_5:>+.4f}  "
                f"Δflow={fv.vol_imbalance_flow:>+.4f}  "
                f"signal={action}  conf={conf:.2%}  {bar}"
            )

    @staticmethod
    def _sparkbar(proba: "np.ndarray") -> str:
        labels = ["S", "N", "B"]
        parts  = []
        for l, p in zip(labels, proba):
            filled = round(p * 10)
            parts.append(f"{l}[{'█'*filled}{'░'*(10-filled)}]{p:.2f}")
        return "  ".join(parts)

    async def run(self):
        from src.collector.websocket_client import BinanceDepthStream
        logger.info(f"[Live] Starting live predictor for {self.symbol} …")
        print(f"\n{'─'*80}")
        print(f"  OBI Predictor — LIVE   symbol={self.symbol}")
        print(f"  Model: {MODEL_TYPE}   Threshold: {self.threshold}")
        print(f"{'─'*80}\n")

        stream = BinanceDepthStream(
            symbol    = self.symbol,
            on_update = self._on_update,
        )
        await stream.run_async()


# ── CLI dispatcher ────────────────────────────────────────

def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "live"

    if mode == "collect":
        logger.info("[main] Mode: collect")
        from src.collector.websocket_client import RawTickCollector
        RawTickCollector(symbol=SYMBOL).run()

    elif mode == "build":
        logger.info("[main] Mode: build dataset")
        from src.dataset.builder import run_full_pipeline
        run_full_pipeline()

    elif mode == "train":
        logger.info("[main] Mode: train")
        from src.models.train import run_training
        run_training(model_type=MODEL_TYPE)

    elif mode == "backtest":
        logger.info("[main] Mode: backtest")
        from src.backtest.backtest import run_backtest
        run_backtest(model_type=MODEL_TYPE)

    elif mode == "live":
        logger.info("[main] Mode: live")
        predictor = LivePredictor()
        asyncio.run(predictor.run())

    else:
        print(f"Unknown mode: '{mode}'")
        print("Valid modes: collect | build | train | backtest | live")
        sys.exit(1)


if __name__ == "__main__":
    main()
    
#collect  →  build  →  train  →  backtest  →  live