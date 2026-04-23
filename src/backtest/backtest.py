# src/backtest/backtest.py
# ─────────────────────────────────────────────────────────
# Event-driven backtester:
#   Replays test-set feature rows through the trained model,
#   simulates trades with fees, stop-loss and take-profit,
#   and reports PnL, Sharpe, max-drawdown, win-rate.
# ─────────────────────────────────────────────────────────

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import (
    DATA_PROCESSED, MODELS_DIR,
    FEATURE_COLS, LABEL_COL, LABEL_MAP,
    INITIAL_CAPITAL, TRADE_SIZE_USD,
    MAKER_FEE, TAKER_FEE,
    SIGNAL_THRESHOLD,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    MODEL_TYPE,
)

logger = logging.getLogger(__name__)


# ── Trade record ───────────────────────────────────────────

@dataclass
class Trade:
    entry_idx:    int
    entry_price:  float
    direction:    int       # +1 BUY, -1 SELL
    size_usd:     float
    exit_idx:     Optional[int]   = None
    exit_price:   Optional[float] = None
    exit_reason:  str             = ""   # "signal" | "stop" | "tp" | "eod"
    pnl_usd:      float           = 0.0
    fee_usd:      float           = 0.0

    @property
    def net_pnl(self) -> float:
        return self.pnl_usd - self.fee_usd

    @property
    def return_pct(self) -> float:
        return self.net_pnl / self.size_usd if self.size_usd > 0 else 0.0


# ── Portfolio state ────────────────────────────────────────

@dataclass
class Portfolio:
    cash:        float = INITIAL_CAPITAL
    position:    int   = 0            # +1 long, -1 short, 0 flat
    entry_price: float = 0.0
    entry_idx:   int   = -1
    trade_size:  float = TRADE_SIZE_USD
    trades:      List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    def open_position(self, direction: int, price: float, idx: int):
        fee = self.trade_size * TAKER_FEE
        self.cash     -= fee
        self.position  = direction
        self.entry_price = price
        self.entry_idx   = idx

    def close_position(self, price: float, idx: int, reason: str) -> Trade:
        direction  = self.position
        pnl        = direction * (price - self.entry_price) / self.entry_price * self.trade_size
        fee        = self.trade_size * TAKER_FEE
        self.cash += pnl - fee
        t = Trade(
            entry_idx   = self.entry_idx,
            entry_price = self.entry_price,
            direction   = direction,
            size_usd    = self.trade_size,
            exit_idx    = idx,
            exit_price  = price,
            exit_reason = reason,
            pnl_usd     = pnl,
            fee_usd     = fee * 2,   # entry + exit
        )
        self.trades.append(t)
        self.position    = 0
        self.entry_price = 0.0
        self.entry_idx   = -1
        return t

    def mark_equity(self, price: float):
        """Record equity including unrealised PnL."""
        if self.position != 0:
            unreal = self.position * (price - self.entry_price) / self.entry_price * self.trade_size
        else:
            unreal = 0.0
        self.equity_curve.append(self.cash + unreal)


# ── Backtester ─────────────────────────────────────────────

class Backtester:
    def __init__(
        self,
        model,
        threshold:     float = SIGNAL_THRESHOLD,
        stop_loss:     float = STOP_LOSS_PCT,
        take_profit:   float = TAKE_PROFIT_PCT,
        trade_size:    float = TRADE_SIZE_USD,
        initial_capital: float = INITIAL_CAPITAL,
    ):
        self.model        = model
        self.threshold    = threshold
        self.stop_loss    = stop_loss
        self.take_profit  = take_profit
        self.trade_size   = trade_size
        self.portfolio    = Portfolio(
            cash=initial_capital, trade_size=trade_size
        )

    def run(self, df: pd.DataFrame) -> "BacktestResult":
        """
        df: test DataFrame with FEATURE_COLS + 'mid_price' + 'timestamp'
        """
        X     = df[FEATURE_COLS].values
        proba = self.model.predict_proba(X)
        preds = proba.argmax(axis=1)          # 0=SELL,1=NEUTRAL,2=BUY
        confs = proba.max(axis=1)

        prices = df["mid_price"].values
        port   = self.portfolio

        for i, (price, pred, conf) in enumerate(zip(prices, preds, confs)):
            # 1. Check stop-loss / take-profit on open position
            if port.position != 0:
                move = port.position * (price - port.entry_price) / port.entry_price
                if move <= -self.stop_loss:
                    port.close_position(price, i, "stop")
                    port.mark_equity(price)
                    continue
                if move >= self.take_profit:
                    port.close_position(price, i, "tp")
                    port.mark_equity(price)
                    continue

            # 2. Translate prediction → direction
            if conf < self.threshold or pred == 1:
                direction = 0   # NEUTRAL / low-confidence → flat
            elif pred == 2:
                direction = +1  # BUY
            else:
                direction = -1  # SELL

            # 3. Signal change logic
            if port.position == 0 and direction != 0:
                port.open_position(direction, price, i)
            elif port.position != 0 and direction != port.position and direction != 0:
                # Reverse position
                port.close_position(price, i, "signal")
                port.open_position(direction, price, i)
            elif port.position != 0 and direction == 0:
                port.close_position(price, i, "signal")

            port.mark_equity(price)

        # Close any open position at end of data
        if port.position != 0:
            port.close_position(prices[-1], len(prices)-1, "eod")

        return BacktestResult(port, df["timestamp"].values if "timestamp" in df else None)


# ── Result & Metrics ───────────────────────────────────────

class BacktestResult:
    def __init__(self, portfolio: Portfolio, timestamps: Optional[np.ndarray] = None):
        self.portfolio  = portfolio
        self.timestamps = timestamps
        self.trades_df  = self._make_trades_df()
        self.equity     = np.array(portfolio.equity_curve)

    def _make_trades_df(self) -> pd.DataFrame:
        if not self.portfolio.trades:
            return pd.DataFrame()
        rows = []
        for t in self.portfolio.trades:
            rows.append({
                "entry_idx":   t.entry_idx,
                "exit_idx":    t.exit_idx,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "direction":   "BUY" if t.direction == 1 else "SELL",
                "pnl_usd":     round(t.pnl_usd, 4),
                "fee_usd":     round(t.fee_usd, 4),
                "net_pnl":     round(t.net_pnl, 4),
                "exit_reason": t.exit_reason,
            })
        return pd.DataFrame(rows)

    # ── Summary metrics ────────────────────────────────────

    @property
    def total_return_pct(self) -> float:
        if len(self.equity) == 0:
            return 0.0
        return (self.equity[-1] / self.equity[0] - 1) * 100

    @property
    def sharpe(self) -> float:
        if len(self.equity) < 2:
            return 0.0
        rets = np.diff(self.equity) / self.equity[:-1]
        std  = rets.std()
        if std == 0:
            return 0.0
        return (rets.mean() / std) * np.sqrt(252 * 24 * 60 * 10)  # annualised at 100ms ticks

    @property
    def max_drawdown_pct(self) -> float:
        if len(self.equity) == 0:
            return 0.0
        peak = np.maximum.accumulate(self.equity)
        dd   = (self.equity - peak) / peak
        return dd.min() * 100

    @property
    def win_rate(self) -> float:
        if self.trades_df.empty:
            return 0.0
        return (self.trades_df["net_pnl"] > 0).mean()

    @property
    def total_trades(self) -> int:
        return len(self.portfolio.trades)

    def summary(self) -> dict:
        return {
            "total_trades":      self.total_trades,
            "win_rate_%":        round(self.win_rate * 100, 2),
            "total_return_%":    round(self.total_return_pct, 4),
            "sharpe_ratio":      round(self.sharpe, 3),
            "max_drawdown_%":    round(self.max_drawdown_pct, 4),
            "final_capital_usd": round(self.portfolio.cash, 2),
            "total_fees_usd":    round(self.trades_df["fee_usd"].sum(), 2) if not self.trades_df.empty else 0.0,
            "net_pnl_usd":       round(self.trades_df["net_pnl"].sum(), 2) if not self.trades_df.empty else 0.0,
        }

    def print_summary(self):
        s = self.summary()
        print("\n" + "═" * 45)
        print("  BACKTEST RESULTS")
        print("═" * 45)
        for k, v in s.items():
            print(f"  {k:<22} {v}")
        print("═" * 45 + "\n")

    # ── Plots ──────────────────────────────────────────────

    def plot_equity(self, save_path: Optional[Path] = None):
        if len(self.equity) == 0:
            return
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.equity, lw=1, color="royalblue", label="Equity")
        ax.axhline(self.equity[0], color="gray", ls="--", lw=0.8, label="Initial")
        ax.set_title("Equity Curve"); ax.set_ylabel("USD"); ax.set_xlabel("Tick")
        ax.legend(); fig.tight_layout()
        out = save_path or (MODELS_DIR / "equity_curve.png")
        fig.savefig(out, dpi=120); plt.close(fig)
        logger.info(f"[Backtest] Equity curve → {out}")

    def plot_pnl_distribution(self, save_path: Optional[Path] = None):
        if self.trades_df.empty:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        self.trades_df["net_pnl"].hist(bins=30, ax=ax, color="steelblue", edgecolor="white")
        ax.axvline(0, color="red", ls="--", lw=1)
        ax.set_title("Trade PnL Distribution"); ax.set_xlabel("Net PnL (USD)")
        fig.tight_layout()
        out = save_path or (MODELS_DIR / "pnl_distribution.png")
        fig.savefig(out, dpi=120); plt.close(fig)
        logger.info(f"[Backtest] PnL distribution → {out}")


# ── CLI entry point ────────────────────────────────────────

def run_backtest(model_type: str = MODEL_TYPE):
    logging.basicConfig(level=logging.INFO)

    # Load model
    if model_type == "lgbm":
        model = joblib.load(MODELS_DIR / "lgbm_model.pkl")
    elif model_type == "logreg":
        model = joblib.load(MODELS_DIR / "logreg_model.pkl")
    else:
        raise ValueError(f"Backtest does not support model_type='{model_type}' directly. "
                         "Wrap LSTM predictions in a sklearn-compatible wrapper.")

    # Load test data
    test_path = DATA_PROCESSED / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}. Run the dataset builder first.")
    df = pd.read_csv(test_path)

    bt     = Backtester(model)
    result = bt.run(df)
    result.print_summary()
    result.plot_equity()
    result.plot_pnl_distribution()

    # Save trade log
    trade_log = MODELS_DIR / "trade_log.csv"
    result.trades_df.to_csv(trade_log, index=False)
    logger.info(f"[Backtest] Trade log → {trade_log}")
    return result


if __name__ == "__main__":
    run_backtest()