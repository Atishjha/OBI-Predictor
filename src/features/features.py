# src/features/features.py
# ─────────────────────────────────────────────────────────
# Extracts a rich feature vector from a sequence of
# BookSnapshots produced by the order book engine.
# ─────────────────────────────────────────────────────────

import sys
from pathlib import Path as _Path
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Deque

import numpy as np

from src.utils.config import (
    OBI_LEVELS, ROLLING_WINDOW_SEC, TOP_LEVELS, FEATURE_COLS
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """One row of features extracted from the order book."""
    timestamp:           float
    mid_price:           float
    obi_1:               float
    obi_3:               float
    obi_5:               float
    obi_10:              float
    obi_20:              float
    obi_delta:           float
    bid_vol:             float
    ask_vol:             float
    depth_ratio:         float
    spread:              float
    spread_pct:          float
    mid_vwap:            float
    bid_slope:           float
    ask_slope:           float
    vol_imbalance_flow:  float

    def to_dict(self) -> Dict[str, float]:
        # Explicit dict — avoids any asdict/dataclass cache issues
        return {
            "timestamp":          self.timestamp,
            "mid_price":          self.mid_price,
            "obi_1":              self.obi_1,
            "obi_3":              self.obi_3,
            "obi_5":              self.obi_5,
            "obi_10":             self.obi_10,
            "obi_20":             self.obi_20,
            "obi_delta":          self.obi_delta,
            "bid_vol":            self.bid_vol,
            "ask_vol":            self.ask_vol,
            "depth_ratio":        self.depth_ratio,
            "spread":             self.spread,
            "spread_pct":         self.spread_pct,
            "mid_vwap":           self.mid_vwap,
            "bid_slope":          self.bid_slope,
            "ask_slope":          self.ask_slope,
            "vol_imbalance_flow": self.vol_imbalance_flow,
        }

    def to_array(self, cols: List[str] = FEATURE_COLS) -> np.ndarray:
        d = self.to_dict()
        return np.array([d[c] for c in cols], dtype=np.float32)


class FeatureEngine:
    """
    Stateful engine that maintains a rolling history of
    BookSnapshots and computes FeatureVectors on demand.
    """

    def __init__(
        self,
        obi_levels:         List[int] = OBI_LEVELS,
        rolling_window_sec: float     = ROLLING_WINDOW_SEC,
        top_levels:         int       = TOP_LEVELS,
    ):
        self.obi_levels         = obi_levels
        self.rolling_window_sec = rolling_window_sec
        self.top_levels         = top_levels
        self._history: Deque    = deque()

    def compute(self, snap) -> FeatureVector:
        """Compute a FeatureVector from a BookSnapshot."""
        ts = snap.timestamp or time.time()

        obis: Dict[int, float] = {}
        for lvl in self.obi_levels:
            obis[lvl] = snap.obi(min(lvl, len(snap.bids)))

        bid_vol = snap.bid_volume(self.top_levels)
        ask_vol = snap.ask_volume(self.top_levels)

        fv = FeatureVector(
            timestamp          = ts,
            mid_price          = snap.mid_price,
            obi_1              = obis.get(1,  0.0),
            obi_3              = obis.get(3,  0.0),
            obi_5              = obis.get(5,  0.0),
            obi_10             = obis.get(10, 0.0),
            obi_20             = obis.get(20, 0.0),
            obi_delta          = self._obi_delta(obis.get(5, 0.0), ts),
            bid_vol            = bid_vol,
            ask_vol            = ask_vol,
            depth_ratio        = snap.depth_ratio(self.top_levels),
            spread             = snap.spread,
            spread_pct         = snap.spread_pct,
            mid_vwap           = snap.mid_vwap(self.top_levels),
            bid_slope          = snap.bid_slope(self.top_levels),
            ask_slope          = snap.ask_slope(self.top_levels),
            vol_imbalance_flow = self._vol_imbalance_flow(bid_vol, ask_vol, ts),
        )

        self._history.append((ts, fv))
        self._prune(ts)
        return fv

    def _prune(self, now: float):
        cutoff = now - self.rolling_window_sec
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _obi_delta(self, current_obi: float, now: float) -> float:
        if not self._history:
            return 0.0
        _, oldest_fv = self._history[0]
        return current_obi - oldest_fv.obi_5

    def _vol_imbalance_flow(self, bid_vol: float, ask_vol: float, now: float) -> float:
        if not self._history:
            return 0.0
        bids_sum = sum(fv.bid_vol for _, fv in self._history) + bid_vol
        asks_sum = sum(fv.ask_vol for _, fv in self._history) + ask_vol
        total    = bids_sum + asks_sum
        return (bids_sum - asks_sum) / total if total > 0 else 0.0

    def reset(self):
        self._history.clear()


class LabelGenerator:
    """
    Adds forward-looking price-direction labels to a feature DataFrame.

    Labels:
      2  = BUY     (price rises  > threshold)
      0  = SELL    (price falls  > threshold)
      1  = NEUTRAL (within threshold)
    """

    def __init__(
        self,
        lookahead:  int   = 10,
        threshold:  float = 0.0002,
        mid_col:    str   = "mid_price",
        label_col:  str   = "label",
    ):
        self.lookahead  = lookahead
        self.threshold  = threshold
        self.mid_col    = mid_col
        self.label_col  = label_col

    def generate(self, df):
        import pandas as pd

        df = df.copy().reset_index(drop=True)

        if self.mid_col not in df.columns:
            raise KeyError(
                f"Column '{self.mid_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        df["_future_mid"] = df[self.mid_col].shift(-self.lookahead)
        df["_return"]     = (df["_future_mid"] - df[self.mid_col]) / df[self.mid_col]

        df[self.label_col] = 1
        df.loc[df["_return"] >  self.threshold, self.label_col] = 2
        df.loc[df["_return"] < -self.threshold, self.label_col] = 0

        df.drop(columns=["_future_mid", "_return"], inplace=True)
        df.dropna(subset=[self.label_col], inplace=True)
        df[self.label_col] = df[self.label_col].astype(int)
        df.reset_index(drop=True, inplace=True)

        dist = df[self.label_col].value_counts().sort_index()
        logger.info(
            f"[LabelGenerator] Label distribution: "
            f"SELL(0)={dist.get(0, 0)}  "
            f"NEUTRAL(1)={dist.get(1, 0)}  "
            f"BUY(2)={dist.get(2, 0)}"
        )
        return df

    def generate_from_snapshots(
        self,
        snapshots: List,
        engine:    FeatureEngine,
    ):
        import pandas as pd

        engine.reset()
        rows = []
        for snap in snapshots:
            fv  = engine.compute(snap)
            row = fv.to_dict()
            rows.append(row)

        df = pd.DataFrame(rows)

        logger.info(
            f"[LabelGenerator] DataFrame shape={df.shape}  "
            f"cols={list(df.columns)}"
        )

        if self.mid_col not in df.columns:
            raise KeyError(
                f"'{self.mid_col}' missing from DataFrame! "
                f"cols={list(df.columns)}"
            )

        return self.generate(df)