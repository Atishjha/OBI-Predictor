# src/engine/orderbook.py
# ─────────────────────────────────────────────────────────
# In-memory Order Book that maintains a sorted bid/ask map
# and applies Binance differential depth update events.
# ─────────────────────────────────────────────────────────

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from sortedcontainers import SortedDict

from src.utils.config import DEPTH_LEVELS, TOP_LEVELS

logger = logging.getLogger(__name__)

PriceLevel = Tuple[float, float]   # (price, quantity)


@dataclass
class BookSnapshot:
    """Immutable snapshot of the top-N order book levels."""
    bids:         List[PriceLevel]
    asks:         List[PriceLevel]
    timestamp:    float            # Unix timestamp (seconds)
    sequence_id:  int              # lastUpdateId from Binance

    # ── Derived helpers ─────────────────────────────────

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_pct(self) -> float:
        mid = self.mid_price
        return self.spread / mid if mid > 0 else 0.0

    def bid_volume(self, levels: int = TOP_LEVELS) -> float:
        return sum(q for _, q in self.bids[:levels])

    def ask_volume(self, levels: int = TOP_LEVELS) -> float:
        return sum(q for _, q in self.asks[:levels])

    def obi(self, levels: int = TOP_LEVELS) -> float:
        """
        Order Book Imbalance in [-1, +1].
          +1 = all volume on bid side  (buy pressure)
          -1 = all volume on ask side  (sell pressure)
        """
        bv = self.bid_volume(levels)
        av = self.ask_volume(levels)
        total = bv + av
        return (bv - av) / total if total > 0 else 0.0

    def vwap_bid(self, levels: int = TOP_LEVELS) -> float:
        vol = self.bid_volume(levels)
        if vol == 0:
            return self.best_bid
        return sum(p * q for p, q in self.bids[:levels]) / vol

    def vwap_ask(self, levels: int = TOP_LEVELS) -> float:
        vol = self.ask_volume(levels)
        if vol == 0:
            return self.best_ask
        return sum(p * q for p, q in self.asks[:levels]) / vol

    def mid_vwap(self, levels: int = TOP_LEVELS) -> float:
        return (self.vwap_bid(levels) + self.vwap_ask(levels)) / 2.0

    def depth_ratio(self, levels: int = TOP_LEVELS) -> float:
        """bid_vol / ask_vol. >1 = more bids than asks."""
        av = self.ask_volume(levels)
        return self.bid_volume(levels) / av if av > 0 else 1.0

    def bid_slope(self, levels: int = 5) -> float:
        """
        Weighted average price distance per unit of volume on the bid side.
        High slope → thin book; low slope → deep book.
        """
        if len(self.bids) < 2:
            return 0.0
        levels = min(levels, len(self.bids))
        prices  = [p for p, _ in self.bids[:levels]]
        weights = [q for _, q in self.bids[:levels]]
        total_w = sum(weights)
        if total_w == 0 or prices[0] == 0:
            return 0.0
        spread_range = prices[0] - prices[-1]
        return (spread_range / prices[0]) / (total_w / levels)

    def ask_slope(self, levels: int = 5) -> float:
        """Equivalent slope metric for the ask side."""
        if len(self.asks) < 2:
            return 0.0
        levels = min(levels, len(self.asks))
        prices  = [p for p, _ in self.asks[:levels]]
        weights = [q for _, q in self.asks[:levels]]
        total_w = sum(weights)
        if total_w == 0 or prices[0] == 0:
            return 0.0
        spread_range = prices[-1] - prices[0]
        return (spread_range / prices[0]) / (total_w / levels)


class OrderBook:
    """
    Maintains a live sorted order book for a single symbol.

    Usage:
        book = OrderBook("BTCUSDT")
        # Seed is called automatically in __init__.
        # Apply WS diff events:
        book.apply_update(ws_event_dict)
        # Get a snapshot:
        snap = book.snapshot()
    """

    def __init__(
        self,
        symbol:       str  = "BTCUSDT",
        depth_levels: int  = DEPTH_LEVELS,
        top_levels:   int  = TOP_LEVELS,
        auto_seed:    bool = False,
    ):
        self.symbol       = symbol.upper()
        self.depth_levels = depth_levels
        self.top_levels   = top_levels

        # SortedDict keyed by price (negated for bids so highest is first)
        self._bids: SortedDict = SortedDict(lambda x: -x)
        self._asks: SortedDict = SortedDict()

        self.last_update_id: int  = 0
        self._update_count:  int  = 0
        self._drop_count:    int  = 0

        if auto_seed:
            self._seed()

    # ── Seeding ────────────────────────────────────────────

    def _seed(self):
        """Fetch REST snapshot to initialise the book."""
        import requests
        from src.utils.config import REST_BASE, REST_SNAPSHOT_LIMIT
        url    = f"{REST_BASE}/depth"
        params = {"symbol": self.symbol, "limit": REST_SNAPSHOT_LIMIT}
        logger.info(f"[OrderBook] Fetching REST snapshot for {self.symbol} …")
        resp   = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data   = resp.json()
        self.apply_snapshot(data)

    def apply_snapshot(self, data: dict):
        """
        Apply a REST /depth response to fully reset the book.
        data = {"lastUpdateId": N, "bids": [[price, qty], …], "asks": [[price, qty], …]}
        """
        self._bids.clear()
        self._asks.clear()
        for price, qty in data["bids"]:
            p, q = float(price), float(qty)
            if q > 0:
                self._bids[p] = q
        for price, qty in data["asks"]:
            p, q = float(price), float(qty)
            if q > 0:
                self._asks[p] = q
        self.last_update_id = data["lastUpdateId"]
        logger.info(
            f"[OrderBook] Snapshot applied. "
            f"lastUpdateId={self.last_update_id} "
            f"bids={len(self._bids)} asks={len(self._asks)}"
        )

    # ── Delta updates ──────────────────────────────────────

    def apply_update(self, msg: dict) -> bool:
        """
        Apply a WS differential depth event.
        Returns True if applied, False if stale/skipped.

        msg keys: U (first update ID), u (last update ID),
                  b (bids list), a (asks list)
        """
        if msg["u"] <= self.last_update_id:
            self._drop_count += 1
            return False

        for price, qty in msg.get("b", []):
            self._apply_level(self._bids, float(price), float(qty))

        for price, qty in msg.get("a", []):
            self._apply_level(self._asks, float(price), float(qty))

        self.last_update_id = msg["u"]
        self._update_count  += 1
        return True

    @staticmethod
    def _apply_level(book_side: SortedDict, price: float, qty: float):
        """Set or remove a price level."""
        if qty == 0.0:
            book_side.pop(price, None)
        else:
            book_side[price] = qty

    # ── Snapshot ───────────────────────────────────────────

    def snapshot(
        self,
        top: Optional[int] = None,
        timestamp: float = 0.0,
    ) -> BookSnapshot:
        """
        Return an immutable BookSnapshot of the top-N levels.
        If top is None, uses self.top_levels.
        """
        import time
        n    = top or self.top_levels
        bids = list(self._bids.items())[:n]
        asks = list(self._asks.items())[:n]
        return BookSnapshot(
            bids        = bids,
            asks        = asks,
            timestamp   = timestamp or time.time(),
            sequence_id = self.last_update_id,
        )

    # ── Direct level access ────────────────────────────────

    def top_bids(self, n: int = TOP_LEVELS) -> List[PriceLevel]:
        return list(self._bids.items())[:n]

    def top_asks(self, n: int = TOP_LEVELS) -> List[PriceLevel]:
        return list(self._asks.items())[:n]

    def best_bid(self) -> Optional[float]:
        return next(iter(self._bids), None)

    def best_ask(self) -> Optional[float]:
        return next(iter(self._asks), None)

    def mid_price(self) -> float:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return 0.0
        return (bb + ba) / 2.0

    # ── Diagnostics ────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "symbol":         self.symbol,
            "last_update_id": self.last_update_id,
            "bid_levels":     len(self._bids),
            "ask_levels":     len(self._asks),
            "updates_applied":self._update_count,
            "updates_dropped":self._drop_count,
            "best_bid":       self.best_bid(),
            "best_ask":       self.best_ask(),
            "mid_price":      self.mid_price(),
        }

    def __repr__(self) -> str:
        return (
            f"<OrderBook {self.symbol} "
            f"bid={self.best_bid():.2f} "
            f"ask={self.best_ask():.2f} "
            f"upd={self._update_count}>"
        )