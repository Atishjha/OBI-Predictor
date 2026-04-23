# src/collector/websocket_client.py
# ─────────────────────────────────────────────────────────
# Connects to Binance partial/differential depth WebSocket,
# seeds the local order book from REST, then streams live
# delta updates into the processing pipeline.
# ─────────────────────────────────────────────────────────

import asyncio
import json
import logging
import time
from collections import deque
from typing import Callable, Awaitable, Optional

import websockets
import requests

from src.utils.config import (
    WS_BASE, REST_BASE, SYMBOL, DEPTH_LEVELS,
    WS_INTERVAL, REST_SNAPSHOT_LIMIT
)

logger = logging.getLogger(__name__)


class BinanceDepthStream:
    """
    Manages a Binance differential depth WebSocket stream.

    Protocol (per Binance docs):
      1. Open WS stream, buffer incoming events.
      2. Fetch REST /depth snapshot (lastUpdateId = U).
      3. Drop buffered events where event.u <= U.
      4. Find the first event where U <= lastUpdateId+1 <= u.
      5. Apply all subsequent events in order.
    """

    def __init__(
        self,
        symbol: str = SYMBOL,
        on_update: Optional[Callable[[dict], Awaitable[None]]] = None,
        depth_levels: int = DEPTH_LEVELS,
        interval: str = WS_INTERVAL,
    ):
        self.symbol       = symbol.lower()
        self.on_update    = on_update
        self.depth_levels = depth_levels
        self.interval     = interval
        self._buffer: deque = deque()
        self._synced: bool  = False
        self._last_update_id: int = 0
        self._running: bool = False
        self._reconnect_delay: float = 1.0
        self._max_reconnect_delay: float = 60.0

    # ── Public API ─────────────────────────────────────────

    def run(self):
        """Blocking entry-point. Run the stream forever."""
        asyncio.run(self._run_forever())

    async def run_async(self):
        """Async entry-point for use inside existing event loops."""
        await self._run_forever()

    # ── Internal ───────────────────────────────────────────

    async def _run_forever(self):
        self._running = True
        while self._running:
            try:
                await self._connect_and_stream()
                self._reconnect_delay = 1.0
            except (websockets.ConnectionClosed,
                    websockets.WebSocketException,
                    ConnectionResetError) as exc:
                logger.warning(f"[WS] Disconnected: {exc}. "
                               f"Reconnecting in {self._reconnect_delay}s …")
                self._synced = False
                self._buffer.clear()
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay
                )
            except Exception as exc:
                logger.error(f"[WS] Unexpected error: {exc}", exc_info=True)
                await asyncio.sleep(5)

    async def _connect_and_stream(self):
        url = f"{WS_BASE}/{self.symbol}@depth@{self.interval}"
        logger.info(f"[WS] Connecting to {url}")
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            logger.info("[WS] Connected. Seeding order book …")
            # Seed the order book in background while buffering events
            seed_task = asyncio.create_task(self._seed_from_rest())

            async for raw in ws:
                msg = json.loads(raw)

                if not self._synced:
                    self._buffer.append(msg)
                    # Check if seed task is done
                    if seed_task.done():
                        await self._flush_buffer()
                else:
                    await self._dispatch(msg)

    async def _seed_from_rest(self):
        """Fetch REST snapshot and set _last_update_id."""
        loop = asyncio.get_event_loop()
        snapshot = await loop.run_in_executor(None, self._fetch_rest_snapshot)
        self._last_update_id = snapshot["lastUpdateId"]
        logger.info(f"[WS] REST snapshot received. lastUpdateId={self._last_update_id}")
        # Signal that the seed snapshot is available
        # (buffer flush happens in _connect_and_stream)

    def _fetch_rest_snapshot(self) -> dict:
        url = f"{REST_BASE}/depth"
        params = {"symbol": self.symbol.upper(), "limit": REST_SNAPSHOT_LIMIT}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    async def _flush_buffer(self):
        """
        Apply buffered events following Binance sync protocol:
        Drop events where u <= lastUpdateId.
        Start from first event where U <= lastUpdateId+1 <= u.
        """
        logger.info(f"[WS] Flushing {len(self._buffer)} buffered events …")
        synced = False
        for msg in self._buffer:
            U = msg["U"]   # first update ID in event
            u = msg["u"]   # last  update ID in event
            if u <= self._last_update_id:
                continue   # stale
            if not synced:
                if U <= self._last_update_id + 1 <= u:
                    synced = True
                else:
                    logger.warning(f"[WS] Gap detected! Dropping event U={U} u={u}")
                    continue
            await self._dispatch(msg)

        self._buffer.clear()
        self._synced = True
        logger.info("[WS] Order book synced. Streaming live updates.")

    async def _dispatch(self, msg: dict):
        """Validate sequence and forward to callback."""
        if msg["u"] <= self._last_update_id:
            return  # duplicate / stale
        expected = self._last_update_id + 1
        if msg["U"] > expected:
            logger.warning(
                f"[WS] Sequence gap! Expected {expected}, got U={msg['U']}. "
                "Restarting stream …"
            )
            self._synced = False
            raise websockets.ConnectionClosed(None, None)
        self._last_update_id = msg["u"]
        if self.on_update:
            await self.on_update(msg)

    def stop(self):
        self._running = False


# ── Collector: save raw ticks to disk ─────────────────────

class RawTickCollector:
    """
    Collects raw depth diff events and writes them to JSONL
    files in data/raw/ for offline labeling and training.
    """

    def __init__(self, symbol: str = SYMBOL, flush_every: int = 1000):
        from src.utils.config import DATA_RAW
        self.symbol      = symbol.upper()
        self.flush_every = flush_every
        self._buf: list  = []
        self._file       = None
        self._path       = DATA_RAW / f"{self.symbol}_{int(time.time())}.jsonl"
        self._open_file()
        self.stream = BinanceDepthStream(
            symbol=symbol,
            on_update=self._handle,
        )

    def _open_file(self):
        self._file = open(self._path, "w", buffering=1)
        logger.info(f"[Collector] Writing to {self._path}")

    async def _handle(self, msg: dict):
        msg["_ts"] = time.time()
        self._buf.append(json.dumps(msg))
        if len(self._buf) >= self.flush_every:
            self._flush()

    def _flush(self):
        if self._file and self._buf:
            self._file.write("\n".join(self._buf) + "\n")
            self._buf.clear()

    def run(self):
        try:
            self.stream.run()
        finally:
            self._flush()
            if self._file:
                self._file.close()
            logger.info(f"[Collector] Closed {self._path}")