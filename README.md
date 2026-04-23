# OBI-Predictor
Order book imbalance predictor predict short term price moves using bid/as voulumne signal

<div align="center">

# OBI Predictor

**Order Book Imbalance Predictor for Short-Term Crypto Price Direction**



## What is OBI?

**Order Book Imbalance (OBI)** measures the balance of buying vs selling pressure in a limit order book:

```
OBI = (bid_volume − ask_volume) / (bid_volume + ask_volume)   ∈ [−1, +1]
```

- **+1** → all volume on the bid side → strong buy pressure → price likely to rise  
- **−1** → all volume on the ask side → strong sell pressure → price likely to fall  
- **0**  → balanced book → no directional signal

This project streams live Binance order book data at **100ms resolution**, computes OBI and 15 related features, and uses a trained machine learning model to predict short-term price direction: **BUY**, **SELL**, or **NEUTRAL**.

---

## Live Output Preview

```
────────────────────────────────────────────────────────────────────────────────
  OBI Predictor — LIVE   symbol=BTCUSDT
  Model: lgbm   Threshold: 0.38
────────────────────────────────────────────────────────────────────────────────

  [  14470]  mid=  76413.71  OBI=+0.2281  Δflow=+0.2344  signal=BUY   conf=72.4%
  [  14480]  mid=  76417.52  OBI=-0.5396  Δflow=-0.1461  signal=SELL  conf=68.1%
  [  14490]  mid=  76418.90  OBI=+0.0583  Δflow=-0.0341  signal=HOLD  conf=91.2%
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Binance WebSocket                        │
│              wss://stream.binance.com:9443/ws               │
│                    depth@100ms stream                       │
└────────────────────────┬────────────────────────────────────┘
                         │  differential depth events (U, u, b, a)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Order Book Engine                         │
│   SortedDict bid/ask · apply delta updates · sequence sync  │
└────────────────────────┬────────────────────────────────────┘
                         │  BookSnapshot (top-N levels)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Feature Engine                            │
│  OBI ×5 depths · delta · spread · VWAP · slope · vol flow  │
└────────────────────────┬────────────────────────────────────┘
                         │  FeatureVector (16 features)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   ML Predictor                              │
│         LightGBM  /  Logistic Regression  /  LSTM          │
└────────────────────────┬────────────────────────────────────┘
                         │  P(SELL), P(NEUTRAL), P(BUY)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Signal Output                             │
│        Confidence-gated:  BUY · SELL · NEUTRAL             │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
obi-predictor/
│
├── data/
│   ├── raw/                    ← JSONL tick files from live collector
│   ├── processed/              ← Parquet features + train/val/test CSVs
│   └── sequences/              ← numpy .npy sequences for LSTM
│
├── src/
│   ├── collector/
│   │   └── websocket_client.py ← Binance WS stream + raw JSONL writer
│   │
│   ├── engine/
│   │   └── orderbook.py        ← In-memory order book state machine
│   │
│   ├── features/
│   │   └── features.py         ← Feature extraction + label generation
│   │
│   ├── dataset/
│   │   └── builder.py          ← Offline pipeline: replay → features → splits
│   │
│   ├── models/
│   │   └── train.py            ← LightGBM / LogReg / LSTM trainers + eval
│   │
│   ├── backtest/
│   │   └── backtest.py         ← Event-driven backtester with PnL metrics
│   │
│   └── utils/
│       └── config.py           ← Central config (all parameters in one place)
│
├── notebooks/
│   └── analysis.ipynb          ← EDA: OBI plots, label dist, correlations
│
├── models/                     ← Saved .pkl models, scaler, plots
├── generate_synthetic_data.py  ← Bootstrap pipeline without live collection
├── requirements.txt
├── main.py                     ← CLI entry point
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Anaconda or virtualenv recommended
- Internet access to Binance (use mirror URL if geo-restricted — see below)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Bootstrap with synthetic data

No internet? Want to test the pipeline immediately? Generate 50,000 synthetic ticks, train, and backtest in ~60 seconds:

```bash
python generate_synthetic_data.py
```

Skip to step 5 after this.

### 3. Collect live order book data

```bash
python main.py collect
```

Streams `BTCUSDT` depth at 100ms and writes to `data/raw/*.jsonl`. Let it run for at least **2–4 hours** (24h+ recommended for better model quality). Stop with `Ctrl+C`.

### 4. Build the dataset

Replays raw ticks, extracts features, labels by forward price movement, and creates train/val/test splits:

```bash
python main.py build
```

### 5. Train the model

```bash
python main.py train
```

Trains LightGBM with early stopping. Saves model to `models/lgbm_model.pkl` and outputs a feature importance chart.

### 6. Backtest

Replays the test split through the trained model and prints performance metrics:

```bash
python main.py backtest
```

### 7. Live prediction

```bash
python main.py live
```

Connects to Binance WebSocket and prints real-time signals every 10 ticks.

---

## Configuration

All parameters are in `src/utils/config.py`. The most important ones:

### Data & Labeling

| Parameter | Default | Description |
|---|---|---|
| `SYMBOL` | `BTCUSDT` | Trading pair |
| `LOOKAHEAD_TICKS` | `5` | Ticks ahead for price direction label |
| `PRICE_THRESHOLD` | `0.00008` | Min % move to count as BUY/SELL (0.008%) |
| `SEQUENCE_LEN` | `50` | Sequence length for LSTM |

### Features

| Parameter | Default | Description |
|---|---|---|
| `TOP_LEVELS` | `5` | Order book levels used for volume features |
| `OBI_LEVELS` | `[1,3,5,10,20]` | Depths for multi-level OBI calculation |
| `ROLLING_WINDOW_SEC` | `5` | Window for OBI delta and volume flow |

### Model

| Parameter | Default | Description |
|---|---|---|
| `MODEL_TYPE` | `lgbm` | `lgbm` / `logreg` / `lstm` |
| `SIGNAL_THRESHOLD` | `0.38` | Min confidence to emit BUY/SELL (not HOLD) |

### Backtest

| Parameter | Default | Description |
|---|---|---|
| `INITIAL_CAPITAL` | `10,000` | Starting USD capital |
| `TRADE_SIZE_USD` | `100` | Per-trade position size |
| `STOP_LOSS_PCT` | `0.005` | 0.5% stop-loss per trade |
| `TAKE_PROFIT_PCT` | `0.010` | 1.0% take-profit per trade |
| `MAKER_FEE` | `0.0002` | Binance maker fee (0.02%) |
| `TAKER_FEE` | `0.0004` | Binance taker fee (0.04%) |

---

## Features Explained

| Feature | Formula | Meaning |
|---|---|---|
| `obi_1` | `(bid₁ − ask₁) / (bid₁ + ask₁)` | Best bid/ask imbalance |
| `obi_3/5/10/20` | Same at deeper levels | Multi-depth pressure |
| `obi_delta` | `obi_5(now) − obi_5(t−5s)` | Momentum of imbalance |
| `bid_vol` | `Σ qty` (top 5 bids) | Total buy-side depth |
| `ask_vol` | `Σ qty` (top 5 asks) | Total sell-side depth |
| `depth_ratio` | `bid_vol / ask_vol` | Relative depth ratio |
| `spread` | `best_ask − best_bid` | Absolute spread |
| `spread_pct` | `spread / mid_price` | Spread as % of price |
| `mid_vwap` | `(vwap_bid + vwap_ask) / 2` | Volume-weighted mid price |
| `bid_slope` | `ΔP / ΔVol` (bid side) | Book thinness on bids |
| `ask_slope` | `ΔP / ΔVol` (ask side) | Book thinness on asks |
| `vol_imbalance_flow` | `Σ(bid−ask) / Σtotal` over 5s | Sustained pressure |

---

## Models

### LightGBM (default, recommended)

Gradient boosted trees. Fast, interpretable, handles class imbalance natively. Early stopping on validation loss. Outputs `models/lgbm_feature_importance.png`.

```python
MODEL_TYPE = "lgbm"   # in config.py
```

### Logistic Regression (baseline)

Use this to verify your features actually carry signal. If logistic regression can't beat 33% accuracy on a balanced set, your features need work.

```python
MODEL_TYPE = "logreg"
```

### LSTM (sequence model, optional)

Looks at the last 50 feature vectors as a sequence. Requires PyTorch:

```bash
pip install torch
```

```python
MODEL_TYPE = "lstm"   # in config.py
```

---

## Backtest Metrics

After `python main.py backtest`, you'll see:

```
═════════════════════════════════════════════
  BACKTEST RESULTS
═════════════════════════════════════════════
  total_trades          1,842
  win_rate_%            54.21
  total_return_%        3.84
  sharpe_ratio          1.23
  max_drawdown_%        -2.41
  final_capital_usd     10,384.00
  total_fees_usd        147.36
  net_pnl_usd           384.00
═════════════════════════════════════════════
```

Output files saved to `models/`:
- `equity_curve.png` — portfolio value over time
- `pnl_distribution.png` — histogram of trade returns
- `trade_log.csv` — every trade with entry/exit/reason

---

## Troubleshooting

### `getaddrinfo failed` — cannot connect to Binance

You are likely in a geo-restricted region (e.g. India). Switch to Binance's public mirror in `config.py`:

```python
WS_BASE   = "wss://data-stream.binance.vision/ws"
REST_BASE = "https://data-api.binance.vision/api/v3"
```

### Model always predicts NEUTRAL

Label distribution is too skewed. Lower `PRICE_THRESHOLD` in `config.py`:

```python
PRICE_THRESHOLD  = 0.00008   # 0.008% — much more sensitive
SIGNAL_THRESHOLD = 0.38      # lower confidence bar
```

Then wipe processed data and rebuild:

```powershell
Remove-Item data\processed\*.csv
Remove-Item data\processed\*.parquet
Remove-Item data\sequences\*.npy
python main.py build
python main.py train
```

### `ModuleNotFoundError: No module named 'src'`

Always run from the project root directory:

```bash
cd C:\Users\YOU\Desktop\Quant\obi-predictor
python main.py live     # correct
```

Never run `src/` files directly.

### `FileNotFoundError: train.csv`

You must run in order — `collect` → `build` → `train` → `backtest` → `live`. Or skip to a working model immediately:

```bash
python generate_synthetic_data.py
```

### sklearn `UserWarning: X does not have valid feature names`

Harmless warning. Fix by passing a named DataFrame instead of numpy array to the scaler in `main.py`. Does not affect predictions.

---

## Data Collection Tips

| Duration | Approx rows | Quality |
|---|---|---|
| 1 hour | ~36,000 | Enough to test pipeline |
| 4 hours | ~144,000 | Decent for initial model |
| 24 hours | ~864,000 | Good — covers day/night sessions |
| 1 week | ~6,000,000 | Strong generalization |

The model needs to see **multiple market regimes** (trending, ranging, high/low volatility) to generalise. More sessions = better signal quality.

---

## Important Disclaimers

- **Research only.** This project is for educational and research purposes.
- **Not financial advice.** Do not deploy real capital based on these signals without extensive independent validation.
- **No lookahead bias.** Labels use strictly forward prices. Splits are chronological, never shuffled.
- **Past performance** of the backtester does not guarantee future results.
- **Binance ToS.** Ensure your usage of the Binance API complies with their terms of service in your jurisdiction.

---


