from pathlib import Path 
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
DATA_SEQ = BASE_DIR / "data" / "sequences"
for _p in (DATA_RAW,DATA_PROCESSED,DATA_SEQ,MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)
SYMBOL = "BTCUSDT"
WS_BASE = "wss://stream.binance.com:9443/ws"
REST_BASE = "https://api.binance.com/api/v3"
DEPTH_LEVELS = 20 
WS_INTERVAL = "100ms"
REST_SNAPSHOT_LIMIT = 1000 
TOP_LEVELS = 5 
ROLLING_WINDOW_SEC = 5 
OBI_LEVELS = [1,3,5,10,20]
TICK_INTERVAL_SEC = 0.1
LOOKAHEAD_TICKS = 10 
PRICE_THRESHOLD = 0.0002 
SEQUENCE_LEN = 50 
TRAIN_RATIO = 0.70 
VAL_RATIO = 0.15 
TEST_RATIO = 0.15
RANDOM_SEED = 42 
MODEL_TYPE = "lgbm"
LGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
} 
LOGREG_PARAMS = {
    "C":1.0,
    "max_iter":1000,
    "class_weight":"balanced",
    "random_state": RANDOM_SEED,
}
INITIAL_CAPITAL = 10_000.0
TRADE_SIZE_USD = 100.0 
MAKER_FEE = 0.0002
TAKER_FEE = 0.0004
SIGNAL_THRESHOLD = 0.55
STOP_LOSS_PCT = 0.005
TAKE_PROFIT_PCT = 0.010
LABEL_MAP = {0:"SELL",1:"NEUTRAL",2:"BUY"}
SIGNAL_MAP = {"SELL":-1,"NEUTRAL":0,"BUY":1}
FEATURE_COLS = [
    "obi_1",  "obi_3",  "obi_5",  "obi_10", "obi_20",
    "obi_delta",
    "bid_vol", "ask_vol",
    "spread",  "spread_pct",
    "depth_ratio",
    "bid_slope", "ask_slope",
    "mid_price", "mid_vwap",
    "vol_imbalance_flow",
]
 
LABEL_COL = "label"