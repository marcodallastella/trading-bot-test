# Universe
UNIVERSE = [
    "AAPL", "ABNB", "AMD", "AMZN", "AVGO",
    "COIN", "CRWD", "DDOG", "DIS", "ENPH",
    "GOOGL", "JNJ", "JPM", "KO", "META",
    "MRVL", "MSFT", "NFLX", "NET", "NVDA",
    "PLTR", "SHOP", "SNOW", "TSLA", "UBER",
]  # 25 tickers, alphabetical
SPY_TICKER = "SPY"

# Signal / strategy
LOOKBACK_DAYS = 60
TREND_FILTER_SMA_PERIOD = 50
PIR_STANDARD_BUY_THRESHOLD = 0.30
PIR_DEEP_DIP_THRESHOLD = 0.15
PIR_VERY_DEEP_THRESHOLD = 0.10
PIR_STRONG_DIP_THRESHOLD = 0.20
DATA_QUALITY_MIN_BAR_RATIO = 0.85

# ATR / volatility
ATR_LOOKBACK = 14
ATR_LOW_THRESHOLD = 0.02
ATR_HIGH_THRESHOLD = 0.04
VOL_FACTOR_LOW = 1.2
VOL_FACTOR_NORMAL = 1.0
VOL_FACTOR_HIGH = 0.7

# Signal multipliers
SIGNAL_MULT_DEEP_CRASH = 1.5
SIGNAL_MULT_STRONG_DIP = 1.2
SIGNAL_MULT_STANDARD   = 1.0

# Regime detection (SPY-based)
REGIME_SMA_PERIOD = 200
REGIME_SLOPE_PERIOD = 20
REGIME_SLOPE_RISING_THRESHOLD  =  0.005
REGIME_SLOPE_FALLING_THRESHOLD = -0.005
REGIME_MULT_BULL       = 1.00
REGIME_MULT_LATE_CYCLE = 0.85
REGIME_MULT_TOPPING    = 0.60
REGIME_MULT_RECOVERY   = 0.60
REGIME_MULT_BEAR       = 0.30

# Kelly sizing
BASE_KELLY_FRACTION     = 0.08
BASE_KELLY_FRACTION_MIN = 0.04
BASE_KELLY_FRACTION_MAX = 0.12

# Position / exposure limits
MAX_PER_TICKER_EXPOSURE_PCT = 0.15
MAX_CONCURRENT_POSITIONS    = 5
MIN_ORDER_NOTIONAL          = 1.00
PER_TICKER_COOLDOWN_DAYS    = 3
MAX_TOTAL_DEPLOYMENT_PCT    = 0.90

# Stop-loss and trailing stop
STOP_LOSS_PCT                = -0.10
TRAILING_STOP_ACTIVATION_PCT =  0.20
TRAILING_STOP_TRAIL_PCT      =  0.10

# Spread guards (basis points)
MAX_SPREAD_BPS_BUY  = 20
MAX_SPREAD_BPS_SELL = 30

# Drawdown and risk controls
DRAWDOWN_HALT_THRESHOLD   = 0.25
DRAWDOWN_RESUME_THRESHOLD = 0.15
ROLLING_PEAK_WINDOW_DAYS  = 365
WASH_SALE_DAYS            = 31

# Equity growth tiers
EQUITY_TIERS: list[dict] = [
    {"min": 0,     "max": 500,          "per_ticker_cap": 0.15, "max_positions": 5,  "kelly_min": 0.08, "kelly_max": 0.12, "order_type": "fractional_market"},
    {"min": 500,   "max": 2500,         "per_ticker_cap": 0.10, "max_positions": 6,  "kelly_min": 0.06, "kelly_max": 0.10, "order_type": "fractional_market"},
    {"min": 2500,  "max": 10000,        "per_ticker_cap": 0.07, "max_positions": 8,  "kelly_min": 0.05, "kelly_max": 0.08, "order_type": "limit_ioc"},
    {"min": 10000, "max": float("inf"), "per_ticker_cap": 0.05, "max_positions": 10, "kelly_min": 0.04, "kelly_max": 0.06, "order_type": "limit_ioc"},
]

# Execution / API
ALPACA_BASE_URL_PAPER = "https://paper-api.alpaca.markets"
ALPACA_BASE_URL_LIVE  = "https://api.alpaca.markets"
GIST_API_BASE_URL     = "https://api.github.com/gists"
STATE_GIST_FILENAME      = "state.json"
DECISIONS_GIST_FILENAME  = "decisions.csv"
GIST_RETRY_DELAY_SECONDS = 3

# Backtest
BACKTEST_SLIPPAGE_BPS    = 10
BACKTEST_STARTING_EQUITY = 100.0
BACKTEST_CACHE_DIR       = ".backtest_cache"

# Paper burn-in
PAPER_BURN_IN_DAYS = 30
