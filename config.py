# =============================================================================
# config.py — Global settings (imported by all scripts)
#
# Used by:
#   barra.py, barra_panel.py  → universe, dates, factor names
#   features.py               → universe, momentum/volatility windows
#   ml_predict.py             → all ML-related parameters
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Universe & Benchmark
# -----------------------------------------------------------------------------

# TICKERS: List of stocks to analyze/predict (yfinance symbols)
# Used in: load_market_data, factor computation, excess returns Y, WLS cross-section
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "AVGO", "TSLA",
    "WMT", "LLY", "JPM", "V", "UNH", "XOM", "MA", "ORCL", "COST", "HD", "PG",
    "JNJ", "NFLX", "BAC", "ABBV", "CRM", "KO", "AMD", "MRK", "PEP", "TMO",
    "CSCO", "ACN", "LIN", "MCD", "ADBE", "WFC", "DIS", "GE", "TXN", "INTU",
    "AMAT", "QCOM", "IBM", "CAT", "VZ", "CMCSA", "AMGN", "PFE", "NOW", "MU",
]

# BENCHMARK: Benchmark ETF for excess returns
# Formula: y_excess = stock return - BENCHMARK return
# Used in: barra.py, barra_panel.py, ml_predict.load_market_data()
BENCHMARK = "SPY"

# -----------------------------------------------------------------------------
# 2. Data Range
# -----------------------------------------------------------------------------

# START_DATE / END_DATE: Historical data download window (Yahoo Finance)
# Used in: yf.download(..., start=START_DATE, end=END_DATE)
START_DATE = "2020-01-01"
END_DATE = "2026-06-16"

# -----------------------------------------------------------------------------
# 3. Factor Engineering
# -----------------------------------------------------------------------------

# LOOKBACK_MOM: Momentum lookback in trading days (~3 months)
# Used in: features.compute_momentum()
LOOKBACK_MOM = 63

# SKIP_RECENT: Skip the most recent N days when computing momentum
# Used in: features.compute_momentum() → hist.iloc[-LOOKBACK_MOM:-SKIP_RECENT]
SKIP_RECENT = 5

# LOOKBACK_VOL: Volatility lookback window (~1 month)
# Used in: features.compute_volatility()
LOOKBACK_VOL = 20

# FACTOR_NAMES: Factor column names (order matches X matrix columns)
# Used in: factor panel, z-score, ML features, WLS regression
FACTOR_NAMES = ["Size", "Value", "Momentum", "Volatility"]

# -----------------------------------------------------------------------------
# 4. Machine Learning
# -----------------------------------------------------------------------------

# FORWARD_DAYS: ML label — cumulative excess return over the next N trading days
# Used in: forward_excess_return(), build_ml_dataset(), column target_{N}d
FORWARD_DAYS = 5

# TRAIN_RATIO: Time-based train/test split (first N% of dates = train)
# Used in: ml_predict.time_split() — no random shuffle (avoids look-ahead bias)
TRAIN_RATIO = 0.8