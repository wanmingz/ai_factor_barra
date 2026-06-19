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
START_DATE = "2016-01-01"
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

# -----------------------------------------------------------------------------
# 5. Backtesting
# -----------------------------------------------------------------------------

# TOP_PCT: Long top & short bottom percentile by prediction at each rebalance
# Used in: backtest.long_short_backtest()
TOP_PCT = 0.2

# COST_BPS: One-way transaction cost in basis points (0 = ignore costs)
# Used in: backtest.long_short_backtest() — charged only on rebalance days
COST_BPS = 0

# REBALANCE_FREQ: "monthly" (first trading day of month) or "daily"
# Used in: backtest.long_short_backtest()
REBALANCE_FREQ = "monthly"

# -----------------------------------------------------------------------------
# 6. AI Theme Agent
# -----------------------------------------------------------------------------

# THEMES_FILE: Stock-to-theme mapping (ticker, theme, weight)
# Used in: theme_agent.load_theme_map()
THEMES_FILE = "themes.csv"

# THEME_ETF_PROXY: ETF used as market context for each theme (recent momentum)
# Used in: theme_agent.fetch_theme_context()
THEME_ETF_PROXY = {
    "AI": "BOTZ",
    "Cloud": "WCLD",
    "Semiconductors": "SMH",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer": "XLY",
    "Energy": "XLE",
    "Media": "XLC",
    "Industrials": "XLI",
    "Software": "IGV",
}

# THEME_CONTEXT_LOOKBACK: Trading days of ETF return history fed to the agent
THEME_CONTEXT_LOOKBACK = 63

# GEMINI_MODEL: Gemini model for theme scoring (requires GEMINI_API_KEY env var)
GEMINI_MODEL = "gemini-2.5-flash-lite"

# THEME_SCORES_DIR: Where daily theme scores are saved as JSON
THEME_SCORES_DIR = "theme_scores"

# ENV_FILE: Local file for API keys (gitignored; copy from .env.example)
ENV_FILE = ".env"

# -----------------------------------------------------------------------------
# 7. News RAG (local embedding — no paid API)
# -----------------------------------------------------------------------------

# NEWS_LOOKBACK_DAYS: Only include news published within this window before as_of
NEWS_LOOKBACK_DAYS = 7

# NEWS_MAX_PER_TICKER: Max headlines fetched per stock from yfinance
NEWS_MAX_PER_TICKER = 10

# NEWS_RAG_TOP_K: Top articles retrieved per theme for the agent prompt
NEWS_RAG_TOP_K = 5

# NEWS_INDEX_DIR: Cached news docs + embedding vectors by date
NEWS_INDEX_DIR = "data/news_index"

# EMBEDDING_MODEL: Local sentence-transformers model (free, runs offline after download)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"