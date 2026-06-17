TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
BENCHMARK = "SPY"

START_DATE = "2020-01-01"
END_DATE = "2026-06-16"
LOOKBACK_MOM = 63
SKIP_RECENT = 5
LOOKBACK_VOL = 20
FACTOR_NAMES = ['Size', 'Value', 'Momentum', 'Volatility']

#ml related parameters
FORWARD_DAYS = 5       # ML label: cumulative excess return over next N days
TRAIN_RATIO = 0.8      # time-based split: first 80% of dates for training
