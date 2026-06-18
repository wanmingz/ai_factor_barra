# Equity Factor ML

A small Barra-style factor demo that downloads market data, builds factor exposures, estimates factor returns via weighted least squares (WLS), and trains ML models to predict forward excess returns.

Three entry points:

- **`barra.py`** — static cross-section demo (snapshot factor exposures on the latest day)
- **`barra_panel.py`** — daily rolling factor panel via `features.py` (time-series factor returns)
- **`ml_predict.py`** — ML pipeline: factor panel → forward-return label → time-based train/test → model comparison

## What it does

1. **Prepare Y (excess returns)** — Download daily prices from Yahoo Finance, compute returns, and subtract the benchmark (`SPY`).
2. **Prepare X (factor exposures)** — Build a cross-sectional factor matrix with four style factors (`Size`, `Value`, `Momentum`, `Volatility`), then Z-score normalize.
3. **Estimate factor returns** — Run WLS on a single trading day:

   `beta = (X^T W X)^(-1) X^T W y`

4. **Predict forward returns (ML)** — Use today's factor exposures to predict cumulative excess return over the next N trading days; evaluate with IC, Rank IC, and a long-short backtest.

## Project structure

```
equity_factor_ml/
├── config.py       # Global settings (universe, dates, factors, ML) — documented inline
├── features.py     # Daily rolling factor computation (panel data)
├── barra.py        # Static cross-section WLS demo
├── barra_panel.py  # Daily rolling exposures + WLS on latest day
├── ml_predict.py   # ML training, evaluation & long-short backtest
├── backtest.py     # Long-short backtest + rebalance holdings log
└── README.md
```

## Requirements

- Python 3.11+
- `yfinance`
- `pandas`
- `numpy`
- `scikit-learn`

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install yfinance pandas numpy scikit-learn
```

## Configuration

All scripts read parameters from `config.py`. The file is organized into five sections; each variable is documented inline with what it defines and which modules use it.

### Config reference

| Variable | Section | Defines | Used by |
|----------|---------|---------|---------|
| `TICKERS` | Universe | Stock universe (50 liquid S&P 500 names, yfinance symbols) | `features.py`, `ml_predict.py`, `barra.py`, `barra_panel.py` |
| `BENCHMARK` | Universe | Benchmark ETF for excess returns (`y_excess = stock − benchmark`) | `load_market_data`, `barra.py`, `barra_panel.py` |
| `START_DATE` | Data range | Historical data start date | All `yf.download` calls |
| `END_DATE` | Data range | Historical data end date | All `yf.download` calls |
| `LOOKBACK_MOM` | Factor engineering | Momentum lookback in trading days (~3 months) | `features.compute_momentum()` |
| `SKIP_RECENT` | Factor engineering | Days skipped at the end of the momentum window | `features.compute_momentum()` |
| `LOOKBACK_VOL` | Factor engineering | Volatility lookback window (~1 month) | `features.compute_volatility()` |
| `FACTOR_NAMES` | Factor engineering | Factor column names (order matches X matrix) | Factor panel, z-score, ML features, WLS |
| `FORWARD_DAYS` | Machine learning | ML label: cumulative excess return over next N days | `forward_excess_return`, `build_ml_dataset` |
| `TRAIN_RATIO` | Machine learning | Time-based train/test split (no random shuffle) | `ml_predict.time_split()` |
| `TOP_PCT` | Backtesting | Long/short leg size at each rebalance | `backtest.long_short_backtest()` |
| `COST_BPS` | Backtesting | One-way transaction cost in bps (`0` = no costs), on rebalance days only | `backtest.long_short_backtest()` |
| `REBALANCE_FREQ` | Backtesting | `"monthly"` or `"daily"` rebalance schedule | `backtest.long_short_backtest()` |

### Example (`config.py`)

```python
# 1. Universe & Benchmark
TICKERS = ["AAPL", "MSFT", "NVDA", ...]  # 50 names — see config.py for full list
BENCHMARK = "SPY"

# 2. Data Range
START_DATE = "2016-01-01"
END_DATE = "2026-06-16"

# 3. Factor Engineering
LOOKBACK_MOM = 63
SKIP_RECENT = 5
LOOKBACK_VOL = 20
FACTOR_NAMES = ["Size", "Value", "Momentum", "Volatility"]

# 4. Machine Learning
FORWARD_DAYS = 5
TRAIN_RATIO = 0.8

# 5. Backtesting
TOP_PCT = 0.2
COST_BPS = 0   # 0 = ignore transaction costs
REBALANCE_FREQ = "monthly"
```

You need at least as many stocks as factors (ideally more) for WLS to be well-conditioned. The default universe is **50 liquid S&P 500 names** — large enough for stable cross-sectional IC, small enough for a learning demo.

## Run

**Static demo** (single-day snapshot exposures):

```bash
python barra.py
```

**Daily rolling panel** (uses `features.py`):

```bash
python barra_panel.py
```

**ML prediction** (factor panel → forward return → model comparison):

```bash
python ml_predict.py
```

> First run with 50 tickers may take several minutes: `yf.download` for 51 symbols plus per-ticker `yfinance` info calls in `compute_value`.

## `barra.py` vs `barra_panel.py` vs `ml_predict.py`

| | `barra.py` | `barra_panel.py` | `ml_predict.py` |
|---|-----------|------------------|-----------------|
| Factor exposures | Static snapshot (`yfinance` info) | Daily rolling (`features.py`) | Daily rolling (`features.py`) |
| X shape | `(N stocks, M factors)` | `(date, ticker) × M factors` | `(date, ticker) × M factors` |
| Label | Same-day excess return | Same-day excess return | Forward N-day cumulative excess return |
| Use case | Quick Barra WLS demo | Factor time series | Predict & rank stocks |

## Factors

### `barra.py` (static)

| Factor | Definition | Data source |
|--------|------------|-------------|
| Size | `ln(market cap)` | `yfinance` ticker info |
| Value | `1 / price-to-book` | `yfinance` ticker info |
| Momentum | Cumulative return over lookback, excluding recent days | Historical prices |
| Volatility | Std of daily returns over lookback | Historical prices |

### `features.py` / `barra_panel.py` / `ml_predict.py` (daily rolling)

| Factor | Definition | Data source |
|--------|------------|-------------|
| Size | `ln(price)` proxy | Daily close prices |
| Value | `book_per_share / price` (B/P proxy) | Daily close + `priceToBook` |
| Momentum | `(1 + r).prod() - 1` over `LOOKBACK_MOM`, skip `SKIP_RECENT` | Daily returns |
| Volatility | Std of returns over `LOOKBACK_VOL` | Daily returns |

`features.py` functions:

- `compute_momentum(returns)` / `compute_volatility(returns)` — price-based rolling factors
- `compute_size(close)` / `compute_value(close)` — size and value proxies
- `build_factor_panel(returns, close)` — merge all factors into a `(date, ticker)` panel
- `zscore_cross_section(panel)` — cross-sectional Z-Score per day

## ML pipeline (`ml_predict.py`)

### Data flow

```
load_market_data()          → close, returns, y_excess
next_day_excess()           → ret_1d (next-day excess, for backtest PnL)
build_factor_panel()        → raw factor panel
zscore_cross_section()      → x_panel (features X)
forward_excess_return()     → target_Nd (label y)
build_ml_dataset()          → ml_df (X + y joined)
time_split()                → train / test (by date, no shuffle)
long_short_backtest()       → daily LS returns + holdings DataFrame
```

ML labels use forward `FORWARD_DAYS`-day excess return. The backtest uses **next-day** excess return (`ret_1d`) with **monthly rebalancing** on the first trading day of each month (`REBALANCE_FREQ = "monthly"`). Set `REBALANCE_FREQ = "daily"` for daily rebalance.

### Long-short backtest (`backtest.py`)

Each rebalance day:

1. Rank stocks in the test set by model prediction `pred`
2. **Long** top `TOP_PCT` (default 20%) — e.g. 10 of 50 names
3. **Short** bottom `TOP_PCT`
4. Hold until the next rebalance; compute daily PnL from held names' `ret_1d`

`long_short_backtest()` returns `(daily_ls, holdings_df)`. `holdings_df` logs every rebalance:

| Column | Meaning |
|--------|---------|
| `date` | Rebalance date |
| `side` | `"long"` or `"short"` |
| `ticker` | Stock symbol |
| `pred` | Model score used for ranking |

`ml_predict.py` prints the **last rebalance** holdings after each model's evaluation.

### Models compared

| Model | Description |
|-------|-------------|
| Baseline (predict 0) | Always predict zero |
| Momentum only (OLS) | Single-factor linear regression on Momentum |
| Ridge (4 factors) | L2-regularized linear model on all factors |
| RandomForest (4 factors) | Tree ensemble on all factors |

### Evaluation metrics

Computed per trading day on the test set, then averaged:

| Metric | Meaning |
|--------|---------|
| **MSE** | Mean squared error between predicted and actual returns |
| **Mean IC** | Pearson correlation of predictions vs actual returns (cross-sectional, per day) |
| **Mean Rank IC** | Spearman rank correlation (per day) — primary metric for stock ranking |
| **Ann Return** | Annualized long-short portfolio return (top/bottom `TOP_PCT`, no transaction costs by default) |
| **Sharpe** | Annualized Sharpe ratio of daily LS returns |
| **Max Drawdown** | Maximum peak-to-trough drawdown of cumulative LS returns |
| **Hit Rate** | Fraction of days with positive LS return |

### Example output (`START_DATE = "2016-01-01"`, 50 tickers)

```
--- ML dataset ---
Rows: 122784 | Features: ['Size', 'Value', 'Momentum', 'Volatility']
Label:  forward 5-day excess return
Train:  98208 rows (through 2024-05-21)
Test:   24576 rows (from 2024-05-22)

--- Ridge (4 factors) ---
Mean Rank IC: 0.0546
Ann Return:   57.87%
Sharpe:       1.95
Max Drawdown: -20.23%
Hit Rate:     55.86%
Holdings on last rebalance (2026-06-01):
      date  side ticker     pred
2026-06-01  long    AMD 0.013234
2026-06-01  long  CMCSA 0.006855
...
2026-06-01 short  BRK-B 0.000711

--- Model comparison (test set) ---
                          mean_ic  mean_rank_ic  ann_return  sharpe  max_drawdown  hit_rate
Ridge (4 factors)          0.0730        0.0546      0.5787  1.9521       -0.2023    0.5586
RandomForest (4 factors)   0.0197        0.0373      0.3174  1.4142       -0.1906    0.5430
Momentum only (OLS)        0.0246        0.0216      0.2193  0.8045       -0.2722    0.5391
Baseline (predict 0)          NaN           NaN     -0.2166 -1.8575       -0.4113    0.4531
```

Rank IC is the primary signal metric. Backtest returns are gross of costs (`COST_BPS = 0`) and can look high on a short test window — treat as illustrative, not live-trading estimates.

## Output (`barra_panel.py`)

```
--- X panel (daily rolling exposures) ---
                         Size     Value  Momentum  Volatility
date       ticker
2025-12-31 AAPL          ...
           MSFT          ...
           ...

--- Factor returns on 2025-12-31 ---
Size_Return: ...
Value_Return: ...
Momentum_Return: ...
Volatility_Return: ...
```

- **Y** — Each stock's daily return minus the benchmark return.
- **X** — Standardized factor exposures per stock per day.
- **Factor returns** — How much each style factor contributed on the chosen day.

## Limitations

This is a learning demo, not production Barra:

- 50-stock subset of the S&P 500 — better than 6 names, but not a full index; current list is a survivorship-biased snapshot.
- `barra_panel.py` aligns `X` and `y` to the same tickers on the chosen WLS day (not all names may have Value factor data).
- `barra.py` uses static `yfinance` snapshots for Size/Value.
- `features.py` Value uses a simplified B/P proxy (not quarterly fundamentals forward-filled); `compute_value` calls `yf.Ticker(t).info` once per ticker.
- Size in the panel uses `ln(price)` as a market-cap proxy.
- WLS weights use `ln(market cap)` (real Barra uses sqrt market cap).
- ML label uses arithmetic sum of daily excess returns (not compounded).
- Single train/test split — no walk-forward validation yet.
- Long-short backtest rebalances monthly by default (`REBALANCE_FREQ = "monthly"`); no transaction costs (`COST_BPS = 0`).
- No industry factors, neutralization, or specific risk model.

## License

Personal / educational use.
