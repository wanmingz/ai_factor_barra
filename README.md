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

4. **Predict forward returns (ML)** — Use today's factor exposures to predict cumulative excess return over the next N trading days; evaluate with IC and Rank IC.

## Project structure

```
equity_factor_ml/
├── config.py       # Global settings (universe, dates, factors, ML) — documented inline
├── features.py     # Daily rolling factor computation (panel data)
├── barra.py        # Static cross-section WLS demo
├── barra_panel.py  # Daily rolling exposures + WLS on latest day
├── ml_predict.py   # ML training & evaluation pipeline
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

All scripts read parameters from `config.py`. The file is organized into four sections; each variable is documented inline with what it defines and which modules use it.

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

### Example (`config.py`)

```python
# 1. Universe & Benchmark
TICKERS = ["AAPL", "MSFT", "NVDA", ...]  # 50 names — see config.py for full list
BENCHMARK = "SPY"

# 2. Data Range
START_DATE = "2020-01-01"
END_DATE = "2026-06-16"

# 3. Factor Engineering
LOOKBACK_MOM = 63
SKIP_RECENT = 5
LOOKBACK_VOL = 20
FACTOR_NAMES = ["Size", "Value", "Momentum", "Volatility"]

# 4. Machine Learning
FORWARD_DAYS = 5
TRAIN_RATIO = 0.8
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
build_factor_panel()        → raw factor panel
zscore_cross_section()      → x_panel (features X)
forward_excess_return()     → target_Nd (label y)
build_ml_dataset()          → ml_df (X + y joined)
time_split()                → train / test (by date, no shuffle)
```

Each row in `ml_df` is one stock on one day: four factor exposures (X) and the cumulative excess return over the next `FORWARD_DAYS` trading days (y).

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

### Example output (50-ticker universe)

```
--- ML dataset ---
Rows: 74496 | Features: ['Size', 'Value', 'Momentum', 'Volatility']
Label:  forward 5-day excess return
Train:  59568 rows (through 2025-03-12)
Test:   14928 rows (from 2025-03-13)

--- Ridge (4 factors) ---
MSE:          0.002262
Mean IC:      0.0951
Mean Rank IC: 0.0767

--- Model comparison (test set) ---
                          mean_ic  mean_rank_ic
Ridge (4 factors)          0.0951        0.0767
RandomForest (4 factors)   0.0470        0.0652
Momentum only (OLS)        0.0601        0.0507
Baseline (predict 0)          NaN           NaN
```

Ridge (4-factor linear model) typically wins on Rank IC. IC values are lower but more credible than a 6-stock universe, where small cross-sections inflate correlation estimates.

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

- 50-stock subset of the S&P 500 — better than 6 names, but not a full index.
- `barra.py` uses static `yfinance` snapshots for Size/Value.
- `features.py` Value uses a simplified B/P proxy (not quarterly fundamentals forward-filled); `compute_value` calls `yf.Ticker(t).info` once per ticker.
- Size in the panel uses `ln(price)` as a market-cap proxy.
- WLS weights use `ln(market cap)` (real Barra uses sqrt market cap).
- ML label uses arithmetic sum of daily excess returns (not compounded).
- Single train/test split — no walk-forward validation or portfolio backtest yet.
- No industry factors, neutralization, or specific risk model.

## License

Personal / educational use.
