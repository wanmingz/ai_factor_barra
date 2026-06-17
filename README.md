# Portfolio Analytics

A small Barra-style factor demo that downloads market data, builds factor exposures, and estimates factor returns via weighted least squares (WLS).

Two entry points:

- **`barra.py`** — static cross-section demo (snapshot factor exposures on the latest day)
- **`barra_panel.py`** — daily rolling factor panel via `features.py` (for time-series factor returns and ML)

## What it does

1. **Prepare Y (excess returns)** — Download daily prices from Yahoo Finance, compute returns, and subtract the benchmark (`SPY`).
2. **Prepare X (factor exposures)** — Build a cross-sectional factor matrix with four style factors (`Size`, `Value`, `Momentum`, `Volatility`), then Z-score normalize.
3. **Estimate factor returns** — Run WLS on a single trading day:

   `beta = (X^T W X)^(-1) X^T W y`

## Project structure

```
portfolio_analytics/
├── config.py       # Tickers, benchmark, date range, factor settings
├── features.py     # Daily rolling factor computation (panel data)
├── barra.py        # Static cross-section WLS demo
├── barra_panel.py  # Daily rolling exposures + WLS on latest day
└── README.md
```

## Requirements

- Python 3.11+
- `yfinance`
- `pandas`
- `numpy`

```bash
pip install yfinance pandas numpy
```

## Configuration

Edit `config.py`:

```python
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
BENCHMARK = "SPY"

START_DATE = "2020-01-01"
END_DATE = "2026-01-01"

LOOKBACK_MOM = 63    # ~3 months momentum (shorter window for more ML samples)
SKIP_RECENT = 5      # skip ~1 week
LOOKBACK_VOL = 20    # ~1 month volatility

FACTOR_NAMES = ['Size', 'Value', 'Momentum', 'Volatility']
```

You need at least as many stocks as factors (ideally more) for WLS to be well-conditioned.

## Run

**Static demo** (single-day snapshot exposures):

```bash
python barra.py
```

**Daily rolling panel** (uses `features.py`):

```bash
python barra_panel.py
```

## `barra.py` vs `barra_panel.py`

| | `barra.py` | `barra_panel.py` |
|---|-----------|------------------|
| Factor exposures | Static snapshot (`yfinance` info) | Daily rolling (`features.py`) |
| X shape | `(N stocks, M factors)` | `(date, ticker) × M factors` |
| Use case | Quick Barra WLS demo | Factor time series, ML feature panel |

## Factors

### `barra.py` (static)

| Factor | Definition | Data source |
|--------|------------|-------------|
| Size | `ln(market cap)` | `yfinance` ticker info |
| Value | `1 / price-to-book` | `yfinance` ticker info |
| Momentum | Cumulative return over lookback, excluding recent days | Historical prices |
| Volatility | Std of daily returns over lookback | Historical prices |

### `features.py` / `barra_panel.py` (daily rolling)

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

## Output

`barra_panel.py` example:

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

## ML-ready panel

`barra_panel.py` produces a daily `(date, ticker)` feature panel. To build a training set:

```python
y = Y_excess.stack().rename("excess_return")
ml_df = X_panel.join(y, how="inner")
```

Each row is one stock on one day with factor exposures and same-day excess return. Add a forward-return label for prediction tasks (use time-based train/test splits, not random shuffle).

## Limitations

This is a learning demo, not production Barra:

- Small stock universe — illustrative only.
- `barra.py` uses static `yfinance` snapshots for Size/Value.
- `features.py` Value uses a simplified B/P proxy (not quarterly fundamentals forward-filled).
- Size in the panel uses `ln(price)` as a market-cap proxy.
- WLS weights use `ln(market cap)` (real Barra uses sqrt market cap).
- No industry factors, neutralization, or specific risk model.

## License

Personal / educational use.
