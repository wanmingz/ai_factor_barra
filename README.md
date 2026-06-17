# Portfolio Analytics

A small Barra-style factor demo that downloads market data, builds factor exposures, and estimates factor returns via weighted least squares (WLS).

## What it does

The pipeline has three steps:

1. **Prepare Y (excess returns)** — Download daily prices from Yahoo Finance, compute returns, and subtract the benchmark (`SPY`) to get excess returns.
2. **Prepare X (factor exposures)** — Build a cross-sectional factor matrix with four Barra-style style factors (`Size`, `Value`, `Momentum`, `Volatility`), then Z-score normalize.
3. **Estimate factor returns** — Run a WLS regression on the latest trading day to solve for factor returns:

   `beta = (X^T W X)^(-1) X^T W y`

## Project structure

```
portfolio_analytics/
├── config.py    # Tickers, benchmark, date range, factor settings
├── barra.py     # Data pipeline and WLS regression
└── README.md
```

## Requirements

- Python 3.11+
- `yfinance`
- `pandas`
- `numpy`

Install dependencies:

```bash
pip install yfinance pandas numpy
```

## Configuration

Edit `config.py` to change the universe, date range, and factor parameters:

```python
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
BENCHMARK = "SPY"

START_DATE = "2024-01-01"
END_DATE = "2026-01-01"

LOOKBACK_MOM = 252   # ~12 months for momentum
SKIP_RECENT = 21     # skip last month (Barra 12M-1M convention)
LOOKBACK_VOL = 60    # ~3 months for volatility

FACTOR_NAMES = ['Size', 'Value', 'Momentum', 'Volatility']
```

**Note:** You need at least as many stocks as factors (ideally more) for the WLS regression to be well-conditioned.

## Run

```bash
cd portfolio_analytics
python barra.py
```

## Output

Example output:

```
--- the dependent variable Y (excess returns) is aligned successfully ---
Ticker          AAPL      MSFT      NVDA     GOOGL      AMZN      META
Date
2025-12-31  0.002941 -0.000509  0.001864  0.004701  0.000055 -0.001391

--- the independent variable X (Barra style factor exposure matrix) is aligned successfully ---
           Size     Value  Momentum  Volatility
AAPL   0.665218 -1.327353 -0.475431   -1.294973
MSFT  -0.235059  0.572073 -0.359180   -1.191009
NVDA   0.960927 -1.090309  0.413741    1.066147
...

--- the factor returns are estimated successfully ---
Size_Return: 0.000691
Value_Return: -0.000524
Momentum_Return: 0.001343
Volatility_Return: -0.000288
```

- **Y** — Each stock's daily return minus the benchmark return.
- **X** — Standardized factor exposures (see table below).
- **Factor returns** — How much each style factor contributed on the latest day.

## Factors

| Factor | Definition | Data source |
|--------|------------|-------------|
| Size | `ln(market cap)` | `yfinance` ticker info |
| Value | `1 / price-to-book` | `yfinance` ticker info |
| Momentum | Cumulative return over past 252 trading days, excluding the most recent 21 days | Historical prices |
| Volatility | Standard deviation of daily returns over the past 60 trading days | Historical prices |

## Limitations

This is a learning demo, not production Barra:

- Small stock universe — results improve with more names, but remain illustrative.
- `Size` and `Value` exposures use a **current snapshot** from `yfinance`, not historical daily values.
- `Momentum` and `Volatility` are computed from price history as of the latest date, but are not rolled forward daily in this script.
- WLS weights use `ln(market cap)` as a simplified stand-in (real Barra uses sqrt market cap).
- No industry factors, neutralization, or specific risk model.

## License

Personal / educational use.
