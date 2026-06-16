# Portfolio Analytics

A small Barra-style factor demo that downloads market data, builds factor exposures, and estimates factor returns via weighted least squares (WLS).

## What it does

The pipeline has three steps:

1. **Prepare Y (excess returns)** — Download daily prices from Yahoo Finance, compute returns, and subtract the benchmark (`SPY`) to get excess returns.
2. **Prepare X (factor exposures)** — Build a cross-sectional factor matrix with Barra-style style factors (currently `Size` and `Value`), then Z-score normalize.
3. **Estimate factor returns** — Run a WLS regression on the latest trading day to solve for factor returns:

   `beta = (X^T W X)^(-1) X^T W y`

## Project structure

```
portfolio_analytics/
├── config.py    # Tickers, benchmark, date range
├── main.py      # Data pipeline and WLS regression
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

Edit `config.py` to change the universe and date range:

```python
TICKERS = ["AAPL", "MSFT", "NVDA"]
BENCHMARK = "SPY"

START_DATE = "2024-01-01"
END_DATE = "2026-01-01"
```

## Run

```bash
cd portfolio_analytics
python main.py
```

## Output

Example output:

```
--- the dependent variable Y (excess returns) is aligned successfully ---
Ticker          AAPL      MSFT      NVDA
Date
2025-12-31  0.002941 -0.000509  0.001863

--- the independent variable X (Barra style factor exposure matrix) is aligned successfully ---
          Size     Value  ...
AAPL  0.309020 -0.684088  ...
MSFT -1.118035  1.147660  ...
NVDA  0.809015 -0.463572  ...

--- 成功解出当天 Barra 纯风格因子的收益率 (Factor Returns) ---
Size_Return: -0.000978
Value_Return: -0.002659
```

- **Y** — Each stock's daily return minus the benchmark return.
- **X** — Standardized factor exposures (`Size` = ln(market cap), `Value` = 1 / price-to-book).
- **Factor returns** — How much each style factor contributed on the latest day.

## Factors

| Factor | Definition |
|--------|------------|
| Size | `ln(market cap)` |
| Value | `1 / price-to-book` |

`Momentum`, `Volatility`, and `AI_Sentiment` columns are placeholders and not yet implemented.

## Limitations

This is a learning demo, not production Barra:

- Only 3 stocks — regression results are unstable with a tiny cross-section.
- Factor exposures use a **current snapshot** from `yfinance`, not historical daily values.
- WLS weights use `ln(market cap)` as a simplified stand-in (real Barra uses sqrt market cap).
- Extra factor columns must be filled or removed before regression, or the matrix becomes singular.

## License

Personal / educational use.
