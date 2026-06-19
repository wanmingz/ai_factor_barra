import yfinance as yf
import pandas as pd
import numpy as np

from config import FACTOR_NAMES, STYLE_FACTOR_NAMES, TICKERS, BENCHMARK, START_DATE, END_DATE
from features import build_factor_panel, zscore_cross_section
from ai_factor import build_ai_panel

# 1. download the data
all_assets = TICKERS + [BENCHMARK]
data = yf.download(all_assets, start=START_DATE, end=END_DATE)

close = data["Close"]
returns = close.pct_change().dropna()

# 2. excess return Y (date × ticker)
Y_excess = returns[TICKERS].sub(returns[BENCHMARK], axis=0) #every column in the TICKERS dataframe is subtracted by the benchmark returns

# 3. daily rolling factor exposure X (date, ticker)× factors (incl. AI)
ai_panel = build_ai_panel(dates=returns.index)
X_panel = build_factor_panel(returns, close, ai_panel=ai_panel)
X_panel = zscore_cross_section(X_panel)

print("--- X panel (daily rolling exposures) ---")
print(X_panel.tail(12))   # the last 12 days

# 4. select one day to do WLS (default the last day)
date = Y_excess.index[-1] #the last day
X_day = X_panel.loc[date][FACTOR_NAMES].dropna() #drop the rows with missing values
tickers_t = X_day.index.tolist()

y_t = Y_excess.loc[date, tickers_t].values.reshape(-1, 1) #the excess returns of the selected day
X_t = X_day.values.astype(float)

weights = [
    np.log(mc) if (mc := yf.Ticker(t).info.get("marketCap")) else 0.0
    for t in tickers_t
]
W_t = np.diag(weights)

X_T_W = X_t.T @ W_t
beta = np.linalg.inv(X_T_W @ X_t) @ X_T_W @ y_t

print(f"\n--- Factor returns on {date.date()} ---")

for i, factor in enumerate(FACTOR_NAMES):
    print(f"{factor}_Return: {beta[i][0]:.6f}")
