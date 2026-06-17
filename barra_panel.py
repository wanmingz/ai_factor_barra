import yfinance as yf
import pandas as pd
import numpy as np

from config import FACTOR_NAMES, TICKERS, BENCHMARK, START_DATE, END_DATE
from features import build_factor_panel, zscore_cross_section

# 1. 下载数据
all_assets = TICKERS + [BENCHMARK]
data = yf.download(all_assets, start=START_DATE, end=END_DATE)

close = data["Close"]
returns = close.pct_change().dropna()

# 2. 超额收益 Y（date × ticker）
Y_excess = returns[TICKERS].sub(returns[BENCHMARK], axis=0)

# 3. 每日滚动因子暴露 X（date, ticker）× factors
X_panel = build_factor_panel(returns, close)
X_panel = zscore_cross_section(X_panel)

print("--- X panel (daily rolling exposures) ---")
print(X_panel.tail(12))   # 最后一天 6 只股票

# 4. 选一天做 WLS（默认最后一天）
date = Y_excess.index[-1]
y_t = Y_excess.loc[date].values.reshape(-1, 1)
X_t = X_panel.loc[date][FACTOR_NAMES].values.astype(float)

weights = [np.log(yf.Ticker(t).info.get("marketCap")) for t in TICKERS]
W_t = np.diag(weights)

X_T_W = X_t.T @ W_t
beta = np.linalg.inv(X_T_W @ X_t) @ X_T_W @ y_t

print(f"\n--- Factor returns on {date.date()} ---")
for i, factor in enumerate(FACTOR_NAMES):
    print(f"{factor}_Return: {beta[i][0]:.6f}")