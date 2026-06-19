import yfinance as yf
import pandas as pd
import numpy as np
from ai_factor import snapshot_exposure

from config import (
    FACTOR_NAMES, TICKERS, BENCHMARK, START_DATE, END_DATE,
    LOOKBACK_MOM, SKIP_RECENT, LOOKBACK_VOL, AI_SCORE_MODE
)

#==========================================
# prepare y 
#==========================================

all_assets = TICKERS + [BENCHMARK]

# download the data from yfinance
data = yf.download(all_assets, start=START_DATE, end=END_DATE)

# Calculate the returns of the assets
returns = data['Close'].pct_change().dropna() 
#pct change means percentage change, which is the change in the price of the asset over the previous day's price
#dropna means drop the rows with missing values

Y_excess = returns[TICKERS].sub(returns[BENCHMARK], axis=0)
#axis=0 means subtract the SPY returns from the returns of the TICKERS
print("--- the dependent variable Y (excess returns) is aligned successfully ---")
print(Y_excess.tail(3))
#==========================================
# prepare x
#==========================================

# prepare an empty dataframe with the TICKERS and size and value columns.
X_raw = pd.DataFrame(index=TICKERS, columns= FACTOR_NAMES)

for t in TICKERS:
    ticker_obj = yf.Ticker(t)
    info = ticker_obj.info
    # size factor is the natural logarithm of the market cap
    market_cap = info.get('marketCap', np.nan)
    X_raw.loc[t, 'Size'] = np.log(market_cap) if market_cap else np.nan
    
    # value factor is the inverse of the price-to-book ratio (Book-to-Market = 1 / PB)
    price_to_book = info.get('priceToBook', np.nan)
    X_raw.loc[t, 'Value'] = (1.0 / price_to_book) if price_to_book else np.nan

    # Momentum: past 12 months returns, excluding the latest 1 month
    r = returns[t]
    if len(r) > LOOKBACK_MOM:
        mom_window = r.iloc[-LOOKBACK_MOM:-SKIP_RECENT]
        X_raw.loc[t, 'Momentum'] = (1 + mom_window).prod() - 1
    else:
        X_raw.loc[t, 'Momentum'] = np.nan
    # Volatility: past 60 days return standard deviation
    vol_window = r.iloc[-LOOKBACK_VOL:]
    X_raw.loc[t, 'Volatility'] = vol_window.std() if len(vol_window) > 1 else np.nan

as_of = returns.index[-1].strftime("%Y-%m-%d")
ai_exposure = snapshot_exposure(as_of, mode=AI_SCORE_MODE)
X_raw["AI"] = ai_exposure.reindex(TICKERS)

# z-score standardization
X_normalized = X_raw.apply(lambda x: (x - x.mean()) / x.std(), axis=0).fillna(0)

print("--- the independent variable X (Barra style factor exposure matrix) is aligned successfully ---")
print(X_normalized)

#==========================================
# barra style factor exposure matrix
#==========================================
#y_t is the excess returns of the latest day
y_t = Y_excess.iloc[-1].values.reshape(-1, 1) # (N, 1)
X_t = X_normalized.values.astype(float)       # (N, M)

weights = [np.log(yf.Ticker(t).info.get('marketCap')) for t in TICKERS]
W_t = np.diag(weights) # (N, N)

# beta = (X^T * W * X)^(-1) * X^T * W * y
X_T_W = X_t.T @ W_t
beta = np.linalg.inv(X_T_W @ X_t) @ X_T_W @ y_t # （M,1）

print("--- the factor returns are estimated successfully ---")
for i, factor in enumerate(FACTOR_NAMES):
    print(f"{factor}_Return: {beta[i][0]:.6f}")