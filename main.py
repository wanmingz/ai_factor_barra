import yfinance as yf
import pandas as pd
import numpy as np

from config import TICKERS, BENCHMARK, START_DATE, END_DATE

#==========================================
# prepare y 
#==========================================

tickers = TICKERS
all_assets = tickers + [BENCHMARK]

# download the data from yfinance
data = yf.download(all_assets, start=START_DATE, end=END_DATE)

# Calculate the returns of the assets
returns = data['Close'].pct_change().dropna() 
#pct change means percentage change, which is the change in the price of the asset over the previous day's price
#dropna means drop the rows with missing values

Y_excess = returns[tickers].sub(returns[BENCHMARK], axis=0)
#axis=0 means subtract the SPY returns from the returns of the tickers
print("--- the dependent variable Y (excess returns) is aligned successfully ---")
print(Y_excess.tail(3))
#==========================================
# prepare x
#==========================================

# prepare an empty dataframe with the tickers and size and value columns.
X_raw = pd.DataFrame(index=tickers, columns=['Size', 'Value', 'Momentum', 'Volatility', 'AI_Sentiment'])

for t in tickers:
    ticker_obj = yf.Ticker(t)
    info = ticker_obj.info
       # size factor is the natural logarithm of the market cap
    market_cap = info.get('marketCap', np.nan)
    X_raw.loc[t, 'Size'] = np.log(market_cap) if market_cap else np.nan
    
    # 调取算 Value 因子的基础数据：账面市值比 (Book-to-Market = 1 / PB)
    price_to_book = info.get('priceToBook', np.nan)
    X_raw.loc[t, 'Value'] = (1.0 / price_to_book) if price_to_book else np.nan

# 核心工业级清洗：Barra 规定必须在横截面上做 Z-Score 标准化，消除不同量纲影响
X_normalized = X_raw.apply(lambda x: (x - x.mean()) / x.std(), axis=0).fillna(0)

print("--- the independent variable X (Barra style factor exposure matrix) is aligned successfully ---")
print(X_normalized)

#==========================================
# barra style factor exposure matrix
#==========================================
# 拿到最新一天的超额收益率作为截面 Y
y_t = Y_excess.iloc[-1].values.reshape(-1, 1) # 形状 (N, 1)
X_t = X_normalized.values.astype(float)       # 形状 (N, M)

# 调取市值作为权重矩阵 W (这里用各个资产的ln(MarketCap)作为WLS的权重权重)
# 真实的 Barra 是用市值平方根，这里用对数市值简化演示
weights = [np.log(yf.Ticker(t).info.get('marketCap')) for t in tickers]
W_t = np.diag(weights) # 构成对角权重矩阵 (N, N)

# 【面试大杀器】不用sklearn，纯数学矩阵手撕 WLS 的闭式解 (Closed-form Solution)
# 公式： beta = (X^T * W * X)^(-1) * X^T * W * y
X_T_W = X_t.T @ W_t
beta = np.linalg.inv(X_T_W @ X_t) @ X_T_W @ y_t

print("\n--- 成功解出当天 Barra 纯风格因子的收益率 (Factor Returns) ---")
for i, factor in enumerate(['Size_Return', 'Value_Return']):
    print(f"{factor}: {beta[i][0]:.6f}")