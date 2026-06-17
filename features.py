# features.py
import numpy as np
import pandas as pd
import yfinance as yf
from config import (
    FACTOR_NAMES, TICKERS, BENCHMARK, START_DATE, END_DATE,
    LOOKBACK_MOM, SKIP_RECENT, LOOKBACK_VOL,
)

def compute_momentum(returns: pd.DataFrame) -> pd.DataFrame:
    """compute the momentum for each ticker"""
    rows = []
    for t in TICKERS:
        r = returns[t]
        for i in range(LOOKBACK_MOM, len(r)):
            date = r.index[i]
            hist = r.iloc[: i + 1] #daily returns up to the current date
            mom_window = hist.iloc[-LOOKBACK_MOM:-SKIP_RECENT] 
            mom = (1 + mom_window).prod() - 1 
            rows.append((date, t, mom))
    df = pd.DataFrame(rows, columns=["date", "ticker", "Momentum"])
    return df.set_index(["date", "ticker"])


def compute_volatility(returns: pd.DataFrame) -> pd.DataFrame:
    """compute the volatility for each ticker"""
    rows = []
    for t in TICKERS:
        r = returns[t]
        for i in range(LOOKBACK_VOL, len(r)):
            date = r.index[i]
            hist = r.iloc[: i + 1]
            vol = hist.iloc[-LOOKBACK_VOL:].std()
            rows.append((date, t, vol))
    df = pd.DataFrame(rows, columns=["date", "ticker", "Volatility"])
    return df.set_index(["date", "ticker"])

def compute_size(close: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in TICKERS:
        price = close[t]
        for i in range(1, len(price)):    # Size can be computed from the first day
            date = price.index[i]
            size = np.log(price.iloc[i]) #natural logarithm of the price
            rows.append((date, t, size))
    df = pd.DataFrame(rows, columns=["date", "ticker", "Size"])
    return df.set_index(["date", "ticker"])


def compute_value(close: pd.DataFrame) -> pd.DataFrame:
    """Value proxy: book_per_share / price (daily B/P)"""
    rows = []
    for t in TICKERS:
        price = close[t].dropna()
        if price.empty:
            continue

        pb = yf.Ticker(t).info.get("priceToBook", np.nan)
        if not pb or pb <= 0:
            continue

        # use the current P/B to estimate the book per share, then divide the daily price to get the daily B/P
        book_per_share = price.iloc[-1] / pb

        for date in price.index:
            value = book_per_share / price.loc[date]
            rows.append((date, t, value))

    df = pd.DataFrame(rows, columns=["date", "ticker", "Value"])
    return df.set_index(["date", "ticker"])

#this is the B/P proxy for free data; in production, we will use the quarterly reports forward-filled.

def build_factor_panel(returns: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
        """merge all the price-related factors"""
        mom = compute_momentum(returns)
        vol = compute_volatility(returns)
        size = compute_size(close)
        value = compute_value(close)
        return mom.join(vol, how="inner").join(size, how="inner").join(value, how="inner")


def zscore_cross_section(panel: pd.DataFrame) -> pd.DataFrame:
    """z-score the cross-section of the panel"""
    return panel.groupby(level="date").transform(
        lambda x: (x - x.mean()) / x.std()
    ).fillna(0)