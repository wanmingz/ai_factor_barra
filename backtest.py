import numpy as np
import pandas as pd


def next_day_excess(y_excess: pd.DataFrame) -> pd.Series:
    """Next-day excess return per (date, ticker) — used for daily LS backtest PnL."""
    ret = y_excess.shift(-1).stack()
    ret.index = ret.index.set_names(["date", "ticker"])
    return ret.rename("ret_1d")


def _month_start_dates(dates: pd.DatetimeIndex) -> set:
    """First trading day of each calendar month."""
    s = pd.Series(dates, index=dates)
    return set(s.groupby(s.index.to_period("M")).first().values)


def long_short_backtest(
    test: pd.DataFrame,
    pred: np.ndarray,
    ret_1d: pd.Series,
    top_pct: float = 0.2,
    cost_bps: float = 0.0,
    rebalance_freq: str = "monthly",
) -> tuple[pd.Series, pd.DataFrame]:
    """Long-short excess return: long top_pct, short bottom_pct, equal-weighted.

    Returns daily LS returns and a holdings log (one row per ticker per rebalance day).

    rebalance_freq: "monthly" — rebalance on the first trading day of each month;
                    "daily" — rebalance every day.
    """
    df = test.copy()
    df["pred"] = pred
    df = df.join(ret_1d, how="inner")

    all_dates = df.index.get_level_values("date").unique().sort_values()
    rebalance_dates = (
        _month_start_dates(all_dates) if rebalance_freq == "monthly" else set(all_dates)
    )

    long_tickers, short_tickers = None, None
    daily_ls, dates, holdings = [], [], []

    for date in all_dates:
        group = df.xs(date, level="date")
        n = len(group)
        k = max(1, int(n * top_pct))

        if date in rebalance_dates or long_tickers is None:
            ranked = group.sort_values("pred", ascending=False)
            long_tickers = ranked.head(k).index.tolist()
            short_tickers = ranked.tail(k).index.tolist()
            for ticker in long_tickers:
                holdings.append({
                    "date": date, "side": "long", "ticker": ticker,
                    "pred": group.loc[ticker, "pred"],
                })
            for ticker in short_tickers:
                holdings.append({
                    "date": date, "side": "short", "ticker": ticker,
                    "pred": group.loc[ticker, "pred"],
                })
            cost = (4 * k / n) * (cost_bps / 10_000) if cost_bps > 0 else 0.0
        else:
            cost = 0.0

        long_ret = group.loc[group.index.isin(long_tickers), "ret_1d"].mean()
        short_ret = group.loc[group.index.isin(short_tickers), "ret_1d"].mean()
        daily_ls.append(long_ret - short_ret - cost)
        dates.append(date)

    holdings_df = pd.DataFrame(holdings)
    daily_ls_series = pd.Series(daily_ls, index=dates, name="ls_return")
    return daily_ls_series, holdings_df


def summarize_backtest(daily_ls: pd.Series) -> dict:
    """Annualized return, Sharpe, max drawdown, and hit rate from daily LS returns."""
    if daily_ls.empty:
        return {"ann_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "hit_rate": np.nan}

    ann = 252
    cum = (1 + daily_ls).cumprod()
    total = cum.iloc[-1] - 1
    ann_ret = (1 + total) ** (ann / len(daily_ls)) - 1
    sharpe = daily_ls.mean() / daily_ls.std() * np.sqrt(ann) if daily_ls.std() > 0 else 0.0
    max_dd = (cum / cum.cummax() - 1).min()
    return {
        "ann_return": float(ann_ret),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "hit_rate": float((daily_ls > 0).mean()),
    }
