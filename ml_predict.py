import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from config import (
    FACTOR_NAMES, STYLE_FACTOR_NAMES, TICKERS, BENCHMARK, START_DATE, END_DATE,
    FORWARD_DAYS, TRAIN_RATIO, TOP_PCT, COST_BPS, REBALANCE_FREQ,
)
from backtest import next_day_excess, long_short_backtest, summarize_backtest
from features import build_factor_panel, zscore_cross_section
from ai_factor import build_ai_panel

########################################################
#define the functions for the ml
########################################################

#load the market data

def load_market_data():
    all_assets = TICKERS + [BENCHMARK]
    data = yf.download(all_assets, start=START_DATE, end=END_DATE)
    close = data["Close"]
    returns = close.pct_change().dropna()
    y_excess = returns[TICKERS].sub(returns[BENCHMARK], axis=0)
    return close, returns, y_excess

#define the function for the forward excess return

def forward_excess_return(y_excess: pd.DataFrame, horizon: int) -> pd.Series:
    """Cumulative excess return from t+1 to t+horizon."""
    fwd = sum(y_excess.shift(-k) for k in range(1, horizon + 1))
    target = fwd.stack() # switch the ticker and date to the columns
    target.index = target.index.set_names(["date", "ticker"])
    return target.rename(f"target_{horizon}d")

#define the function for the ml dataset

def build_ml_dataset(x_panel: pd.DataFrame, y_excess: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target = forward_excess_return(y_excess, horizon)
    return x_panel.join(target, how="inner").dropna()

#define the function for the time split

def time_split(df: pd.DataFrame, train_ratio: float):
    dates = df.index.get_level_values("date").unique().sort_values()
    split_at = int(len(dates) * train_ratio)
    train_dates = dates[:split_at]
    test_dates = dates[split_at:]
    train = df[df.index.get_level_values("date").isin(train_dates)]
    test = df[df.index.get_level_values("date").isin(test_dates)]
    return train, test, train_dates[-1], test_dates[0]

#define the function for the rank ic

def rank_ic(y_true: pd.Series, y_pred: pd.Series) -> float: 
    return y_true.corr(y_pred, method="spearman") 

    #ic is the correlation between the true and predicted returns
    #spearman correlation is the correlation between the true and predicted returns rank


#define the function for the evaluate predictions

def evaluate_predictions(
    test: pd.DataFrame,
    pred: np.ndarray,
    target_col: str,
    label: str,
    ret_1d: pd.Series | None = None,
    top_pct: float = TOP_PCT,
    cost_bps: float = COST_BPS,
    rebalance_freq: str = REBALANCE_FREQ,
):
    test = test.copy()
    test["pred"] = pred
    daily_ics, daily_rank_ics = [], []
    for _, group in test.groupby(level="date"):
        if len(group) < 3:
            continue
        daily_ics.append(group["pred"].corr(group[target_col])) #correlation between the predicted and true returns pearson correlation
        daily_rank_ics.append(rank_ic(group[target_col], group["pred"]))
    mse = mean_squared_error(test[target_col], pred)
    mean_ic = float(np.nanmean(daily_ics)) if daily_ics else float("nan") #mean of the daily ic
    mean_rank_ic = float(np.nanmean(daily_rank_ics)) if daily_rank_ics else float("nan")
    print(f"\n--- {label} ---")
    print(f"MSE:          {mse:.6f}")
    print(f"Mean IC:      {mean_ic:.4f}")
    print(f"Mean Rank IC: {mean_rank_ic:.4f}")

    bt_stats = {}
    holdings_df = pd.DataFrame()
    if ret_1d is not None:
        daily_ls, holdings_df = long_short_backtest(
            test, pred, ret_1d, top_pct, cost_bps, rebalance_freq
        )
        bt_stats = summarize_backtest(daily_ls)
        print(f"Ann Return:   {bt_stats['ann_return']:.2%}")
        print(f"Sharpe:       {bt_stats['sharpe']:.2f}")
        print(f"Max Drawdown: {bt_stats['max_drawdown']:.2%}")
        print(f"Hit Rate:     {bt_stats['hit_rate']:.2%}")
        if not holdings_df.empty:
            last_date = holdings_df["date"].max()
            print(f"Holdings on last rebalance ({pd.Timestamp(last_date).date()}):")
            print(holdings_df[holdings_df["date"] == last_date].to_string(index=False))

    return mean_ic, mean_rank_ic, bt_stats

#define the main function

def main():
    target_col = f"target_{FORWARD_DAYS}d"
    print("Loading market data and building factor panel...")
    close, returns, y_excess = load_market_data()
    ret_1d = next_day_excess(y_excess)
    ai_panel = build_ai_panel(dates=returns.index)
    x_panel = zscore_cross_section(build_factor_panel(returns, close, ai_panel=ai_panel))
    ml_df = build_ml_dataset(x_panel, y_excess, FORWARD_DAYS)

    train, test, last_train_date, first_test_date = time_split(ml_df, TRAIN_RATIO)
    print("\n--- ML dataset ---")
    print(f"Rows: {len(ml_df)} | Features: {FACTOR_NAMES}")
    print(f"Label:  forward {FORWARD_DAYS}-day excess return")
    print(f"Train:  {len(train)} rows (through {last_train_date.date()})")
    print(f"Test:   {len(test)} rows (from {first_test_date.date()})")

#split the data into train and test

    y_train = train[target_col].values

#define the models
    models = {
        "Baseline (predict 0)": None,
        "Momentum only (OLS)": "momentum",
        "Ridge (4 factors)": ("ridge4", Ridge(alpha=1.0)),
        "Ridge (5 factors)": ("ridge5", Ridge(alpha=1.0)),
        "RandomForest (5 factors)": (
            "rf5",
            RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
        ),
    }

#define the function for the model comparison
    results = []
    for name, model in models.items():
        if model is None:
            pred = np.zeros(len(test))
        elif model == "momentum":
            mom_train = train[["Momentum"]].values
            mom_test = test[["Momentum"]].values
            coef = np.linalg.lstsq(mom_train, y_train, rcond=None)[0]
            pred = mom_test @ coef
        else:
            kind, est = model
            feat_cols = STYLE_FACTOR_NAMES if kind == "ridge4" else FACTOR_NAMES
            est.fit(train[feat_cols].values, y_train)
            pred = est.predict(test[feat_cols].values)
        mean_ic, mean_rank_ic, bt_stats = evaluate_predictions(
            test, pred, target_col, name, ret_1d=ret_1d
        )
        results.append({"model": name, "mean_ic": mean_ic, "mean_rank_ic": mean_rank_ic, **bt_stats})

    print("\n--- Model comparison (test set) ---")
    summary = pd.DataFrame(results).set_index("model")
    print(summary.sort_values("mean_rank_ic", ascending=False).to_string(float_format="%.4f"))

if __name__ == "__main__":
    main()