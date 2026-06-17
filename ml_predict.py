import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from config import (
    FACTOR_NAMES, TICKERS, BENCHMARK, START_DATE, END_DATE,
    FORWARD_DAYS, TRAIN_RATIO,
)
from features import build_factor_panel, zscore_cross_section

#define the functions for the ml

def load_market_data():
    all_assets = TICKERS + [BENCHMARK]
    data = yf.download(all_assets, start=START_DATE, end=END_DATE)
    close = data["Close"]
    returns = close.pct_change().dropna()
    y_excess = returns[TICKERS].sub(returns[BENCHMARK], axis=0)
    return close, returns, y_excess

def forward_excess_return(y_excess: pd.DataFrame, horizon: int) -> pd.Series:
    """Cumulative excess return from t+1 to t+horizon."""
    fwd = sum(y_excess.shift(-k) for k in range(1, horizon + 1))
    target = fwd.stack()
    target.index = target.index.set_names(["date", "ticker"])
    return target.rename(f"target_{horizon}d")

def build_ml_dataset(x_panel: pd.DataFrame, y_excess: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target = forward_excess_return(y_excess, horizon)
    return x_panel.join(target, how="inner").dropna()

def time_split(df: pd.DataFrame, train_ratio: float):
    dates = df.index.get_level_values("date").unique().sort_values()
    split_at = int(len(dates) * train_ratio)
    train_dates = dates[:split_at]
    test_dates = dates[split_at:]
    train = df[df.index.get_level_values("date").isin(train_dates)]
    test = df[df.index.get_level_values("date").isin(test_dates)]
    return train, test, train_dates[-1], test_dates[0]

def rank_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    return y_true.corr(y_pred, method="spearman")

def evaluate_predictions(test: pd.DataFrame, pred: np.ndarray, target_col: str, label: str):
    test = test.copy()
    test["pred"] = pred
    daily_ics, daily_rank_ics = [], []
    for _, group in test.groupby(level="date"):
        if len(group) < 3:
            continue
        daily_ics.append(group["pred"].corr(group[target_col]))
        daily_rank_ics.append(rank_ic(group[target_col], group["pred"]))
    mse = mean_squared_error(test[target_col], pred)
    mean_ic = float(np.nanmean(daily_ics))
    mean_rank_ic = float(np.nanmean(daily_rank_ics))
    print(f"\n--- {label} ---")
    print(f"MSE:          {mse:.6f}")
    print(f"Mean IC:      {mean_ic:.4f}")
    print(f"Mean Rank IC: {mean_rank_ic:.4f}")
    return mean_ic, mean_rank_ic

def main():
    target_col = f"target_{FORWARD_DAYS}d"
    print("Loading market data and building factor panel...")
    close, returns, y_excess = load_market_data()
    x_panel = zscore_cross_section(build_factor_panel(returns, close))
    ml_df = build_ml_dataset(x_panel, y_excess, FORWARD_DAYS)

    train, test, last_train_date, first_test_date = time_split(ml_df, TRAIN_RATIO)
    print("\n--- ML dataset ---")
    print(f"Rows: {len(ml_df)} | Features: {FACTOR_NAMES}")
    print(f"Label:  forward {FORWARD_DAYS}-day excess return")
    print(f"Train:  {len(train)} rows (through {last_train_date.date()})")
    print(f"Test:   {len(test)} rows (from {first_test_date.date()})")

    x_train = train[FACTOR_NAMES].values
    y_train = train[target_col].values
    x_test = test[FACTOR_NAMES].values

    models = {
        "Baseline (predict 0)": None,
        "Momentum only (OLS)": "momentum",
        "Ridge (4 factors)": Ridge(alpha=1.0),
        "RandomForest (4 factors)": RandomForestRegressor(
            n_estimators=100, max_depth=3, random_state=42
        ),
    }

    results = []
    for name, model in models.items():
        if model is None:
            pred = np.zeros(len(x_test))
        elif model == "momentum":
            mom_train = train[["Momentum"]].values
            mom_test = test[["Momentum"]].values
            coef = np.linalg.lstsq(mom_train, y_train, rcond=None)[0]
            pred = mom_test @ coef
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
        mean_ic, mean_rank_ic = evaluate_predictions(test, pred, target_col, name)
        results.append({"model": name, "mean_ic": mean_ic, "mean_rank_ic": mean_rank_ic})

    print("\n--- Model comparison (test set) ---")
    summary = pd.DataFrame(results).set_index("model")
    print(summary.sort_values("mean_rank_ic", ascending=False).to_string(float_format="%.4f"))

if __name__ == "__main__":
    main()