# ai_factor.py — Map theme scores to stock-level AI exposure for Barra / ML pipelines.
#
# Usage:
#   python ai_factor.py                     # build & cache mock AI panel
#   python ai_factor.py --mode auto         # Gemini JSON when available, else mock
#   python ai_factor.py --date 2026-06-19   # single-day exposure snapshot

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    TICKERS,
    START_DATE,
    END_DATE,
    THEMES_FILE,
    THEME_ETF_PROXY,
    THEME_CONTEXT_LOOKBACK,
    THEME_SCORES_DIR,
    LOOKBACK_MOM,
    AI_FACTOR_CACHE,
    AI_SCORE_MODE,
    AI_LAG_DAYS,
)
from theme_agent import (
    load_theme_map,
    list_themes,
    load_theme_scores,
    score_themes_mock,
    fetch_theme_context,
)


def theme_scores_to_exposure(
    scores: dict[str, float],
    theme_map: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Map theme scores to per-ticker raw AI exposure.

    Raw_AI(i) = sum_k weight(i, k) * ThemeScore(k)
    """
    df = theme_map if theme_map is not None else load_theme_map()
    mapped = df.copy()
    mapped["score"] = mapped["theme"].map(scores)
    if mapped["score"].isna().any():
        missing = mapped.loc[mapped["score"].isna(), "theme"].unique().tolist()
        raise ValueError(f"Missing theme scores for: {missing}")

    mapped["weighted"] = mapped["weight"] * mapped["score"]
    exposure = mapped.groupby("ticker")["weighted"].sum()
    return exposure.reindex(TICKERS).dropna()


def _theme_score_json_path(as_of: str) -> Path:
    return Path(THEME_SCORES_DIR) / f"{as_of}.json"


def load_or_compute_theme_scores(
    as_of: str,
    themes: list[str] | None = None,
    mode: str = AI_SCORE_MODE,
) -> dict[str, float]:
    """Load saved scores for a date, or compute mock scores."""
    themes = themes or list_themes()
    if mode == "auto" and _theme_score_json_path(as_of).is_file():
        return load_theme_scores(as_of)

    context = fetch_theme_context(themes, as_of)
    return score_themes_mock(themes, context)


def build_mock_theme_score_panel(
    dates: pd.DatetimeIndex | list,
    themes: list[str] | None = None,
) -> pd.DataFrame:
    """
    Batch-compute mock theme scores for many dates (ETF 3m momentum z-scored).

    Returns DataFrame indexed by date, columns = theme names.
    """
    themes = themes or list_themes()
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
    if len(dates) == 0:
        return pd.DataFrame(columns=themes)

    etfs = sorted(set(THEME_ETF_PROXY.values()))
    start = (dates.min() - pd.Timedelta(days=THEME_CONTEXT_LOOKBACK * 2)).strftime("%Y-%m-%d")
    end = (dates.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    hist = yf.download(etfs, start=start, end=end, progress=False, auto_adjust=True)
    if hist.empty:
        raise RuntimeError("Failed to download theme ETF history")

    close = hist["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(etf=etfs[0])

    rows: list[dict[str, float]] = []
    for date in dates:
        context: dict[str, dict] = {}
        for theme in themes:
            etf = THEME_ETF_PROXY.get(theme)
            if not etf or etf not in close.columns:
                context[theme] = {"return_3m": 0.0}
                continue

            series = close[etf].dropna()
            series = series[series.index <= date]
            if len(series) < THEME_CONTEXT_LOOKBACK:
                r3 = 0.0
            else:
                r3 = float(
                    series.iloc[-1]
                    / series.iloc[-min(THEME_CONTEXT_LOOKBACK, len(series))]
                    - 1
                )
            context[theme] = {"return_3m": r3}

        rows.append(score_themes_mock(themes, context))

    return pd.DataFrame(rows, index=dates).reindex(columns=themes)


def build_theme_score_panel(
    dates: pd.DatetimeIndex | list,
    themes: list[str] | None = None,
    mode: str = AI_SCORE_MODE,
) -> pd.DataFrame:
    """
    Build daily theme score panel.

    mode:
      - mock: ETF proxy only
      - auto: use theme_scores/*.json when present, else mock for that date
    """
    themes = themes or list_themes()
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
    mock_panel = build_mock_theme_score_panel(dates, themes)

    if mode == "mock":
        return mock_panel

    rows = []
    for date in dates:
        as_of = pd.Timestamp(date).strftime("%Y-%m-%d")
        if _theme_score_json_path(as_of).is_file():
            scores = load_theme_scores(as_of)
            rows.append({t: scores.get(t, 0.0) for t in themes})
        else:
            rows.append(mock_panel.loc[date].to_dict())

    return pd.DataFrame(rows, index=dates).reindex(columns=themes)


def build_ai_exposure_panel(
    dates: pd.DatetimeIndex | list,
    theme_score_panel: pd.DataFrame | None = None,
    theme_map: pd.DataFrame | None = None,
    mode: str = AI_SCORE_MODE,
) -> pd.DataFrame:
    """
    Build (date, ticker) AI raw exposure panel with column 'AI'.
    """
    themes = list_themes()
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
    theme_map = theme_map if theme_map is not None else load_theme_map()
    score_panel = (
        theme_score_panel
        if theme_score_panel is not None
        else build_theme_score_panel(dates, themes, mode=mode)
    )

    rows = []
    for date in dates:
        if date not in score_panel.index:
            continue
        scores = score_panel.loc[date].to_dict()
        exposure = theme_scores_to_exposure(scores, theme_map)
        for ticker, value in exposure.items():
            rows.append((date, ticker, value))

    if not rows:
        return pd.DataFrame(columns=["AI"]).set_index(["date", "ticker"])

    df = pd.DataFrame(rows, columns=["date", "ticker", "AI"])
    return df.set_index(["date", "ticker"]).sort_index()


def lag_ai_exposure(panel: pd.DataFrame, days: int = AI_LAG_DAYS) -> pd.DataFrame:
    """
    Lag AI exposure by N trading days per ticker.

    On date t, AI reflects the score from t-N (point-in-time: score after close at t-N
    is tradable from t-N+1; with N=1, date t uses score from t-1).
    """
    if days <= 0 or panel.empty or "AI" not in panel.columns:
        return panel

    wide = panel["AI"].unstack("ticker")
    lagged = wide.shift(days)
    out = lagged.stack(future_stack=True).to_frame("AI")
    out.index = out.index.set_names(["date", "ticker"])
    return out.sort_index()


def load_ai_panel(cache_path: str | Path = AI_FACTOR_CACHE) -> pd.DataFrame:
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"No cached AI panel at {path}. Run build_ai_panel() first.")
    return pd.read_parquet(path)


def save_ai_panel(panel: pd.DataFrame, cache_path: str | Path = AI_FACTOR_CACHE) -> Path:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path)
    return path


def build_ai_panel(
    dates: pd.DatetimeIndex | list | None = None,
    *,
    mode: str = AI_SCORE_MODE,
    use_cache: bool = True,
    cache_path: str | Path = AI_FACTOR_CACHE,
    lag_days: int | None = None,
) -> pd.DataFrame:
    """
    Load cached AI exposure panel or build from theme scores.

    If dates is None, uses START_DATE..END_DATE trading window aligned with momentum panel.
    Applies AI_LAG_DAYS shift on return (cache stores unlagged scores).
    """
    lag = AI_LAG_DAYS if lag_days is None else lag_days

    def _with_lag(df: pd.DataFrame) -> pd.DataFrame:
        return lag_ai_exposure(df, days=lag)

    path = Path(cache_path)
    req_dates = (
        pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
        if dates is not None
        else None
    )

    if use_cache and path.is_file():
        panel = load_ai_panel(path)
        if req_dates is None:
            return _with_lag(panel)
        cached_dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique())
        if cached_dates.isin(req_dates).all() and req_dates.isin(cached_dates).all():
            mask = panel.index.get_level_values("date").isin(req_dates)
            return _with_lag(panel[mask].sort_index())

    if req_dates is None:
        tickers_data = yf.download(
            TICKERS[:1],
            start=START_DATE,
            end=END_DATE,
            progress=False,
            auto_adjust=True,
        )
        all_dates = tickers_data.index
        req_dates = all_dates[LOOKBACK_MOM:]

    panel = build_ai_exposure_panel(req_dates, mode=mode)
    if use_cache:
        save_ai_panel(panel, path)
    return _with_lag(panel)


def snapshot_exposure(as_of: str, mode: str = AI_SCORE_MODE) -> pd.Series:
    """Single-date raw AI exposure for all mapped tickers."""
    scores = load_or_compute_theme_scores(as_of, mode=mode)
    return theme_scores_to_exposure(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stock-level AI theme exposure panel")
    parser.add_argument("--mode", default=AI_SCORE_MODE, choices=["auto", "mock"])
    parser.add_argument("--date", default=None, help="Print single-day exposure YYYY-MM-DD")
    parser.add_argument("--rebuild", action="store_true", help="Ignore cache and rebuild panel")
    args = parser.parse_args()

    if args.date:
        exposure = snapshot_exposure(args.date, mode=args.mode)
        print(f"\n--- AI raw exposure on {args.date} ({args.mode}) ---")
        ranked = exposure.sort_values(ascending=False)
        for ticker, val in ranked.items():
            print(f"  {ticker:8s}  {val:+.4f}")
        return

    print(f"Building AI exposure panel (mode={args.mode})...")
    panel = build_ai_panel(mode=args.mode, use_cache=not args.rebuild)
    print(f"Rows: {len(panel)} | Dates: {panel.index.get_level_values('date').nunique()}")
    print(f"Saved to {AI_FACTOR_CACHE}")
    print("\nSample (last date, top 5):")
    last_date = panel.index.get_level_values("date").max()
    top = panel.xs(last_date, level="date").sort_values("AI", ascending=False).head()
    print(top.to_string())


if __name__ == "__main__":
    main()
