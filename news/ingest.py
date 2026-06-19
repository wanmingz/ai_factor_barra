# news/ingest.py — fetch and normalize yfinance news per investment theme

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import yfinance as yf

from config import NEWS_LOOKBACK_DAYS, NEWS_MAX_PER_TICKER, THEMES_FILE
from theme_agent import load_theme_map, list_themes


def theme_ticker_map(path: str = THEMES_FILE) -> dict[str, list[str]]:
    """Map each theme to the list of universe tickers with that exposure."""
    df = load_theme_map(path)
    return {
        theme: sorted(df.loc[df["theme"] == theme, "ticker"].unique().tolist())
        for theme in list_themes(path)
    }


def _parse_publish_time(raw) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    if isinstance(raw, str):
        try:
            return pd.Timestamp(raw).tz_localize("UTC")
        except (TypeError, ValueError):
            return None
    return None


def _normalize_news_item(item: dict) -> dict | None:
    """Support yfinance news v1 (flat) and v2 (nested under 'content')."""
    if not item:
        return None

    payload = item.get("content") if isinstance(item.get("content"), dict) else item
    title = (payload.get("title") or "").strip()
    if not title:
        return None

    provider = payload.get("provider") or {}
    publisher = (
        payload.get("publisher")
        or payload.get("publisherName")
        or provider.get("displayName")
        or ""
    )

    link = payload.get("link") or ""
    if not link:
        click = payload.get("clickThroughUrl") or payload.get("canonicalUrl") or {}
        if isinstance(click, dict):
            link = click.get("url") or ""

    pub_raw = (
        payload.get("providerPublishTime")
        or payload.get("publishedAt")
        or payload.get("pubDate")
        or payload.get("displayTime")
    )

    return {
        "title": title,
        "publisher": publisher,
        "link": link,
        "published_at": _parse_publish_time(pub_raw),
    }


def fetch_ticker_news(ticker: str, max_items: int = NEWS_MAX_PER_TICKER) -> list[dict]:
    """Return recent news dicts for one ticker (title used as RAG document)."""
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []

    docs = []
    for item in items[:max_items]:
        normalized = _normalize_news_item(item)
        if not normalized:
            continue
        docs.append({"ticker": ticker, **normalized})
    return docs


def fetch_theme_news(
    themes: list[str] | None = None,
    as_of: str | date | None = None,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
    themes_file: str = THEMES_FILE,
) -> dict[str, list[dict]]:
    """
    Fetch news for all tickers under each theme, dedupe by title, filter by date window.

    Returns {theme: [{theme, ticker, title, publisher, link, published_at}, ...]}.
    """
    as_of_ts = pd.Timestamp(as_of or date.today()).normalize()
    cutoff = (as_of_ts - pd.Timedelta(days=lookback_days)).tz_localize("UTC")
    as_of_end = (as_of_ts + pd.Timedelta(days=1)).tz_localize("UTC")
    mapping = theme_ticker_map(themes_file)
    target_themes = themes or list(mapping.keys())

    out: dict[str, list[dict]] = {}
    for theme in target_themes:
        seen_titles: set[str] = set()
        theme_docs: list[dict] = []

        for ticker in mapping.get(theme, []):
            for doc in fetch_ticker_news(ticker):
                pub = doc.get("published_at")
                if pub is not None:
                    pub_ts = pd.Timestamp(pub)
                    if pub_ts.tzinfo is None:
                        pub_ts = pub_ts.tz_localize("UTC")
                    else:
                        pub_ts = pub_ts.tz_convert("UTC")
                    if pub_ts > as_of_end:
                        continue
                    if pub_ts < cutoff:
                        continue

                key = doc["title"].lower()
                if key in seen_titles:
                    continue
                seen_titles.add(key)

                theme_docs.append(
                    {
                        "theme": theme,
                        "ticker": doc["ticker"],
                        "title": doc["title"],
                        "publisher": doc["publisher"],
                        "link": doc["link"],
                        "published_at": pub.isoformat() if pub else None,
                    }
                )

        out[theme] = theme_docs
    return out


if __name__ == "__main__":
    import json

    as_of = date.today().isoformat()
    news = fetch_theme_news(as_of=as_of)
    for theme, docs in news.items():
        print(f"\n{theme} ({len(docs)} articles)")
        for d in docs[:3]:
            print(f"  [{d['ticker']}] {d['title'][:80]}")
