# multi_asset/ingest.py — fetch macro news via ETF/FX proxies per asset class

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import yfinance as yf

from config import (
    ASSET_CLASSES_FILE,
    MULTI_ASSET_NEWS_MAX_PER_PROXY,
    NEWS_LOOKBACK_DAYS,
)
from multi_asset.universe import load_asset_classes


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


def fetch_proxy_news(
    symbol: str,
    max_items: int = MULTI_ASSET_NEWS_MAX_PER_PROXY,
) -> list[dict]:
    """Return recent news for an ETF proxy (FX pairs typically have no news)."""
    if symbol.endswith("=X"):
        return []

    try:
        items = yf.Ticker(symbol).news or []
    except Exception:
        return []

    docs = []
    for item in items[:max_items]:
        normalized = _normalize_news_item(item)
        if not normalized:
            continue
        docs.append({"proxy": symbol, **normalized})
    return docs


def fetch_asset_class_news(
    asset_classes: list[str] | None = None,
    as_of: str | date | None = None,
    lookback_days: int = NEWS_LOOKBACK_DAYS,
    classes_file: str = ASSET_CLASSES_FILE,
) -> dict[str, list[dict]]:
    """
    Fetch news for each asset class via its ETF proxy, dedupe by title, filter by date.

    Returns {asset_class: [{asset_class, proxy, title, publisher, link, published_at}, ...]}.
    """
    df = load_asset_classes(classes_file)
    if asset_classes is not None:
        df = df[df["asset_class"].isin(asset_classes)]

    as_of_ts = pd.Timestamp(as_of or date.today()).normalize()
    cutoff = (as_of_ts - pd.Timedelta(days=lookback_days)).tz_localize("UTC")
    as_of_end = (as_of_ts + pd.Timedelta(days=1)).tz_localize("UTC")

    proxy_cache: dict[str, list[dict]] = {}
    out: dict[str, list[dict]] = {}

    for _, row in df.iterrows():
        ac = row["asset_class"]
        proxy = row["proxy_symbol"]
        if proxy not in proxy_cache:
            proxy_cache[proxy] = fetch_proxy_news(proxy)

        seen_titles: set[str] = set()
        class_docs: list[dict] = []

        for doc in proxy_cache[proxy]:
            pub = doc.get("published_at")
            if pub is not None:
                pub_ts = pd.Timestamp(pub)
                if pub_ts.tzinfo is None:
                    pub_ts = pub_ts.tz_localize("UTC")
                else:
                    pub_ts = pub_ts.tz_convert("UTC")
                if pub_ts > as_of_end or pub_ts < cutoff:
                    continue

            key = doc["title"].lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)

            class_docs.append(
                {
                    "asset_class": ac,
                    "proxy": proxy,
                    "title": doc["title"],
                    "publisher": doc["publisher"],
                    "link": doc["link"],
                    "published_at": pub.isoformat() if pub else None,
                }
            )

        out[ac] = class_docs

    return out
