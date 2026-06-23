# multi_asset/universe.py — asset class definitions and ETF/FX proxy map

from __future__ import annotations

import pandas as pd

from config import ASSET_CLASSES_FILE

_REQUIRED = {"asset_class", "category", "proxy_symbol", "proxy_type", "rag_query"}


def load_asset_classes(path: str = ASSET_CLASSES_FILE) -> pd.DataFrame:
    """Load asset class universe from CSV."""
    df = pd.read_csv(path)
    if not _REQUIRED.issubset(df.columns):
        raise ValueError(f"{path} must have columns: {sorted(_REQUIRED)}")
    return df


def list_asset_classes(path: str = ASSET_CLASSES_FILE) -> list[str]:
    return load_asset_classes(path)["asset_class"].tolist()


def list_categories(path: str = ASSET_CLASSES_FILE) -> list[str]:
    return sorted(load_asset_classes(path)["category"].unique().tolist())


def asset_classes_by_category(path: str = ASSET_CLASSES_FILE) -> dict[str, list[str]]:
    df = load_asset_classes(path)
    return {
        cat: df.loc[df["category"] == cat, "asset_class"].tolist()
        for cat in sorted(df["category"].unique())
    }


def proxy_map(path: str = ASSET_CLASSES_FILE) -> dict[str, dict]:
    """Map asset_class -> {proxy_symbol, proxy_type, category, rag_query}."""
    df = load_asset_classes(path)
    return {
        row["asset_class"]: {
            "proxy_symbol": row["proxy_symbol"],
            "proxy_type": row["proxy_type"],
            "category": row["category"],
            "rag_query": row["rag_query"],
        }
        for _, row in df.iterrows()
    }
