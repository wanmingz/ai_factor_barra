"""Multi-asset scoring agent — cross-sectional views on global asset classes."""

from multi_asset.universe import (
    asset_classes_by_category,
    list_asset_classes,
    list_categories,
    load_asset_classes,
    proxy_map,
)

__all__ = [
    "list_asset_classes",
    "list_categories",
    "load_asset_classes",
    "asset_classes_by_category",
    "proxy_map",
]
