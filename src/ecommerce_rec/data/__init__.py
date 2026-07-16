from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# Event types mirroring a large-retailer engagement funnel
EVENT_IMPRESSION = "impression"
EVENT_CLICK = "click"
EVENT_ATC = "add_to_cart"
EVENT_PURCHASE = "purchase"

EVENT_WEIGHTS_DEFAULT = {
    EVENT_IMPRESSION: 0.001,
    EVENT_CLICK: 0.01,
    EVENT_ATC: 0.1,
    EVENT_PURCHASE: 1.0,
}


@dataclass
class CatalogTables:
    """Offline warehouse snapshot for PDP personalization."""

    products: pd.DataFrame
    users: pd.DataFrame
    events: pd.DataFrame
    # Precomputed co-occurrence edges used for similar-item style anchors
    co_purchase: pd.DataFrame


REQUIRED_PRODUCT_COLS = (
    "item_id",
    "title",
    "brand_id",
    "category_id",
    "price",
    "price_band",
    "avg_rating",
    "n_reviews",
    "in_stock",
)

REQUIRED_EVENT_COLS = (
    "event_id",
    "user_id",
    "item_id",
    "anchor_item_id",
    "event_type",
    "timestamp",
    "session_id",
)
