"""Stage-2 feature engineering: retrieval signals + Customer Understanding + item-item context."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ecommerce_rec.data import EVENT_ATC, EVENT_CLICK, EVENT_PURCHASE


POSITIVE_EVENTS = {EVENT_CLICK, EVENT_ATC, EVENT_PURCHASE}


@dataclass
class CustomerUnderstanding:
    """SPIR-style explicit prefs: brand affinity + price affinity per category."""

    # user_id -> brand_id -> score
    brand_affinity: dict[int, dict[int, float]]
    # user_id -> preferred price band (0-4)
    price_affinity: dict[int, int]
    # user interaction counts
    train_counts: dict[int, int]


def build_customer_understanding(train_events: pd.DataFrame, products: pd.DataFrame) -> CustomerUnderstanding:
    pos = train_events[train_events["event_type"].isin(POSITIVE_EVENTS)]
    joined = pos.merge(products[["item_id", "brand_id", "category_id", "price_band"]], on="item_id", how="left")

    brand_affinity: dict[int, dict[int, float]] = {}
    # TF-style frequency within category: count interactions with brand
    for (uid, brand), g in joined.groupby(["user_id", "brand_id"]):
        brand_affinity.setdefault(int(uid), {})[int(brand)] = float(len(g))
    # L2 normalize per user
    for uid, d in brand_affinity.items():
        norm = float(np.sqrt(sum(v * v for v in d.values()))) or 1.0
        brand_affinity[uid] = {b: v / norm for b, v in d.items()}

    price_affinity: dict[int, int] = {}
    for uid, g in joined.groupby("user_id"):
        price_affinity[int(uid)] = int(g["price_band"].mode().iloc[0]) if len(g) else 2

    train_counts = pos.groupby("user_id").size().astype(int).to_dict()
    train_counts = {int(k): int(v) for k, v in train_counts.items()}
    return CustomerUnderstanding(brand_affinity, price_affinity, train_counts)


def build_co_purchase_lookup(co: pd.DataFrame) -> dict[tuple[int, int], float]:
    if len(co) == 0:
        return {}
    mx = float(co["count"].max()) or 1.0
    out: dict[tuple[int, int], float] = {}
    for r in co.itertuples(index=False):
        a, b, c = int(r.item_a), int(r.item_b), float(r.count) / mx
        out[(a, b)] = c
        out[(b, a)] = c
    return out


FEATURE_NAMES = [
    "retrieval_score",
    "retrieval_rank",
    "brand_affinity",
    "price_affinity_match",
    "price_band_delta",
    "same_category",
    "same_brand",
    "co_purchase_score",
    "log_price",
    "avg_rating",
    "log_n_reviews",
    "in_stock",
    "log_user_train_cnt",
    "anchor_price_delta",
]


def _as_item_index(products: pd.DataFrame) -> pd.DataFrame:
    if products.index.name == "item_id":
        return products
    if "item_id" in products.columns:
        return products.set_index("item_id")
    return products


def assemble_pair_features(
    user_id: int,
    anchor_id: int,
    cand_id: int,
    retrieval_score: float,
    retrieval_rank: int,
    products: pd.DataFrame,
    cu: CustomerUnderstanding,
    co_lookup: dict[tuple[int, int], float],
) -> list[float]:
    prod = _as_item_index(products)
    cand = prod.loc[cand_id]
    anchor = prod.loc[anchor_id]

    brand_aff = cu.brand_affinity.get(user_id, {}).get(int(cand["brand_id"]), 0.0)
    user_price = cu.price_affinity.get(user_id, 2)
    price_match = 1.0 if int(cand["price_band"]) == user_price else 0.0
    price_delta = abs(int(cand["price_band"]) - int(anchor["price_band"]))
    same_cat = 1.0 if int(cand["category_id"]) == int(anchor["category_id"]) else 0.0
    same_brand = 1.0 if int(cand["brand_id"]) == int(anchor["brand_id"]) else 0.0
    co = co_lookup.get((anchor_id, cand_id), 0.0)
    train_cnt = cu.train_counts.get(user_id, 0)

    return [
        float(retrieval_score),
        float(retrieval_rank),
        float(brand_aff),
        float(price_match),
        float(price_delta),
        float(same_cat),
        float(same_brand),
        float(co),
        float(np.log1p(cand["price"])),
        float(cand["avg_rating"]),
        float(np.log1p(cand["n_reviews"])),
        float(cand["in_stock"]),
        float(np.log1p(train_cnt)),
        float(abs(float(cand["price"]) - float(anchor["price"]))),
    ]
