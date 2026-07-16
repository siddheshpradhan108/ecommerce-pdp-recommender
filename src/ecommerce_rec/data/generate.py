"""Synthetic large-retailer catalog and PDP engagement logs.

Why synthetic?
  Production behavioral logs are proprietary. This generator produces a
  structurally faithful proxy: category/brand catalog, funnel events, session-
  anchored PDP views, and co-purchase graphs — enough to train Two-Tower + LGBM
  end-to-end and demo production architecture choices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecommerce_rec.config import DataConfig, ProjectConfig
from ecommerce_rec.data import (
    EVENT_ATC,
    EVENT_CLICK,
    EVENT_IMPRESSION,
    EVENT_PURCHASE,
    CatalogTables,
)

CATEGORY_NAMES = [
    "Grocery",
    "Electronics",
    "Home",
    "Apparel",
    "Beauty",
    "Toys",
    "Sports",
    "Pharmacy",
    "Baby",
    "Pets",
    "Automotive",
    "Office",
]


def _make_products(cfg: DataConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_items
    cat = rng.integers(0, cfg.n_categories, size=n)
    brand = rng.integers(0, cfg.n_brands, size=n)
    # Log-normal prices with category bias
    base = rng.lognormal(mean=2.8, sigma=0.7, size=n)
    cat_mult = 1.0 + 0.15 * (cat % 5)
    price = np.round(base * cat_mult, 2)
    # Price bands within category (SPIR-style 5 buckets)
    price_band = np.zeros(n, dtype=np.int32)
    for c in range(cfg.n_categories):
        mask = cat == c
        if mask.sum() == 0:
            continue
        qs = np.quantile(price[mask], [0.2, 0.4, 0.6, 0.8])
        bands = np.digitize(price[mask], qs)
        price_band[mask] = bands

    titles = [
        f"{CATEGORY_NAMES[c % len(CATEGORY_NAMES)]} Item {i} Brand{b}"
        for i, (c, b) in enumerate(zip(cat, brand))
    ]
    return pd.DataFrame(
        {
            "item_id": np.arange(n, dtype=np.int32),
            "title": titles,
            "brand_id": brand.astype(np.int32),
            "category_id": cat.astype(np.int32),
            "price": price.astype(np.float32),
            "price_band": price_band,
            "avg_rating": np.clip(rng.normal(3.9, 0.55, n), 1.0, 5.0).astype(np.float32),
            "n_reviews": rng.integers(0, 5000, n).astype(np.int32),
            "in_stock": rng.random(n) > 0.05,
        }
    )


def _make_users(cfg: DataConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_users
    # Latent preference vectors over categories / brands (Customer Understanding prior)
    preferred_category = rng.integers(0, cfg.n_categories, size=n)
    preferred_brand = rng.integers(0, cfg.n_brands, size=n)
    price_affinity = rng.integers(0, 5, size=n)  # preferred price band
    return pd.DataFrame(
        {
            "user_id": np.arange(n, dtype=np.int32),
            "preferred_category": preferred_category.astype(np.int32),
            "preferred_brand": preferred_brand.astype(np.int32),
            "price_affinity": price_affinity.astype(np.int32),
            "activity_level": rng.choice(["sparse", "medium", "dense"], n, p=[0.35, 0.45, 0.20]),
        }
    )


def _user_n_events(activity: str, rng: np.random.Generator) -> int:
    if activity == "sparse":
        return int(rng.integers(5, 12))
    if activity == "medium":
        return int(rng.integers(12, 35))
    return int(rng.integers(35, 80))


def _sample_item_for_user(
    user: pd.Series,
    products: pd.DataFrame,
    rng: np.random.Generator,
    p_pref_cat: float = 0.55,
    p_pref_brand: float = 0.25,
) -> int:
    """Biased item sampling — users prefer certain categories/brands (ground truth)."""
    if rng.random() < p_pref_cat:
        pool = products[products["category_id"] == user["preferred_category"]]
        if len(pool) == 0:
            pool = products
    else:
        pool = products
    if rng.random() < p_pref_brand and len(pool) > 0:
        branded = pool[pool["brand_id"] == user["preferred_brand"]]
        if len(branded) > 0:
            pool = branded
    # Soft preference for matching price band
    if len(pool) > 10 and rng.random() < 0.4:
        near = pool[np.abs(pool["price_band"] - user["price_affinity"]) <= 1]
        if len(near) > 0:
            pool = near
    if len(pool) == 0:
        pool = products
    return int(pool.sample(1, random_state=int(rng.integers(0, 1_000_000)))["item_id"].iloc[0])


def _generate_events(
    cfg: DataConfig,
    products: pd.DataFrame,
    users: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate PDP sessions: user views anchor, then interacts with recommendations / catalog."""
    rows: list[dict] = []
    co_counts: dict[tuple[int, int], int] = {}
    event_id = 0
    t0 = pd.Timestamp("2024-01-01")
    prod_by_id = products.set_index("item_id")

    for _, user in users.iterrows():
        n_ev = _user_n_events(user["activity_level"], rng)
        # Build a rough chronological basket of items the user would like
        liked = [_sample_item_for_user(user, products, rng) for _ in range(max(3, n_ev // 2))]
        session_counter = 0
        for i in range(n_ev):
            session_counter += 1 if rng.random() < 0.35 else 0
            session_id = f"u{user['user_id']}_s{session_counter}"
            anchor = liked[i % len(liked)]
            # Candidate: complementary or similar (same cat / co-purchase style)
            if rng.random() < 0.6:
                cat_id = int(prod_by_id.loc[anchor, "category_id"])
                same_cat = products[products["category_id"] == cat_id]
                if len(same_cat) == 0:
                    same_cat = products
                candidate = int(
                    same_cat.sample(1, random_state=int(rng.integers(0, 1_000_000)))["item_id"].iloc[0]
                )
            else:
                candidate = _sample_item_for_user(user, products, rng)

            # Funnel: impressions always; click/ATC/purchase cascade
            ts = t0 + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 180)))
            for etype in (EVENT_IMPRESSION,):
                rows.append(
                    {
                        "event_id": event_id,
                        "user_id": int(user["user_id"]),
                        "item_id": candidate,
                        "anchor_item_id": anchor,
                        "event_type": etype,
                        "timestamp": ts,
                        "session_id": session_id,
                    }
                )
                event_id += 1

            # Probability of deeper engagement rises if prefs align
            cand_row = prod_by_id.loc[candidate]
            align = (
                float(cand_row["category_id"] == user["preferred_category"])
                + float(cand_row["brand_id"] == user["preferred_brand"])
                + float(abs(cand_row["price_band"] - user["price_affinity"]) <= 1)
            ) / 3.0
            p_click = 0.15 + 0.45 * align
            p_atc = 0.08 + 0.35 * align
            p_buy = 0.03 + 0.25 * align

            if rng.random() < p_click:
                rows.append(
                    {
                        "event_id": event_id,
                        "user_id": int(user["user_id"]),
                        "item_id": candidate,
                        "anchor_item_id": anchor,
                        "event_type": EVENT_CLICK,
                        "timestamp": ts + pd.Timedelta(seconds=5),
                        "session_id": session_id,
                    }
                )
                event_id += 1
            if rng.random() < p_atc:
                rows.append(
                    {
                        "event_id": event_id,
                        "user_id": int(user["user_id"]),
                        "item_id": candidate,
                        "anchor_item_id": anchor,
                        "event_type": EVENT_ATC,
                        "timestamp": ts + pd.Timedelta(seconds=20),
                        "session_id": session_id,
                    }
                )
                event_id += 1
            if rng.random() < p_buy:
                rows.append(
                    {
                        "event_id": event_id,
                        "user_id": int(user["user_id"]),
                        "item_id": candidate,
                        "anchor_item_id": anchor,
                        "event_type": EVENT_PURCHASE,
                        "timestamp": ts + pd.Timedelta(minutes=2),
                        "session_id": session_id,
                    }
                )
                event_id += 1
                a, b = sorted((anchor, candidate))
                if a != b:
                    co_counts[(a, b)] = co_counts.get((a, b), 0) + 1

    events = pd.DataFrame(rows)
    co = pd.DataFrame(
        [{"item_a": a, "item_b": b, "count": c} for (a, b), c in co_counts.items()]
    )
    if len(co) == 0:
        co = pd.DataFrame(columns=["item_a", "item_b", "count"])
    return events, co


def generate_catalog(cfg: DataConfig) -> CatalogTables:
    rng = np.random.default_rng(cfg.seed)
    products = _make_products(cfg, rng)
    users = _make_users(cfg, rng)
    events, co = _generate_events(cfg, products, users, rng)
    # Drop ultra-sparse users
    counts = events.groupby("user_id").size()
    keep = counts[counts >= cfg.min_events_per_user].index
    events = events[events["user_id"].isin(keep)].reset_index(drop=True)
    users = users[users["user_id"].isin(keep)].reset_index(drop=True)
    return CatalogTables(products=products, users=users, events=events, co_purchase=co)


def persist_catalog(tables: CatalogTables, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables.products.to_parquet(out_dir / "products.parquet", index=False)
    tables.users.to_parquet(out_dir / "users.parquet", index=False)
    tables.events.to_parquet(out_dir / "events.parquet", index=False)
    tables.co_purchase.to_parquet(out_dir / "co_purchase.parquet", index=False)


def load_catalog(data_dir: Path) -> CatalogTables:
    processed = data_dir / "processed"
    return CatalogTables(
        products=pd.read_parquet(processed / "products.parquet"),
        users=pd.read_parquet(processed / "users.parquet"),
        events=pd.read_parquet(processed / "events.parquet"),
        co_purchase=pd.read_parquet(processed / "co_purchase.parquet"),
    )


def temporal_split(events: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = events.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(events) * (1.0 - test_fraction))
    cut = max(1, min(cut, len(events) - 1))
    return events.iloc[:cut].copy(), events.iloc[cut:].copy()


def engagement_weight(event_type: str, cfg: DataConfig) -> float:
    return {
        EVENT_IMPRESSION: cfg.weight_impression,
        EVENT_CLICK: cfg.weight_click,
        EVENT_ATC: cfg.weight_atc,
        EVENT_PURCHASE: cfg.weight_purchase,
    }.get(event_type, 0.0)


def build_pairs(events: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Collapse funnel events into (user, item, anchor) engagement weights for training."""
    events = events.copy()
    events["weight"] = events["event_type"].map(lambda e: engagement_weight(e, cfg))
    g = (
        events.groupby(["user_id", "item_id", "anchor_item_id"], as_index=False)
        .agg(weight=("weight", "sum"), last_ts=("timestamp", "max"))
        .sort_values("last_ts")
    )
    return g


def generate_and_save(cfg: ProjectConfig) -> CatalogTables:
    tables = generate_catalog(cfg.data)
    persist_catalog(tables, cfg.paths.data_dir / "processed")
    return tables
