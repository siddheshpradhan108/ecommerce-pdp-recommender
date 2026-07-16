from __future__ import annotations

from pathlib import Path

from ecommerce_rec.config import load_config
from ecommerce_rec.data.generate import generate_catalog, temporal_split, build_pairs
from ecommerce_rec.ranking.features import FEATURE_NAMES, build_customer_understanding


def test_generate_catalog_shapes():
    cfg = load_config(None)
    cfg.data.n_users = 50
    cfg.data.n_items = 40
    cfg.data.n_categories = 5
    cfg.data.n_brands = 10
    tables = generate_catalog(cfg.data)
    assert len(tables.products) == 40
    assert len(tables.users) > 0
    assert len(tables.events) > 0
    assert set(tables.products.columns) >= {
        "item_id",
        "brand_id",
        "category_id",
        "price",
        "in_stock",
    }


def test_temporal_split_and_pairs():
    cfg = load_config(None)
    cfg.data.n_users = 40
    cfg.data.n_items = 30
    cfg.data.n_categories = 5
    cfg.data.n_brands = 10
    tables = generate_catalog(cfg.data)
    train, test = temporal_split(tables.events, 0.2)
    assert len(train) + len(test) == len(tables.events)
    assert train["timestamp"].max() <= test["timestamp"].min()
    pairs = build_pairs(train, cfg.data)
    assert {"user_id", "item_id", "anchor_item_id", "weight"}.issubset(pairs.columns)


def test_customer_understanding_and_feature_names():
    cfg = load_config(None)
    cfg.data.n_users = 40
    cfg.data.n_items = 30
    cfg.data.n_categories = 5
    cfg.data.n_brands = 10
    tables = generate_catalog(cfg.data)
    train, _ = temporal_split(tables.events, 0.2)
    cu = build_customer_understanding(train, tables.products)
    assert len(FEATURE_NAMES) == 14
    assert isinstance(cu.brand_affinity, dict)
