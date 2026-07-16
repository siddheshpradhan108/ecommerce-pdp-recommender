"""Build LambdaRank training matrix and train LightGBM ranker."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ecommerce_rec.config import ProjectConfig
from ecommerce_rec.data import EVENT_ATC, EVENT_CLICK, EVENT_PURCHASE
from ecommerce_rec.data.generate import load_catalog, temporal_split
from ecommerce_rec.ranking.features import (
    FEATURE_NAMES,
    assemble_pair_features,
    build_co_purchase_lookup,
    build_customer_understanding,
)
from ecommerce_rec.retrieval.faiss_index import ann_search, load_faiss_index
from ecommerce_rec.retrieval.model import item_numeric_features
from ecommerce_rec.retrieval.train import load_two_tower

POSITIVE_EVENTS = {EVENT_CLICK, EVENT_ATC, EVENT_PURCHASE}


def _encode_context_batch(
    model,
    user_ids: list[int],
    anchors: list[int],
    histories: dict[int, list[int]],
    products: pd.DataFrame,
    max_history: int,
    device: torch.device,
) -> np.ndarray:
    """Encode context vectors for a list of (user, anchor) queries."""
    prod = products.set_index("item_id")
    b = len(user_ids)
    h = max_history

    hist_ids = np.zeros((b, h), dtype=np.int64)
    hist_mask = np.zeros((b, h), dtype=np.float32)
    h_brand = np.zeros((b, h), dtype=np.int64)
    h_cat = np.zeros((b, h), dtype=np.int64)
    h_price = np.zeros((b, h), dtype=np.float32)
    h_band = np.zeros((b, h), dtype=np.float32)
    h_rating = np.zeros((b, h), dtype=np.float32)
    h_reviews = np.zeros((b, h), dtype=np.float32)

    a_brand = np.zeros(b, dtype=np.int64)
    a_cat = np.zeros(b, dtype=np.int64)
    a_price = np.zeros(b, dtype=np.float32)
    a_band = np.zeros(b, dtype=np.float32)
    a_rating = np.zeros(b, dtype=np.float32)
    a_reviews = np.zeros(b, dtype=np.float32)

    for i, (uid, aid) in enumerate(zip(user_ids, anchors)):
        hist = histories.get(uid, [])[-h:]
        for j, iid in enumerate(hist):
            row = prod.loc[iid]
            hist_ids[i, j] = iid
            hist_mask[i, j] = 1.0
            h_brand[i, j] = int(row["brand_id"])
            h_cat[i, j] = int(row["category_id"])
            h_price[i, j] = float(row["price"])
            h_band[i, j] = float(row["price_band"])
            h_rating[i, j] = float(row["avg_rating"])
            h_reviews[i, j] = float(row["n_reviews"])
        arow = prod.loc[aid]
        a_brand[i] = int(arow["brand_id"])
        a_cat[i] = int(arow["category_id"])
        a_price[i] = float(arow["price"])
        a_band[i] = float(arow["price_band"])
        a_rating[i] = float(arow["avg_rating"])
        a_reviews[i] = float(arow["n_reviews"])

    with torch.no_grad():
        hist_numeric = item_numeric_features(
            torch.tensor(h_price, device=device),
            torch.tensor(h_band, device=device),
            torch.tensor(h_rating, device=device),
            torch.tensor(h_reviews, device=device),
        )
        anchor_numeric = item_numeric_features(
            torch.tensor(a_price, device=device),
            torch.tensor(a_band, device=device),
            torch.tensor(a_rating, device=device),
            torch.tensor(a_reviews, device=device),
        )
        ctx = model.encode_context(
            user_ids=torch.tensor(user_ids, dtype=torch.long, device=device),
            history_item_ids=torch.tensor(hist_ids, dtype=torch.long, device=device),
            history_mask=torch.tensor(hist_mask, dtype=torch.float32, device=device),
            history_brand=torch.tensor(h_brand, dtype=torch.long, device=device),
            history_cat=torch.tensor(h_cat, dtype=torch.long, device=device),
            history_numeric=hist_numeric,
            anchor_ids=torch.tensor(anchors, dtype=torch.long, device=device),
            anchor_brand=torch.tensor(a_brand, dtype=torch.long, device=device),
            anchor_cat=torch.tensor(a_cat, dtype=torch.long, device=device),
            anchor_numeric=anchor_numeric,
        )
    return ctx.cpu().numpy()


def _train_histories(train_events: pd.DataFrame) -> dict[int, list[int]]:
    pos = train_events[train_events["event_type"].isin(POSITIVE_EVENTS)].sort_values("timestamp")
    out: dict[int, list[int]] = {}
    for r in pos.itertuples(index=False):
        out.setdefault(int(r.user_id), []).append(int(r.item_id))
    return out


def _queries_from_events(events: pd.DataFrame) -> pd.DataFrame:
    """One query = (user, anchor) with positive item set from engagement."""
    pos = events[events["event_type"].isin(POSITIVE_EVENTS)]
    g = (
        pos.groupby(["user_id", "anchor_item_id"])["item_id"]
        .apply(lambda s: set(int(x) for x in s))
        .reset_index()
        .rename(columns={"item_id": "positives"})
    )
    return g


def build_ltr_dataset(
    cfg: ProjectConfig,
    events: pd.DataFrame,
    products: pd.DataFrame,
    co_lookup: dict,
    cu,
    model,
    index,
    histories: dict[int, list[int]],
    max_queries: int,
    seed: int,
    for_train: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    queries = _queries_from_events(events)
    if len(queries) > max_queries:
        queries = queries.sample(n=max_queries, random_state=seed).reset_index(drop=True)

    device = torch.device(cfg.retrieval.device)
    prod_idx = products.set_index("item_id")
    k = cfg.retrieval.candidate_k
    neg_n = cfg.ranking.negatives_per_positive

    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    groups: list[int] = []

    # Encode contexts in mini-batches for speed
    batch_size = 64
    user_list = queries["user_id"].astype(int).tolist()
    anchor_list = queries["anchor_item_id"].astype(int).tolist()
    pos_list = queries["positives"].tolist()

    for start in tqdm(range(0, len(queries), batch_size), desc="LTR features"):
        sl = slice(start, start + batch_size)
        uids = user_list[sl]
        aids = anchor_list[sl]
        ctx = _encode_context_batch(
            model, uids, aids, histories, products, cfg.retrieval.max_history, device
        )
        scores, cand_ids = ann_search(index, ctx, k)

        for bi, (uid, aid, positives) in enumerate(zip(uids, aids, pos_list[sl])):
            cands = cand_ids[bi].tolist()
            sc = scores[bi].tolist()
            # Ensure positives appear in training set
            cand_set = set(int(c) for c in cands if c >= 0)
            for p in positives:
                if p not in cand_set and p in prod_idx.index:
                    cands.append(p)
                    sc.append(0.0)

            labels: dict[int, float] = {}
            for rank, (cid, s) in enumerate(zip(cands, sc)):
                cid = int(cid)
                if cid < 0 or cid not in prod_idx.index:
                    continue
                if cid == aid:
                    continue
                labels[cid] = 1.0 if cid in positives else 0.0

            # Sample negatives for training; keep all for eval lists capped
            pos_ids = [i for i, lab in labels.items() if lab > 0]
            neg_ids = [i for i, lab in labels.items() if lab == 0]
            if for_train:
                if not pos_ids:
                    continue
                if len(neg_ids) > neg_n * max(len(pos_ids), 1):
                    neg_ids = list(rng.choice(neg_ids, size=neg_n * len(pos_ids), replace=False))
                keep = set(pos_ids + neg_ids)
            else:
                keep = set(labels.keys())

            # map id -> retrieval score/rank
            score_map = {int(c): float(s) for c, s in zip(cands, sc)}
            rank_map = {int(c): r for r, c in enumerate(cands)}

            n_kept = 0
            for cid in keep:
                feats = assemble_pair_features(
                    uid,
                    aid,
                    cid,
                    score_map.get(cid, 0.0),
                    rank_map.get(cid, k),
                    prod_idx,
                    cu,
                    co_lookup,
                )
                X_rows.append(feats)
                y_rows.append(labels.get(cid, 0.0))
                n_kept += 1
            if n_kept:
                groups.append(n_kept)

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    g = np.asarray(groups, dtype=np.int32)
    return X, y, g, FEATURE_NAMES


def _split_groups(
    X: np.ndarray, y: np.ndarray, g: np.ndarray, val_frac: float = 0.15, seed: int = 42
) -> tuple:
    rng = np.random.default_rng(seed)
    n_groups = len(g)
    idx = np.arange(n_groups)
    rng.shuffle(idx)
    n_val = max(1, int(n_groups * val_frac))
    val_g = set(idx[:n_val].tolist())
    # Expand group offsets
    offsets = np.cumsum(np.concatenate([[0], g]))
    tr_rows, va_rows = [], []
    tr_y, va_y = [], []
    tr_g, va_g = [], []
    for gi in range(n_groups):
        a, b = int(offsets[gi]), int(offsets[gi + 1])
        if gi in val_g:
            va_rows.append(X[a:b])
            va_y.append(y[a:b])
            va_g.append(g[gi])
        else:
            tr_rows.append(X[a:b])
            tr_y.append(y[a:b])
            tr_g.append(g[gi])
    X_tr = np.vstack(tr_rows)
    y_tr = np.concatenate(tr_y)
    X_va = np.vstack(va_rows)
    y_va = np.concatenate(va_y)
    return X_tr, y_tr, np.asarray(tr_g), X_va, y_va, np.asarray(va_g)


def train_ranker(cfg: ProjectConfig) -> dict:
    art = cfg.paths.artifacts_dir
    tables = load_catalog(cfg.paths.data_dir)
    train_events, test_events = temporal_split(tables.events, cfg.data.test_fraction)

    device = torch.device(cfg.retrieval.device)
    model, meta = load_two_tower(art / "two_tower.pt", device=device)
    index, _ = load_faiss_index(art)
    histories = _train_histories(train_events)
    cu = build_customer_understanding(train_events, tables.products)
    co_lookup = build_co_purchase_lookup(tables.co_purchase)

    X, y, g, names = build_ltr_dataset(
        cfg,
        train_events,
        tables.products,
        co_lookup,
        cu,
        model,
        index,
        histories,
        cfg.ranking.max_train_queries,
        cfg.data.seed,
        for_train=True,
    )
    X_tr, y_tr, g_tr, X_va, y_va, g_va = _split_groups(X, y, g)

    dtrain = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=names, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, label=y_va, group=g_va, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": cfg.ranking.ndcg_eval_at,
        "learning_rate": cfg.ranking.learning_rate,
        "num_leaves": cfg.ranking.num_leaves,
        "min_data_in_leaf": cfg.ranking.min_data_in_leaf,
        "feature_fraction": cfg.ranking.feature_fraction,
        "bagging_fraction": cfg.ranking.bagging_fraction,
        "bagging_freq": cfg.ranking.bagging_freq,
        "verbosity": -1,
        "seed": cfg.data.seed,
    }
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=cfg.ranking.max_rounds,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(cfg.ranking.early_stopping_rounds),
            lgb.log_evaluation(period=50),
        ],
    )

    model_path = art / "lgbm_ranker.txt"
    booster.save_model(str(model_path))
    with (art / "customer_understanding.pkl").open("wb") as f:
        pickle.dump(cu, f)
    with (art / "co_lookup.pkl").open("wb") as f:
        pickle.dump(co_lookup, f)
    with (art / "feature_names.json").open("w") as f:
        json.dump(names, f, indent=2)

    best = {
        "best_iteration": booster.best_iteration,
        "best_score": booster.best_score,
        "n_train_rows": int(len(y_tr)),
        "n_valid_rows": int(len(y_va)),
        "n_train_groups": int(len(g_tr)),
    }
    # Flatten best_score for JSON
    serializable = {
        "best_iteration": best["best_iteration"],
        "best_score": {k: {m: float(v) for m, v in d.items()} for k, d in booster.best_score.items()},
        "n_train_rows": best["n_train_rows"],
        "n_valid_rows": best["n_valid_rows"],
        "n_train_groups": best["n_train_groups"],
    }
    (art / "ranker_train_metrics.json").write_text(json.dumps(serializable, indent=2))
    print(f"Saved LightGBM ranker → {model_path}")
    return serializable
