"""Offline evaluation: NDCG@K for retrieval-only vs two-stage, sparse/dense cohorts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ndcg_score
from tqdm import tqdm

from ecommerce_rec.config import ProjectConfig
from ecommerce_rec.data import EVENT_ATC, EVENT_CLICK, EVENT_PURCHASE
from ecommerce_rec.data.generate import load_catalog, temporal_split
from ecommerce_rec.ranking.features import assemble_pair_features
from ecommerce_rec.ranking.train import (
    POSITIVE_EVENTS,
    _encode_context_batch,
    _queries_from_events,
    _train_histories,
)
from ecommerce_rec.retrieval.faiss_index import ann_search, load_faiss_index
from ecommerce_rec.retrieval.train import load_two_tower


def _ndcg_at(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if y_true.sum() < 1:
        return float("nan")
    return float(ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k))


def evaluate(cfg: ProjectConfig) -> dict:
    art = cfg.paths.artifacts_dir
    tables = load_catalog(cfg.paths.data_dir)
    train_events, test_events = temporal_split(tables.events, cfg.data.test_fraction)

    device = torch.device(cfg.retrieval.device)
    model, _ = load_two_tower(art / "two_tower.pt", device=device)
    index, _ = load_faiss_index(art)
    booster = lgb.Booster(model_file=str(art / "lgbm_ranker.txt"))
    with (art / "customer_understanding.pkl").open("rb") as f:
        cu = pickle.load(f)
    with (art / "co_lookup.pkl").open("rb") as f:
        co_lookup = pickle.load(f)

    histories = _train_histories(train_events)
    queries = _queries_from_events(test_events)
    # Keep users known at train time
    queries = queries[queries["user_id"].isin(cu.train_counts.keys())].reset_index(drop=True)
    if len(queries) > cfg.ranking.max_eval_queries:
        queries = queries.sample(n=cfg.ranking.max_eval_queries, random_state=cfg.data.seed)

    prod_idx = tables.products.set_index("item_id")
    k_ret = cfg.retrieval.candidate_k
    k_show = cfg.serving.top_k
    sparse_th = cfg.ranking.sparse_threshold

    metrics = {
        "retrieval_ndcg": [],
        "twostage_ndcg": [],
        "retrieval_ndcg_sparse": [],
        "twostage_ndcg_sparse": [],
        "retrieval_ndcg_dense": [],
        "twostage_ndcg_dense": [],
        "recall_at_k": [],
    }

    batch_size = 32
    user_list = queries["user_id"].astype(int).tolist()
    anchor_list = queries["anchor_item_id"].astype(int).tolist()
    pos_list = queries["positives"].tolist()

    for start in tqdm(range(0, len(queries), batch_size), desc="evaluate"):
        sl = slice(start, start + batch_size)
        uids = user_list[sl]
        aids = anchor_list[sl]
        ctx = _encode_context_batch(
            model, uids, aids, histories, tables.products, cfg.retrieval.max_history, device
        )
        scores, cand_ids = ann_search(index, ctx, k_ret)

        for bi, (uid, aid, positives) in enumerate(zip(uids, aids, pos_list[sl])):
            positives = {p for p in positives if p != aid and p in prod_idx.index}
            if not positives:
                continue
            cands = [int(c) for c in cand_ids[bi].tolist() if int(c) >= 0 and int(c) != aid]
            sc = scores[bi][: len(cands)].tolist()
            retrieved = set(cands)
            recall = len(positives & retrieved) / max(len(positives), 1)
            metrics["recall_at_k"].append(float(recall))
            # Include missing positives so NDCG is well-defined on the candidate list
            for p in positives:
                if p not in cands:
                    cands.append(p)
                    sc.append(-1.0)

            y_true = np.asarray([1.0 if c in positives else 0.0 for c in cands], dtype=np.float32)
            y_ret = np.asarray(sc, dtype=np.float32)

            X = []
            for rank, (cid, s) in enumerate(zip(cands, sc)):
                X.append(
                    assemble_pair_features(
                        uid, aid, cid, float(s), rank, prod_idx, cu, co_lookup
                    )
                )
            X = np.asarray(X, dtype=np.float32)
            y_rank = booster.predict(X)

            nd_ret = _ndcg_at(y_true, y_ret, k_show)
            nd_two = _ndcg_at(y_true, y_rank, k_show)
            if np.isnan(nd_ret) or np.isnan(nd_two):
                continue

            metrics["retrieval_ndcg"].append(nd_ret)
            metrics["twostage_ndcg"].append(nd_two)
            sparse = cu.train_counts.get(uid, 0) < sparse_th
            if sparse:
                metrics["retrieval_ndcg_sparse"].append(nd_ret)
                metrics["twostage_ndcg_sparse"].append(nd_two)
            else:
                metrics["retrieval_ndcg_dense"].append(nd_ret)
                metrics["twostage_ndcg_dense"].append(nd_two)

    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    report = {
        "n_queries": len(metrics["retrieval_ndcg"]),
        "k": k_show,
        "candidate_k": k_ret,
        "recall_at_candidate_k": mean(metrics["recall_at_k"]),
        "ndcg_retrieval": mean(metrics["retrieval_ndcg"]),
        "ndcg_two_stage": mean(metrics["twostage_ndcg"]),
        "delta_two_stage_minus_retrieval": mean(metrics["twostage_ndcg"])
        - mean(metrics["retrieval_ndcg"]),
        "ndcg_retrieval_sparse": mean(metrics["retrieval_ndcg_sparse"]),
        "ndcg_two_stage_sparse": mean(metrics["twostage_ndcg_sparse"]),
        "ndcg_retrieval_dense": mean(metrics["retrieval_ndcg_dense"]),
        "ndcg_two_stage_dense": mean(metrics["twostage_ndcg_dense"]),
        "n_sparse": len(metrics["retrieval_ndcg_sparse"]),
        "n_dense": len(metrics["retrieval_ndcg_dense"]),
    }
    out = art / "eval_metrics.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report
