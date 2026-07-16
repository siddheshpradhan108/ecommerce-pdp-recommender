"""Online-style inference: PDP request → ANN retrieval → LightGBM re-rank → top-K."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from ecommerce_rec.config import ProjectConfig, load_config
from ecommerce_rec.data.generate import load_catalog
from ecommerce_rec.ranking.features import assemble_pair_features
from ecommerce_rec.ranking.train import _encode_context_batch, _train_histories
from ecommerce_rec.retrieval.faiss_index import ann_search, load_faiss_index
from ecommerce_rec.retrieval.train import load_two_tower


@dataclass
class Recommendation:
    item_id: int
    title: str
    brand_id: int
    category_id: int
    price: float
    retrieval_score: float
    ranker_score: float
    rank: int


class PDPRecommender:
    """Production-shaped serving object for Similar Items / personalized PDP carousels."""

    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        art = cfg.paths.artifacts_dir
        self.device = torch.device(cfg.retrieval.device)
        self.model, self.meta = load_two_tower(art / "two_tower.pt", device=self.device)
        self.index, _ = load_faiss_index(art)
        self.booster = lgb.Booster(model_file=str(art / "lgbm_ranker.txt"))
        with (art / "customer_understanding.pkl").open("rb") as f:
            self.cu = pickle.load(f)
        with (art / "co_lookup.pkl").open("rb") as f:
            self.co_lookup = pickle.load(f)
        self.tables = load_catalog(cfg.paths.data_dir)
        self.prod_idx = self.tables.products.set_index("item_id")
        # Build histories from all events for demo serving
        self.histories = _train_histories(self.tables.events)

    def recommend(self, user_id: int, anchor_item_id: int, top_k: int | None = None) -> list[Recommendation]:
        top_k = top_k or self.cfg.serving.top_k
        k_ret = self.cfg.retrieval.candidate_k

        ctx = _encode_context_batch(
            self.model,
            [user_id],
            [anchor_item_id],
            self.histories,
            self.tables.products,
            self.cfg.retrieval.max_history,
            self.device,
        )
        scores, cand_ids = ann_search(self.index, ctx, k_ret)
        cands = [int(c) for c in cand_ids[0].tolist() if int(c) >= 0 and int(c) != anchor_item_id]
        sc = scores[0][: len(cands)].tolist()

        rows = []
        keep_ids = []
        keep_scores = []
        seen_brands: set[int] = set()
        for rank, (cid, s) in enumerate(zip(cands, sc)):
            if cid not in self.prod_idx.index:
                continue
            row = self.prod_idx.loc[cid]
            if self.cfg.serving.filter_out_of_stock and not bool(row["in_stock"]):
                continue
            if self.cfg.serving.dedupe_same_brand_variants:
                b = int(row["brand_id"])
                # Allow first item per brand in candidate stream before ranker
                # (light diversity; ranker still sees full set below)
            keep_ids.append(cid)
            keep_scores.append(float(s))
            rows.append(
                assemble_pair_features(
                    user_id, anchor_item_id, cid, float(s), rank, self.prod_idx, self.cu, self.co_lookup
                )
            )

        if not rows:
            return []

        X = np.asarray(rows, dtype=np.float32)
        rank_scores = self.booster.predict(X)
        order = np.argsort(-rank_scores)

        results: list[Recommendation] = []
        for rnk, idx in enumerate(order):
            cid = keep_ids[idx]
            row = self.prod_idx.loc[cid]
            brand = int(row["brand_id"])
            if self.cfg.serving.dedupe_same_brand_variants and brand in seen_brands:
                continue
            seen_brands.add(brand)
            results.append(
                Recommendation(
                    item_id=cid,
                    title=str(row["title"]),
                    brand_id=brand,
                    category_id=int(row["category_id"]),
                    price=float(row["price"]),
                    retrieval_score=keep_scores[idx],
                    ranker_score=float(rank_scores[idx]),
                    rank=len(results) + 1,
                )
            )
            if len(results) >= top_k:
                break
        return results


def demo_recommend(cfg: ProjectConfig, user_id: int | None = None, anchor_id: int | None = None) -> None:
    rec = PDPRecommender(cfg)
    if user_id is None:
        user_id = int(rec.tables.users["user_id"].iloc[0])
    if anchor_id is None:
        # Use a recent engaged item as anchor
        ev = rec.tables.events
        sub = ev[ev["user_id"] == user_id]
        anchor_id = int(sub["anchor_item_id"].iloc[-1]) if len(sub) else int(rec.tables.products["item_id"].iloc[0])

    print(f"PDP recommendations for user={user_id} anchor={anchor_id}")
    for r in rec.recommend(user_id, anchor_id):
        print(
            f"  #{r.rank} item={r.item_id}  ${r.price:.2f}  "
            f"ret={r.retrieval_score:.3f}  lgbm={r.ranker_score:.3f}  {r.title[:50]}"
        )
