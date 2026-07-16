"""Two-Tower (dual encoder) retrieval for PDP context → product candidates.

Context tower:   user history + anchor item  →  embedding u
Candidate tower: product features            →  embedding v
Score:           cosine(u, v) / temperature  (InfoNCE / in-batch negatives)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ItemTower(nn.Module):
    """Encode catalog attributes into a unit vector (precomputable for ANN)."""

    def __init__(
        self,
        n_items: int,
        n_brands: int,
        n_categories: int,
        id_dim: int,
        hidden: int,
        out_dim: int,
    ):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, id_dim)
        self.brand_emb = nn.Embedding(n_brands, id_dim)
        self.cat_emb = nn.Embedding(n_categories, id_dim)
        # price_band (0-4), log_price, avg_rating, log_reviews
        numeric_dim = 4
        self.mlp = MLP(id_dim * 3 + numeric_dim, hidden, out_dim)

    def forward(
        self,
        item_ids: torch.Tensor,
        brand_ids: torch.Tensor,
        category_ids: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                self.item_emb(item_ids),
                self.brand_emb(brand_ids),
                self.cat_emb(category_ids),
                numeric,
            ],
            dim=-1,
        )
        return F.normalize(self.mlp(x), dim=-1)


class ContextTower(nn.Module):
    """Encode (user, history, anchor) into a unit vector for ANN query."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_brands: int,
        n_categories: int,
        id_dim: int,
        hidden: int,
        out_dim: int,
        item_tower: ItemTower,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, id_dim)
        self.item_tower = item_tower  # share item encoder for history / anchor
        # user + mean(history) + anchor
        self.mlp = MLP(id_dim + out_dim * 2, hidden, out_dim)

    def forward(
        self,
        user_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        history_mask: torch.Tensor,
        history_brand: torch.Tensor,
        history_cat: torch.Tensor,
        history_numeric: torch.Tensor,
        anchor_ids: torch.Tensor,
        anchor_brand: torch.Tensor,
        anchor_cat: torch.Tensor,
        anchor_numeric: torch.Tensor,
    ) -> torch.Tensor:
        # history_*: [B, H, ...]
        b, h = history_item_ids.shape
        flat_ids = history_item_ids.reshape(-1)
        flat_brand = history_brand.reshape(-1)
        flat_cat = history_cat.reshape(-1)
        flat_num = history_numeric.reshape(-1, history_numeric.shape[-1])
        hist_emb = self.item_tower(flat_ids, flat_brand, flat_cat, flat_num).view(b, h, -1)
        mask = history_mask.unsqueeze(-1)  # [B, H, 1]
        hist_sum = (hist_emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        hist_mean = hist_sum / denom

        anchor = self.item_tower(anchor_ids, anchor_brand, anchor_cat, anchor_numeric)
        user = self.user_emb(user_ids)
        x = torch.cat([user, hist_mean, anchor], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_brands: int,
        n_categories: int,
        id_dim: int = 32,
        hidden: int = 128,
        out_dim: int = 64,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.item_tower = ItemTower(n_items, n_brands, n_categories, id_dim, hidden, out_dim)
        self.context_tower = ContextTower(
            n_users, n_items, n_brands, n_categories, id_dim, hidden, out_dim, self.item_tower
        )

    def encode_items(
        self,
        item_ids: torch.Tensor,
        brand_ids: torch.Tensor,
        category_ids: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_tower(item_ids, brand_ids, category_ids, numeric)

    def encode_context(self, **kwargs) -> torch.Tensor:
        return self.context_tower(**kwargs)

    def info_nce(
        self,
        context: torch.Tensor,
        positive_items: torch.Tensor,
    ) -> torch.Tensor:
        """In-batch InfoNCE: positives are paired rows; other batch rows are negatives."""
        # context, positive_items: [B, D] unit vectors
        logits = (context @ positive_items.T) / self.temperature  # [B, B]
        labels = torch.arange(context.size(0), device=context.device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def score(self, context: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (context * items).sum(dim=-1)


def item_numeric_features(
    price: torch.Tensor,
    price_band: torch.Tensor,
    avg_rating: torch.Tensor,
    n_reviews: torch.Tensor,
) -> torch.Tensor:
    """Shared numeric featurization for item / history tensors."""
    log_price = torch.log1p(price)
    log_reviews = torch.log1p(n_reviews.float())
    band = price_band.float() / 4.0
    rating = avg_rating / 5.0
    return torch.stack([log_price, band, rating, log_reviews], dim=-1)
