"""Dataset + training loop for Two-Tower retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ecommerce_rec.config import ProjectConfig
from ecommerce_rec.data import EVENT_ATC, EVENT_CLICK, EVENT_PURCHASE
from ecommerce_rec.data.generate import build_pairs, load_catalog, temporal_split
from ecommerce_rec.retrieval.model import TwoTowerModel, item_numeric_features


POSITIVE_EVENTS = {EVENT_CLICK, EVENT_ATC, EVENT_PURCHASE}


class RetrievalDataset(Dataset):
    """Each sample: context (user, history before t, anchor) → positive engaged item."""

    def __init__(
        self,
        pairs: pd.DataFrame,
        products: pd.DataFrame,
        user_histories: dict[int, list[tuple[pd.Timestamp, int]]],
        max_history: int,
        n_items: int,
    ):
        self.pairs = pairs.reset_index(drop=True)
        self.products = products.set_index("item_id")
        self.user_histories = user_histories
        self.max_history = max_history
        self.n_items = n_items

    def __len__(self) -> int:
        return len(self.pairs)

    def _history_tensor(self, user_id: int, before_ts: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        hist = [iid for ts, iid in self.user_histories.get(user_id, []) if ts < before_ts]
        hist = hist[-self.max_history :]
        ids = np.zeros(self.max_history, dtype=np.int64)
        mask = np.zeros(self.max_history, dtype=np.float32)
        if not hist:
            # cold-start: pad with zeros (item 0); mask stays 0 so mean is unused
            return ids, mask
        for i, iid in enumerate(hist):
            ids[i] = iid
            mask[i] = 1.0
        return ids, mask

    def _item_feats(self, item_id: int) -> tuple[int, int, int, float, float, float, float]:
        row = self.products.loc[int(item_id)]
        return (
            int(item_id),
            int(row["brand_id"]),
            int(row["category_id"]),
            float(row["price"]),
            float(row["price_band"]),
            float(row["avg_rating"]),
            float(row["n_reviews"]),
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.pairs.iloc[idx]
        user_id = int(row["user_id"])
        item_id = int(row["item_id"])
        anchor_id = int(row["anchor_item_id"])
        ts = row["last_ts"]

        hist_ids, hist_mask = self._history_tensor(user_id, ts)
        # Build feature arrays for history slots
        h_brand = np.zeros(self.max_history, dtype=np.int64)
        h_cat = np.zeros(self.max_history, dtype=np.int64)
        h_price = np.zeros(self.max_history, dtype=np.float32)
        h_band = np.zeros(self.max_history, dtype=np.float32)
        h_rating = np.zeros(self.max_history, dtype=np.float32)
        h_reviews = np.zeros(self.max_history, dtype=np.float32)
        for i in range(self.max_history):
            if hist_mask[i] < 0.5:
                continue
            _, b, c, p, pb, r, nr = self._item_feats(int(hist_ids[i]))
            h_brand[i], h_cat[i] = b, c
            h_price[i], h_band[i], h_rating[i], h_reviews[i] = p, pb, r, nr

        _, ab, ac, ap, apb, ar, anr = self._item_feats(anchor_id)
        _, ib, ic, ip, ipb, ir, inr = self._item_feats(item_id)

        return {
            "user_ids": torch.tensor(user_id, dtype=torch.long),
            "history_item_ids": torch.tensor(hist_ids, dtype=torch.long),
            "history_mask": torch.tensor(hist_mask, dtype=torch.float32),
            "history_brand": torch.tensor(h_brand, dtype=torch.long),
            "history_cat": torch.tensor(h_cat, dtype=torch.long),
            "history_price": torch.tensor(h_price, dtype=torch.float32),
            "history_band": torch.tensor(h_band, dtype=torch.float32),
            "history_rating": torch.tensor(h_rating, dtype=torch.float32),
            "history_reviews": torch.tensor(h_reviews, dtype=torch.float32),
            "anchor_ids": torch.tensor(anchor_id, dtype=torch.long),
            "anchor_brand": torch.tensor(ab, dtype=torch.long),
            "anchor_cat": torch.tensor(ac, dtype=torch.long),
            "anchor_price": torch.tensor(ap, dtype=torch.float32),
            "anchor_band": torch.tensor(apb, dtype=torch.float32),
            "anchor_rating": torch.tensor(ar, dtype=torch.float32),
            "anchor_reviews": torch.tensor(anr, dtype=torch.float32),
            "pos_ids": torch.tensor(item_id, dtype=torch.long),
            "pos_brand": torch.tensor(ib, dtype=torch.long),
            "pos_cat": torch.tensor(ic, dtype=torch.long),
            "pos_price": torch.tensor(ip, dtype=torch.float32),
            "pos_band": torch.tensor(ipb, dtype=torch.float32),
            "pos_rating": torch.tensor(ir, dtype=torch.float32),
            "pos_reviews": torch.tensor(inr, dtype=torch.float32),
            "weight": torch.tensor(float(row["weight"]), dtype=torch.float32),
        }


def _build_user_histories(events: pd.DataFrame) -> dict[int, list[tuple[pd.Timestamp, int]]]:
    pos = events[events["event_type"].isin(POSITIVE_EVENTS)].sort_values("timestamp")
    hist: dict[int, list[tuple[pd.Timestamp, int]]] = {}
    for r in pos.itertuples(index=False):
        hist.setdefault(int(r.user_id), []).append((r.timestamp, int(r.item_id)))
    return hist


def _batch_to_model_inputs(batch: dict, device: torch.device) -> tuple[dict, dict]:
    hist_numeric = item_numeric_features(
        batch["history_price"].to(device),
        batch["history_band"].to(device),
        batch["history_rating"].to(device),
        batch["history_reviews"].to(device),
    )
    anchor_numeric = item_numeric_features(
        batch["anchor_price"].to(device),
        batch["anchor_band"].to(device),
        batch["anchor_rating"].to(device),
        batch["anchor_reviews"].to(device),
    )
    pos_numeric = item_numeric_features(
        batch["pos_price"].to(device),
        batch["pos_band"].to(device),
        batch["pos_rating"].to(device),
        batch["pos_reviews"].to(device),
    )
    ctx = {
        "user_ids": batch["user_ids"].to(device),
        "history_item_ids": batch["history_item_ids"].to(device),
        "history_mask": batch["history_mask"].to(device),
        "history_brand": batch["history_brand"].to(device),
        "history_cat": batch["history_cat"].to(device),
        "history_numeric": hist_numeric,
        "anchor_ids": batch["anchor_ids"].to(device),
        "anchor_brand": batch["anchor_brand"].to(device),
        "anchor_cat": batch["anchor_cat"].to(device),
        "anchor_numeric": anchor_numeric,
    }
    pos = {
        "item_ids": batch["pos_ids"].to(device),
        "brand_ids": batch["pos_brand"].to(device),
        "category_ids": batch["pos_cat"].to(device),
        "numeric": pos_numeric,
    }
    return ctx, pos


def train_two_tower(cfg: ProjectConfig) -> dict:
    art = cfg.paths.artifacts_dir
    art.mkdir(parents=True, exist_ok=True)

    tables = load_catalog(cfg.paths.data_dir)
    train_events, _ = temporal_split(tables.events, cfg.data.test_fraction)
    # Positives only for contrastive pairs (engaged items)
    eng = train_events[train_events["event_type"].isin(POSITIVE_EVENTS)]
    pairs = build_pairs(eng, cfg.data)
    # Cap extreme soft labels for stability
    pairs = pairs[pairs["weight"] >= cfg.data.weight_click].reset_index(drop=True)

    histories = _build_user_histories(train_events)
    n_users = int(tables.users["user_id"].max()) + 1
    n_items = int(tables.products["item_id"].max()) + 1
    n_brands = int(tables.products["brand_id"].max()) + 1
    n_cats = int(tables.products["category_id"].max()) + 1

    ds = RetrievalDataset(pairs, tables.products, histories, cfg.retrieval.max_history, n_items)
    loader = DataLoader(
        ds,
        batch_size=cfg.retrieval.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    device = torch.device(cfg.retrieval.device)
    model = TwoTowerModel(
        n_users=n_users,
        n_items=n_items,
        n_brands=n_brands,
        n_categories=n_cats,
        id_dim=cfg.retrieval.id_embed_dim,
        hidden=cfg.retrieval.hidden_dim,
        out_dim=cfg.retrieval.embed_dim,
        temperature=cfg.retrieval.temperature,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.retrieval.learning_rate)

    history_loss: list[float] = []
    model.train()
    for epoch in range(cfg.retrieval.epochs):
        total = 0.0
        n = 0
        for batch in tqdm(loader, desc=f"two-tower epoch {epoch+1}/{cfg.retrieval.epochs}"):
            ctx_in, pos_in = _batch_to_model_inputs(batch, device)
            context = model.encode_context(**ctx_in)
            pos_emb = model.encode_items(**pos_in)
            loss = model.info_nce(context, pos_emb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        epoch_loss = total / max(n, 1)
        history_loss.append(epoch_loss)
        print(f"epoch {epoch+1}: loss={epoch_loss:.4f}")

    ckpt = art / "two_tower.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "n_users": n_users,
                "n_items": n_items,
                "n_brands": n_brands,
                "n_categories": n_cats,
                "id_dim": cfg.retrieval.id_embed_dim,
                "hidden": cfg.retrieval.hidden_dim,
                "out_dim": cfg.retrieval.embed_dim,
                "temperature": cfg.retrieval.temperature,
                "max_history": cfg.retrieval.max_history,
            },
            "train_loss": history_loss,
        },
        ckpt,
    )
    metrics = {"train_loss_by_epoch": history_loss, "n_pairs": len(pairs)}
    (art / "retrieval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved Two-Tower checkpoint → {ckpt}")
    return metrics


def load_two_tower(ckpt_path: Path, device: str | torch.device = "cpu") -> tuple[TwoTowerModel, dict]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = payload["meta"]
    model = TwoTowerModel(
        n_users=meta["n_users"],
        n_items=meta["n_items"],
        n_brands=meta["n_brands"],
        n_categories=meta["n_categories"],
        id_dim=meta["id_dim"],
        hidden=meta["hidden"],
        out_dim=meta["out_dim"],
        temperature=meta["temperature"],
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, meta
