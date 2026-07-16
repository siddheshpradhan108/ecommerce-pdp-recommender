"""Offline item embedding export + FAISS ANN index for serving."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch

from ecommerce_rec.config import ProjectConfig
from ecommerce_rec.data.generate import load_catalog
from ecommerce_rec.retrieval.model import item_numeric_features
from ecommerce_rec.retrieval.train import load_two_tower


@torch.no_grad()
def encode_all_items(model, products: pd.DataFrame, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    n = len(products)
    dim = model.item_tower.mlp.net[-1].out_features
    out = np.zeros((int(products["item_id"].max()) + 1, dim), dtype=np.float32)

    ids = products["item_id"].to_numpy()
    brands = products["brand_id"].to_numpy()
    cats = products["category_id"].to_numpy()
    prices = products["price"].to_numpy(dtype=np.float32)
    bands = products["price_band"].to_numpy(dtype=np.float32)
    ratings = products["avg_rating"].to_numpy(dtype=np.float32)
    reviews = products["n_reviews"].to_numpy(dtype=np.float32)

    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        item_ids = torch.tensor(ids[sl], dtype=torch.long, device=device)
        brand_ids = torch.tensor(brands[sl], dtype=torch.long, device=device)
        category_ids = torch.tensor(cats[sl], dtype=torch.long, device=device)
        numeric = item_numeric_features(
            torch.tensor(prices[sl], device=device),
            torch.tensor(bands[sl], device=device),
            torch.tensor(ratings[sl], device=device),
            torch.tensor(reviews[sl], device=device),
        )
        emb = model.encode_items(item_ids, brand_ids, category_ids, numeric).cpu().numpy()
        out[ids[sl]] = emb
    return out


def build_faiss_index(embeddings: np.ndarray, cfg: ProjectConfig) -> faiss.Index:
    """Inner-product index on L2-normalized vectors ≡ cosine ANN."""
    d = embeddings.shape[1]
    xb = np.ascontiguousarray(embeddings.astype(np.float32))
    if cfg.faiss.index_type == "ivf" and len(xb) >= cfg.faiss.nlist * 10:
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, cfg.faiss.nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(xb)
        index.add(xb)
        index.nprobe = cfg.faiss.nprobe
    else:
        index = faiss.IndexFlatIP(d)
        index.add(xb)
    return index


def build_and_persist_index(cfg: ProjectConfig) -> dict:
    art = cfg.paths.artifacts_dir
    device = torch.device(cfg.retrieval.device)
    model, meta = load_two_tower(art / "two_tower.pt", device=device)
    tables = load_catalog(cfg.paths.data_dir)
    emb = encode_all_items(model, tables.products, device)
    index = build_faiss_index(emb, cfg)

    emb_path = art / "item_embeddings.npy"
    idx_path = art / "faiss.index"
    np.save(emb_path, emb)
    faiss.write_index(index, str(idx_path))

    meta_out = {
        "n_vectors": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "index_type": cfg.faiss.index_type,
        "candidate_k": cfg.retrieval.candidate_k,
    }
    (art / "faiss_meta.json").write_text(json.dumps(meta_out, indent=2))
    print(f"Saved item embeddings → {emb_path}")
    print(f"Saved FAISS index → {idx_path}")
    return meta_out


def load_faiss_index(art: Path) -> tuple[faiss.Index, np.ndarray]:
    index = faiss.read_index(str(art / "faiss.index"))
    emb = np.load(art / "item_embeddings.npy")
    return index, emb


def ann_search(index: faiss.Index, query_vecs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (scores [B,k], item_ids [B,k])."""
    q = np.ascontiguousarray(query_vecs.astype(np.float32))
    scores, ids = index.search(q, k)
    return scores, ids
