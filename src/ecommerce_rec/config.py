from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")


@dataclass
class DataConfig:
    n_users: int = 4000
    n_items: int = 2500
    n_categories: int = 12
    n_brands: int = 80
    seed: int = 42
    test_fraction: float = 0.15
    min_events_per_user: int = 5
    weight_impression: float = 0.001
    weight_click: float = 0.01
    weight_atc: float = 0.1
    weight_purchase: float = 1.0


@dataclass
class RetrievalConfig:
    embed_dim: int = 64
    hidden_dim: int = 128
    id_embed_dim: int = 32
    batch_size: int = 512
    epochs: int = 8
    learning_rate: float = 1e-3
    temperature: float = 0.07
    hard_negatives: int = 5
    candidate_k: int = 100
    max_history: int = 20
    device: str = "cpu"


@dataclass
class FaissConfig:
    index_type: str = "flat"
    nlist: int = 64
    nprobe: int = 8


@dataclass
class RankingConfig:
    negatives_per_positive: int = 15
    max_train_queries: int = 6000
    max_eval_queries: int = 800
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    max_rounds: int = 300
    early_stopping_rounds: int = 30
    ndcg_eval_at: list[int] = field(default_factory=lambda: [8, 10])
    sparse_threshold: int = 8


@dataclass
class ServingConfig:
    top_k: int = 8
    filter_out_of_stock: bool = True
    dedupe_same_brand_variants: bool = True


@dataclass
class ProjectConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    faiss: FaissConfig = field(default_factory=FaissConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def resolve(self, root: Path | None = None) -> ProjectConfig:
        """Make paths absolute relative to project root."""
        root = root or Path.cwd()
        self.paths.data_dir = (root / self.paths.data_dir).resolve()
        self.paths.artifacts_dir = (root / self.paths.artifacts_dir).resolve()
        return self


def _merge_dataclass(dc: Any, values: dict[str, Any]) -> Any:
    for k, v in values.items():
        if hasattr(dc, k):
            setattr(dc, k, v)
    return dc


def load_config(path: Path | str | None = None, root: Path | None = None) -> ProjectConfig:
    cfg = ProjectConfig()
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if "paths" in raw:
            _merge_dataclass(cfg.paths, {k: Path(v) if "dir" in k else v for k, v in raw["paths"].items()})
        if "data" in raw:
            _merge_dataclass(cfg.data, raw["data"])
        if "retrieval" in raw:
            _merge_dataclass(cfg.retrieval, raw["retrieval"])
        if "faiss" in raw:
            _merge_dataclass(cfg.faiss, raw["faiss"])
        if "ranking" in raw:
            _merge_dataclass(cfg.ranking, raw["ranking"])
        if "serving" in raw:
            _merge_dataclass(cfg.serving, raw["serving"])
    return cfg.resolve(root)
