"""CLI entrypoint for the e-commerce two-stage PDP recommender."""

from __future__ import annotations

import argparse
from pathlib import Path

from ecommerce_rec.config import load_config


def _root() -> Path:
    return Path.cwd()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="E-commerce PDP recommender: Two-Tower retrieval + LightGBM ranker"
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config (relative to CWD)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("generate-data", help="Generate synthetic e-commerce catalog + engagement logs")
    sub.add_parser("train-retrieval", help="Train Two-Tower dual encoder")
    sub.add_parser("build-index", help="Encode all items and build FAISS ANN index")
    sub.add_parser("train-ranker", help="Train LightGBM LambdaRank on retrieved candidates")
    sub.add_parser("evaluate", help="Offline NDCG: retrieval-only vs two-stage")
    sub.add_parser("run-all", help="Run full pipeline: data → retrieval → index → ranker → eval")

    demo = sub.add_parser("demo", help="Print top-K PDP recommendations for a user/anchor")
    demo.add_argument("--user-id", type=int, default=None)
    demo.add_argument("--anchor-id", type=int, default=None)

    args = p.parse_args(argv)
    root = _root()
    cfg = load_config(root / args.config, root=root)

    # Lazy imports: avoid loading LightGBM+OpenMP before PyTorch on macOS.
    if args.cmd == "generate-data":
        from ecommerce_rec.data.generate import generate_and_save

        tables = generate_and_save(cfg)
        print(
            f"Generated products={len(tables.products)} users={len(tables.users)} "
            f"events={len(tables.events)} co_edges={len(tables.co_purchase)}"
        )
    elif args.cmd == "train-retrieval":
        from ecommerce_rec.retrieval.train import train_two_tower

        train_two_tower(cfg)
    elif args.cmd == "build-index":
        from ecommerce_rec.retrieval.faiss_index import build_and_persist_index

        build_and_persist_index(cfg)
    elif args.cmd == "train-ranker":
        from ecommerce_rec.ranking.train import train_ranker

        train_ranker(cfg)
    elif args.cmd == "evaluate":
        from ecommerce_rec.ranking.evaluate import evaluate

        evaluate(cfg)
    elif args.cmd == "demo":
        from ecommerce_rec.serving import demo_recommend

        demo_recommend(cfg, user_id=args.user_id, anchor_id=args.anchor_id)
    elif args.cmd == "run-all":
        from ecommerce_rec.data.generate import generate_and_save
        from ecommerce_rec.retrieval.train import train_two_tower
        from ecommerce_rec.retrieval.faiss_index import build_and_persist_index

        print("== 1/5 generate-data ==")
        generate_and_save(cfg)
        print("== 2/5 train-retrieval ==")
        train_two_tower(cfg)
        print("== 3/5 build-index ==")
        build_and_persist_index(cfg)
        print("== 4/5 train-ranker ==")
        from ecommerce_rec.ranking.train import train_ranker

        train_ranker(cfg)
        print("== 5/5 evaluate ==")
        from ecommerce_rec.ranking.evaluate import evaluate

        evaluate(cfg)
        print("== demo ==")
        from ecommerce_rec.serving import demo_recommend

        demo_recommend(cfg)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
