"""End-to-end pipeline: load → preprocess → train both models → evaluate →
save locally → push to HF Hub.

Usage:
    python scripts/train_all.py                  # full pipeline
    python scripts/train_all.py --sample 500000  # subsample PaySim for speed
    python scripts/train_all.py --no-push        # skip HF upload
"""
from __future__ import annotations

import argparse
import json
import sys

from src.config import MODELS_LOCAL_DIR
from src.evaluation.hypothesis_test import mcnemar_test
from src.hf_hub.uploader import push_all
from src.models.trainer import train_all


def _format(metrics: dict, indent: int = 0) -> str:
    pad = "  " * indent
    keep = ("accuracy", "precision", "recall", "f1", "roc_auc",
            "true_positives", "false_positives", "false_negatives", "true_negatives")
    return "\n".join(f"{pad}{k}: {metrics.get(k)}" for k in keep if k in metrics)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample PaySim to N rows (keeps all fraud)")
    parser.add_argument("--no-push", action="store_true",
                        help="Skip pushing to Hugging Face Hub")
    args = parser.parse_args()

    print("=" * 60)
    print("Training fraud detection models")
    print("=" * 60)
    metadata = train_all(sample_paysim=args.sample)

    print("\n[ Random Forest on PaySim ]")
    print(_format(metadata["paysim"]["rf"], indent=1))
    print("\n[ Rule-based baseline on PaySim ]")
    print(_format(metadata["paysim"]["rule_based"], indent=1))
    print("\n[ Isolation Forest on custom dataset ]")
    print(_format(metadata["custom"]["isolation_forest"], indent=1))
    print("\n[ Rule-based baseline on custom dataset ]")
    print(_format(metadata["custom"]["rule_based"], indent=1))

    # Persist consolidated metadata with model-card-friendly format
    out_file = MODELS_LOCAL_DIR / "metadata.json"
    out_file.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\nMetadata saved -> {out_file}")

    if args.no_push:
        print("\n[--no-push] Skipping HF Hub upload.")
        return 0

    print("\nPushing artifacts to Hugging Face Hub…")
    push_result = push_all()
    print(f"  repo: {push_result['repo']}")
    print(f"  uploaded: {push_result['uploaded']}")
    if push_result["skipped"]:
        print(f"  skipped (missing): {push_result['skipped']}")
    print(f"\nDone. View at https://huggingface.co/{push_result['repo']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
