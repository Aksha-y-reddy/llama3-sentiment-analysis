#!/usr/bin/env python3
"""Export the same balanced 150K 3-class dataset used by the baseline notebook.

This mirrors the data-loading logic in:
    notebooks/01_baselines/02_baseline_150k_sequential_training.ipynb

For the neutral-class fix pipeline we preserve extra metadata that the notebook
removed before training: rating, category, hash, title, split, and review_id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def compute_sample_hash(text: str, rating: float) -> str:
    content = f"{text}_{rating}"
    return hashlib.sha256(content.encode()).hexdigest()


def load_tracking(path: Path, category: str, num_classes: int) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "category": category,
        "num_classes": num_classes,
        "used_hashes": [],
        "baseline_count": 0,
        "sequential_count": 0,
    }


def save_tracking(path: Path, tracking: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tracking, indent=2), encoding="utf-8")


def sample_to_record(
    review: dict[str, Any],
    category: str,
    rating: float,
    label: int,
    sample_hash: str,
) -> dict[str, Any]:
    text = review.get("text", "") or ""
    title = review.get("title", "") or ""
    return {
        "review_id": sample_hash,
        "category": category,
        "rating": rating,
        "title": title,
        "text": text,
        "label": label,
        "hash": sample_hash,
        "asin": review.get("asin", ""),
        "user_id": review.get("user_id", ""),
        "timestamp": review.get("timestamp", ""),
    }


def collect_balanced_samples(
    category: str,
    num_classes: int,
    train_per_class: int,
    eval_per_class: int,
    training_phase: str,
    tracking_path: Path,
    min_text_chars: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    print("\n" + "=" * 70)
    print(f"LOADING DATA: {category} ({num_classes}-class, {training_phase} phase)")
    print("=" * 70)

    tracking = load_tracking(tracking_path, category, num_classes)
    used_hashes = set(tracking.get("used_hashes", []))

    file_path = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename=f"raw/review_categories/{category}.jsonl",
        repo_type="dataset",
    )

    negative_samples: list[dict[str, Any]] = []
    neutral_samples: list[dict[str, Any]] = []
    positive_samples: list[dict[str, Any]] = []
    target_per_class = int((train_per_class + eval_per_class) * 1.2)

    total_read = 0
    skipped_used = 0
    skipped_short = 0

    print("\nProcessing reviews...")
    print(f"  Existing tracked: {len(used_hashes):,} samples")
    print(f"  Target per class with buffer: {target_per_class:,}")

    with open(file_path, "r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Reading"):
            if num_classes == 2:
                enough = len(negative_samples) >= target_per_class and len(positive_samples) >= target_per_class
            else:
                enough = (
                    len(negative_samples) >= target_per_class
                    and len(neutral_samples) >= target_per_class
                    and len(positive_samples) >= target_per_class
                )
            if enough:
                break

            try:
                review = json.loads(line)
                total_read += 1
                rating = float(review.get("rating", 3.0))
                text = review.get("text", "") or ""
                if len(text.strip()) <= min_text_chars:
                    skipped_short += 1
                    continue

                sample_hash = compute_sample_hash(text, rating)
                if training_phase == "sequential" and sample_hash in used_hashes:
                    skipped_used += 1
                    continue

                if rating <= 2.0 and len(negative_samples) < target_per_class:
                    negative_samples.append(sample_to_record(review, category, rating, 0, sample_hash))
                elif rating == 3.0 and num_classes == 3 and len(neutral_samples) < target_per_class:
                    neutral_samples.append(sample_to_record(review, category, rating, 1, sample_hash))
                elif rating >= 4.0 and len(positive_samples) < target_per_class:
                    label = 1 if num_classes == 2 else 2
                    positive_samples.append(sample_to_record(review, category, rating, label, sample_hash))
            except Exception:
                continue

    print("\nData Collection Stats:")
    print(f"  Total reviews read: {total_read:,}")
    print(f"  Skipped too short:  {skipped_short:,}")
    print(f"  Skipped used:       {skipped_used:,}")
    print(
        f"  Collected: neg={len(negative_samples):,}, "
        f"neu={len(neutral_samples):,}, pos={len(positive_samples):,}"
    )

    random.seed(seed)
    samples_per_class = train_per_class + eval_per_class
    if num_classes == 2:
        samples_per_class = min(samples_per_class, len(negative_samples), len(positive_samples))
        random.shuffle(negative_samples)
        random.shuffle(positive_samples)
        all_samples = negative_samples[:samples_per_class] + positive_samples[:samples_per_class]
    else:
        samples_per_class = min(
            samples_per_class,
            len(negative_samples),
            len(neutral_samples),
            len(positive_samples),
        )
        random.shuffle(negative_samples)
        random.shuffle(neutral_samples)
        random.shuffle(positive_samples)
        all_samples = (
            negative_samples[:samples_per_class]
            + neutral_samples[:samples_per_class]
            + positive_samples[:samples_per_class]
        )
    random.shuffle(all_samples)

    eval_size = eval_per_class * num_classes
    train_samples = all_samples[:-eval_size]
    eval_samples = all_samples[-eval_size:]
    for sample in train_samples:
        sample["split"] = "train"
    for sample in eval_samples:
        sample["split"] = "eval"

    tracking["used_hashes"].extend(sample["hash"] for sample in all_samples)
    if training_phase == "baseline":
        tracking["baseline_count"] = len(all_samples)
    else:
        tracking["sequential_count"] = len(all_samples)
    save_tracking(tracking_path, tracking)

    df = pd.DataFrame(train_samples + eval_samples)
    print("\nFinal dataset:")
    print(f"  Train: {(df['split'] == 'train').sum():,}")
    print(f"  Eval:  {(df['split'] == 'eval').sum():,}")
    print(f"  Total: {len(df):,}")
    return df, tracking


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_json(path, orient="records", indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw_150k.parquet")
    parser.add_argument("--category", default="Cell_Phones_and_Accessories")
    parser.add_argument("--num-classes", type=int, default=3, choices=[2, 3])
    parser.add_argument("--train-per-class", type=int, default=50_000)
    parser.add_argument("--eval-per-class", type=int, default=5_000)
    parser.add_argument("--training-phase", choices=["baseline", "sequential"], default="baseline")
    parser.add_argument("--tracking-file", default="data/llama3-data-tracking-Cell_Phones_and_Accessories-3class.json")
    parser.add_argument("--min-text-chars", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_classes == 2 and args.train_per_class == 50_000:
        args.train_per_class = 75_000
    df, tracking = collect_balanced_samples(
        category=args.category,
        num_classes=args.num_classes,
        train_per_class=args.train_per_class,
        eval_per_class=args.eval_per_class,
        training_phase=args.training_phase,
        tracking_path=Path(args.tracking_file),
        min_text_chars=args.min_text_chars,
        seed=args.seed,
    )
    write_table(df, Path(args.output))
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "output": args.output,
                "category": args.category,
                "num_classes": args.num_classes,
                "training_phase": args.training_phase,
                "train_per_class": args.train_per_class,
                "eval_per_class": args.eval_per_class,
                "rows": len(df),
                "split_counts": df["split"].value_counts().to_dict(),
                "label_counts": df["label"].value_counts().sort_index().to_dict(),
                "tracked_hashes": len(tracking.get("used_hashes", [])),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
