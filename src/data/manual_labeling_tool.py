#!/usr/bin/env python3
"""Create and label the 750-sample neutral-class gold-standard pilot.

Commands:

1. Create a labeling file:
   python src/data/manual_labeling_tool.py init \
     --dataset data/raw_150k.parquet \
     --cleanlab results/cleanlab_audit.json \
     --output results/manual_labels_750.csv

2. Label samples in the terminal:
   python src/data/manual_labeling_tool.py label \
     --input results/manual_labels_750.csv

The tool never uses star ratings as the manual answer. The human label should
reflect the review text only.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd


LABELS = {"n": "negative", "u": "neutral", "p": "positive", "s": "skip"}
CONFIDENCE = {"h": "high", "m": "medium", "l": "low"}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    return pd.read_json(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def star_label(value: float | int | str) -> str:
    rating = float(value)
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def ensure_review_id(df: pd.DataFrame) -> pd.DataFrame:
    if "review_id" not in df.columns:
        df = df.copy()
        df["review_id"] = [f"sample_{i:06d}" for i in range(len(df))]
    return df


def init_labeling_file(args: argparse.Namespace) -> None:
    dataset = ensure_review_id(read_table(Path(args.dataset)))
    required = {"text"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    if "rating" not in dataset.columns and "label" not in dataset.columns:
        raise ValueError("Dataset must include either rating or label")

    if "rating" in dataset.columns:
        three_star = dataset[dataset["rating"].astype(float) == 3.0].copy()
    else:
        three_star = dataset[dataset["label"].astype(str).str.lower().isin(["1", "neutral", "neu"])].copy()

    random_part = three_star.sample(
        n=min(args.random_count, len(three_star)),
        random_state=args.seed,
        replace=False,
    )

    cleanlab_part = pd.DataFrame()
    if args.cleanlab:
        issues = read_table(Path(args.cleanlab))
        if "row_index" in issues.columns:
            flagged = dataset.iloc[issues["row_index"].astype(int).head(args.flagged_count)].copy()
        elif "review_id" in issues.columns:
            flagged_ids = issues["review_id"].head(args.flagged_count)
            flagged = dataset[dataset["review_id"].isin(flagged_ids)].copy()
        else:
            raise ValueError("Cleanlab file must include row_index or review_id")
        cleanlab_part = flagged

    pilot = pd.concat([random_part, cleanlab_part], ignore_index=True)
    pilot = pilot.drop_duplicates(subset=["review_id"]).head(
        args.random_count + args.flagged_count
    )

    out = pd.DataFrame()
    out["sample_id"] = [f"manual_{i:04d}" for i in range(len(pilot))]
    out["source"] = [
        "random_3star" if i < len(random_part) else "cleanlab_flagged"
        for i in range(len(pilot))
    ]
    out["review_id"] = pilot["review_id"].values
    out["category"] = pilot["category"].values if "category" in pilot.columns else ""
    out["rating"] = pilot["rating"].values if "rating" in pilot.columns else ""
    out["original_label"] = (
        [star_label(v) for v in pilot["rating"].values]
        if "rating" in pilot.columns
        else pilot.get("label", "")
    )
    out["title"] = pilot["title"].values if "title" in pilot.columns else ""
    out["text"] = pilot["text"].fillna("").astype(str).values
    out["manual_label"] = ""
    out["confidence"] = ""
    out["notes"] = ""

    write_csv(out, Path(args.output))
    print(f"Wrote {len(out):,} samples to {args.output}")


def render_sample(row: pd.Series, width: int = 100) -> None:
    print("\n" + "=" * width)
    print(f"sample_id: {row.get('sample_id', '')}")
    print(f"source: {row.get('source', '')}")
    print(f"rating/original: {row.get('rating', '')} / {row.get('original_label', '')}")
    title = str(row.get("title", "") or "").strip()
    if title:
        print(f"title: {title}")
    print("-" * width)
    print(textwrap.fill(str(row.get("text", "")), width=width))
    print("=" * width)


def label_file(args: argparse.Namespace) -> None:
    path = Path(args.input)
    df = pd.read_csv(path).fillna("")

    for idx, row in df.iterrows():
        if row.get("manual_label"):
            continue
        render_sample(row)
        label_key = input("Label [n=negative, u=neutral, p=positive, s=skip, q=quit]: ").strip().lower()
        if label_key == "q":
            break
        if label_key not in LABELS:
            print("Invalid label, skipping this sample for now.")
            continue
        if LABELS[label_key] == "skip":
            df.loc[idx, "manual_label"] = "skip"
            df.loc[idx, "confidence"] = "low"
            df.loc[idx, "notes"] = "Skipped during manual labeling"
        else:
            conf_key = input("Confidence [h=high, m=medium, l=low]: ").strip().lower()
            if conf_key not in CONFIDENCE:
                conf_key = "m"
            notes = input("Notes (optional): ").strip()
            df.loc[idx, "manual_label"] = LABELS[label_key]
            df.loc[idx, "confidence"] = CONFIDENCE[conf_key]
            df.loc[idx, "notes"] = notes
        write_csv(df, path)

    completed = (df["manual_label"].astype(str).str.len() > 0).sum()
    print(f"Progress: {completed:,}/{len(df):,} samples labeled")


def status(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input).fillna("")
    labeled = df[df["manual_label"].astype(str).str.len() > 0]
    print(f"Total samples: {len(df):,}")
    print(f"Labeled: {len(labeled):,}")
    if not labeled.empty:
        print("\nManual labels:")
        print(labeled["manual_label"].value_counts(dropna=False).to_string())
        print("\nConfidence:")
        print(labeled["confidence"].value_counts(dropna=False).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create the manual-labeling CSV")
    init.add_argument("--dataset", required=True)
    init.add_argument("--cleanlab", default=None)
    init.add_argument("--output", default="results/manual_labels_750.csv")
    init.add_argument("--random-count", type=int, default=500)
    init.add_argument("--flagged-count", type=int, default=250)
    init.add_argument("--seed", type=int, default=42)

    label = sub.add_parser("label", help="Label samples in terminal")
    label.add_argument("--input", default="results/manual_labels_750.csv")

    stat = sub.add_parser("status", help="Show labeling progress")
    stat.add_argument("--input", default="results/manual_labels_750.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        init_labeling_file(args)
    elif args.command == "label":
        label_file(args)
    elif args.command == "status":
        status(args)


if __name__ == "__main__":
    main()
