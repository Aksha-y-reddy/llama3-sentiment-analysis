#!/usr/bin/env python3
"""Run a Confident Learning audit for the 3-class sentiment dataset.

This script is intentionally model-agnostic:

1. If you already have out-of-fold probabilities from a LLaMA/QLoRA model, pass
   them with ``--pred-probs`` and the script will run cleanlab directly.
2. If not, the script can build a lightweight TF-IDF + logistic regression
   baseline to produce out-of-fold probabilities. This is much cheaper than
   running 5-fold LLaMA on Colab and is sufficient for the first data-quality
   audit.

Expected input columns:
    - text
    - label (0=negative, 1=neutral, 2=positive)

Optional useful columns are preserved in the output:
    review_id, category, rating, title
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

try:
    from cleanlab.filter import find_label_issues
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "cleanlab is required. Install it with: pip install cleanlab"
    ) from exc


LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        df.to_json(path, orient="records", indent=2)


def normalize_labels(labels: Iterable[object]) -> np.ndarray:
    mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
        "neg": 0,
        "neu": 1,
        "pos": 2,
    }
    normalized: list[int] = []
    for value in labels:
        if isinstance(value, str):
            key = value.strip().lower()
            if key not in mapping:
                raise ValueError(f"Unknown label string: {value}")
            normalized.append(mapping[key])
        else:
            normalized.append(int(value))
    arr = np.asarray(normalized, dtype=int)
    if not set(arr.tolist()).issubset({0, 1, 2}):
        raise ValueError("Labels must map to 0=negative, 1=neutral, 2=positive")
    return arr


def load_pred_probs(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        probs = np.load(path)
    else:
        df = read_table(path)
        expected = ["prob_negative", "prob_neutral", "prob_positive"]
        if not all(col in df.columns for col in expected):
            raise ValueError(
                f"Prediction file must include columns {expected} or be a .npy array"
            )
        probs = df[expected].to_numpy(dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("pred_probs must have shape (n_samples, 3)")
    row_sums = probs.sum(axis=1, keepdims=True)
    return probs / np.clip(row_sums, 1e-12, None)


def compute_tfidf_oof_probs(
    texts: pd.Series,
    labels: np.ndarray,
    folds: int,
    seed: int,
    max_features: int,
) -> np.ndarray:
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=2,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=seed,
                    multi_class="auto",
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(
        pipeline,
        texts.fillna("").astype(str),
        labels,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )


def build_issue_frame(
    df: pd.DataFrame,
    labels: np.ndarray,
    pred_probs: np.ndarray,
    ranked_indices: np.ndarray,
) -> pd.DataFrame:
    pred_labels = pred_probs.argmax(axis=1)
    self_conf = pred_probs[np.arange(len(labels)), labels]
    margin = pred_probs.max(axis=1) - self_conf

    base_cols = [
        col
        for col in ["review_id", "category", "rating", "title", "text"]
        if col in df.columns
    ]
    issue_df = df.iloc[ranked_indices][base_cols].copy()
    issue_df.insert(0, "rank", np.arange(1, len(issue_df) + 1))
    issue_df["row_index"] = ranked_indices
    issue_df["given_label"] = [LABEL_NAMES[int(labels[i])] for i in ranked_indices]
    issue_df["predicted_label"] = [
        LABEL_NAMES[int(pred_labels[i])] for i in ranked_indices
    ]
    issue_df["self_confidence"] = self_conf[ranked_indices]
    issue_df["confidence_margin"] = margin[ranked_indices]
    issue_df["prob_negative"] = pred_probs[ranked_indices, 0]
    issue_df["prob_neutral"] = pred_probs[ranked_indices, 1]
    issue_df["prob_positive"] = pred_probs[ranked_indices, 2]
    return issue_df


def write_summary(
    output_path: Path,
    labels: np.ndarray,
    issue_df: pd.DataFrame,
    method: str,
    pred_probs_path: str | None,
) -> None:
    summary_path = output_path.with_suffix(".summary.json")
    by_given = (
        issue_df["given_label"].value_counts().rename_axis("label").to_dict()
        if not issue_df.empty
        else {}
    )
    by_flow = (
        issue_df.groupby(["given_label", "predicted_label"]).size().to_dict()
        if not issue_df.empty
        else {}
    )
    summary = {
        "method": method,
        "pred_probs_path": pred_probs_path,
        "total_samples": int(len(labels)),
        "issue_count": int(len(issue_df)),
        "issue_rate": float(len(issue_df) / max(len(labels), 1)),
        "class_distribution": {
            LABEL_NAMES[int(label)]: int(count)
            for label, count in zip(*np.unique(labels, return_counts=True))
        },
        "issues_by_given_label": by_given,
        "issues_by_flow": {str(key): int(value) for key, value in by_flow.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV/parquet/jsonl dataset path")
    parser.add_argument("--output", default="results/cleanlab_audit.json")
    parser.add_argument("--pred-probs", default=None, help="Optional .npy or table")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--top-k", type=int, default=0, help="0 keeps all issues")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df = read_table(input_path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    labels = normalize_labels(df["label"])
    if args.pred_probs:
        pred_probs = load_pred_probs(Path(args.pred_probs))
        method = "precomputed_pred_probs"
    else:
        pred_probs = compute_tfidf_oof_probs(
            df["text"],
            labels,
            folds=args.folds,
            seed=args.seed,
            max_features=args.max_features,
        )
        method = "tfidf_logreg_oof"

    if len(pred_probs) != len(df):
        raise ValueError("pred_probs row count does not match input dataset")

    ranked_indices = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    if args.top_k > 0:
        ranked_indices = ranked_indices[: args.top_k]

    issue_df = build_issue_frame(df, labels, pred_probs, ranked_indices)
    write_table(issue_df, output_path)
    write_summary(output_path, labels, issue_df, method, args.pred_probs)

    print(f"Wrote {len(issue_df):,} ranked label issues to {output_path}")
    print(f"Wrote summary to {output_path.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
