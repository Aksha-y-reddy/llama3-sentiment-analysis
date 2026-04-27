#!/usr/bin/env python3
"""Combine raw data, cleanlab issues, and LLM-judge outputs into cleaned_150k_v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}
PROB_COLS = ["prob_negative", "prob_neutral", "prob_positive"]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_json(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_json(path, orient="records", indent=2)


def ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "review_id" not in df.columns:
        df = df.copy()
        df["review_id"] = [f"row_{i:06d}" for i in range(len(df))]
    return df


def normalize_original_label(df: pd.DataFrame) -> pd.Series:
    if "label" in df.columns:
        return df["label"].apply(
            lambda value: ID_TO_LABEL.get(int(value), str(value).lower())
            if str(value).strip().isdigit()
            else str(value).lower().strip()
        )
    if "rating" in df.columns:
        rating = df["rating"].astype(float)
        return rating.apply(lambda r: "negative" if r <= 2 else "neutral" if r == 3 else "positive")
    raise ValueError("Raw dataset must include label or rating")


def load_cleanlab_flags(path: Path | None, df: pd.DataFrame) -> pd.DataFrame:
    if path is None:
        out = df[["review_id"]].copy()
        out["cleanlab_issue"] = False
        out["cleanlab_rank"] = pd.NA
        out["cleanlab_predicted_label"] = ""
        return out

    issues = read_table(path)
    flags = df[["review_id"]].copy()
    flags["cleanlab_issue"] = False
    flags["cleanlab_rank"] = pd.NA
    flags["cleanlab_predicted_label"] = ""

    if "row_index" in issues.columns:
        issue_map = issues.set_index("row_index")
        for row_index, issue in issue_map.iterrows():
            if int(row_index) < len(flags):
                flags.loc[int(row_index), "cleanlab_issue"] = True
                flags.loc[int(row_index), "cleanlab_rank"] = issue.get("rank", pd.NA)
                flags.loc[int(row_index), "cleanlab_predicted_label"] = issue.get("predicted_label", "")
    elif "review_id" in issues.columns:
        issue_cols = ["review_id"]
        for col in ["rank", "predicted_label"]:
            if col in issues.columns:
                issue_cols.append(col)
        flags = flags.merge(issues[issue_cols], on="review_id", how="left")
        flags["cleanlab_issue"] = flags["rank"].notna()
        flags["cleanlab_rank"] = flags.get("rank", pd.NA)
        flags["cleanlab_predicted_label"] = flags.get("predicted_label", "")
    else:
        raise ValueError("Cleanlab file must include row_index or review_id")
    return flags


def attach_judge(df: pd.DataFrame, judge_path: Path) -> pd.DataFrame:
    judge = read_table(judge_path)
    required = {"review_id", "judge_label", "judge_confidence", *PROB_COLS}
    missing = required - set(judge.columns)
    if missing:
        raise ValueError(f"Judge output missing columns: {sorted(missing)}")
    judge_cols = [
        "review_id",
        "judge_label",
        "judge_confidence",
        "prob_negative",
        "prob_neutral",
        "prob_positive",
        "judge_reasoning",
    ]
    return df.merge(judge[[c for c in judge_cols if c in judge.columns]], on="review_id", how="left")


def decide_clean_labels(df: pd.DataFrame, confidence_threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["original_label_name"] = normalize_original_label(out)
    out["judge_label"] = out["judge_label"].fillna("")
    out["judge_confidence"] = pd.to_numeric(out["judge_confidence"], errors="coerce").fillna(0.0)
    out["judge_available"] = out["judge_label"].isin(LABEL_TO_ID)
    out["judge_agrees_original"] = out["judge_label"] == out["original_label_name"]
    out["judge_high_confidence"] = out["judge_confidence"] >= confidence_threshold

    out["label_source"] = "original"
    out.loc[out["judge_agrees_original"] & out["judge_available"], "label_source"] = "easy_clean"
    out.loc[
        out["judge_available"] & ~out["judge_agrees_original"] & out["judge_high_confidence"],
        "label_source",
    ] = "llm_relabel"
    out.loc[
        out["judge_available"] & ~out["judge_agrees_original"] & ~out["judge_high_confidence"],
        "label_source",
    ] = "soft_ambiguous"

    out["clean_label_name"] = out["original_label_name"]
    relabel_mask = out["label_source"] == "llm_relabel"
    out.loc[relabel_mask, "clean_label_name"] = out.loc[relabel_mask, "judge_label"]
    out["clean_label"] = out["clean_label_name"].map(LABEL_TO_ID)

    # Soft targets: use judge probabilities when present, otherwise one-hot.
    for label_name, col in zip(["negative", "neutral", "positive"], PROB_COLS):
        out[col] = pd.to_numeric(out.get(col, 0.0), errors="coerce").fillna(0.0)
        one_hot = (out["clean_label_name"] == label_name).astype(float)
        out[f"soft_{label_name}"] = out[col].where(out["judge_available"], one_hot)

    total = out[["soft_negative", "soft_neutral", "soft_positive"]].sum(axis=1)
    total = total.mask(total <= 0, 1.0)
    for col in ["soft_negative", "soft_neutral", "soft_positive"]:
        out[col] = out[col] / total
    return out


def write_summary(df: pd.DataFrame, path: Path) -> None:
    summary = {
        "total_samples": int(len(df)),
        "label_source_counts": df["label_source"].value_counts().to_dict(),
        "clean_label_counts": df["clean_label_name"].value_counts().to_dict(),
        "judge_available": int(df["judge_available"].sum()),
        "cleanlab_issue_count": int(df.get("cleanlab_issue", pd.Series(False)).sum()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--judge-output", required=True)
    parser.add_argument("--cleanlab", default=None)
    parser.add_argument("--output", default="data/cleaned_150k_v1.parquet")
    parser.add_argument("--summary", default="results/cleaned_150k_v1_summary.json")
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = ensure_ids(read_table(Path(args.dataset))).copy()
    cleanlab_flags = load_cleanlab_flags(Path(args.cleanlab) if args.cleanlab else None, df)
    df = df.merge(cleanlab_flags, on="review_id", how="left")
    df = attach_judge(df, Path(args.judge_output))
    cleaned = decide_clean_labels(df, args.confidence_threshold)
    write_table(cleaned, Path(args.output))
    write_summary(cleaned, Path(args.summary))
    print(f"Wrote cleaned dataset to {args.output}")
    print(f"Wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
