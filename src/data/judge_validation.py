#!/usr/bin/env python3
"""Validate LLM-as-judge outputs against manual labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix


LABEL_ORDER = ["negative", "neutral", "positive"]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_json(path)


def normalize_label(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-labels", default="results/manual_labels_750.csv")
    parser.add_argument("--judge-output", default="results/judge_manual_750.jsonl")
    parser.add_argument("--output-json", default="results/judge_validation.json")
    parser.add_argument("--kappa-threshold", type=float, default=0.70)
    args = parser.parse_args()

    manual = read_table(Path(args.manual_labels))
    judge = read_table(Path(args.judge_output))
    if "review_id" not in manual.columns or "review_id" not in judge.columns:
        raise ValueError("Both files must include review_id")

    manual["manual_label"] = normalize_label(manual["manual_label"])
    judge["judge_label"] = normalize_label(judge["judge_label"])
    merged = manual.merge(judge, on="review_id", how="inner")
    merged = merged[
        merged["manual_label"].isin(LABEL_ORDER)
        & merged["judge_label"].isin(LABEL_ORDER)
    ].copy()
    if merged.empty:
        raise ValueError("No overlapping valid manual/judge labels found")

    y_true = merged["manual_label"]
    y_pred = merged["judge_label"]
    kappa = cohen_kappa_score(y_true, y_pred, labels=LABEL_ORDER)
    accuracy = float((y_true == y_pred).mean())
    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    result = {
        "sample_count": int(len(merged)),
        "accuracy": accuracy,
        "cohen_kappa": float(kappa),
        "kappa_threshold": args.kappa_threshold,
        "passed_gate": bool(kappa >= args.kappa_threshold),
        "labels": LABEL_ORDER,
        "confusion_matrix_manual_rows_judge_cols": cm.tolist(),
        "classification_report": report,
        "mean_judge_confidence": float(merged.get("judge_confidence", pd.Series(dtype=float)).astype(float).mean())
        if "judge_confidence" in merged.columns
        else None,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
