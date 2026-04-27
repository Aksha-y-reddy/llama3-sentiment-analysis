#!/usr/bin/env python3
"""Analyze the manual-labeling pilot and estimate the 3-star noise rate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


LABEL_ORDER = ["negative", "neutral", "positive"]


def load_manual(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    if "manual_label" not in df.columns:
        raise ValueError("Manual label file must include manual_label column")
    df["manual_label"] = df["manual_label"].astype(str).str.lower().str.strip()
    return df[df["manual_label"].isin(LABEL_ORDER)].copy()


def estimate_noise(df: pd.DataFrame) -> dict:
    random_3star = df[df.get("source", "") == "random_3star"]
    if random_3star.empty and "rating" in df.columns:
        random_3star = df[df["rating"].astype(str).isin(["3", "3.0"])]
    base = random_3star if not random_3star.empty else df

    label_counts = base["manual_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    total = int(label_counts.sum())
    true_neutral = int(label_counts["neutral"])
    noisy = total - true_neutral

    return {
        "evaluation_subset": "random_3star" if not random_3star.empty else "all_labeled",
        "total_labeled_used_for_noise_rate": total,
        "manual_label_counts": {k: int(v) for k, v in label_counts.items()},
        "true_neutral_rate": true_neutral / total if total else 0.0,
        "estimated_noise_rate": noisy / total if total else 0.0,
        "textually_negative_rate": int(label_counts["negative"]) / total if total else 0.0,
        "textually_positive_rate": int(label_counts["positive"]) / total if total else 0.0,
    }


def analyze_cleanlab_agreement(df: pd.DataFrame) -> dict:
    flagged = df[df.get("source", "") == "cleanlab_flagged"]
    if flagged.empty:
        return {"available": False}
    counts = flagged["manual_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    total = int(counts.sum())
    return {
        "available": True,
        "flagged_labeled_count": total,
        "flagged_manual_label_counts": {k: int(v) for k, v in counts.items()},
        "flagged_non_neutral_rate": (total - int(counts["neutral"])) / total if total else 0.0,
    }


def analyze_original_vs_manual(df: pd.DataFrame) -> dict:
    if "original_label" not in df.columns:
        return {"available": False}
    original = df["original_label"].astype(str).str.lower().str.strip()
    manual = df["manual_label"].astype(str).str.lower().str.strip()
    valid = original.isin(LABEL_ORDER) & manual.isin(LABEL_ORDER)
    if valid.sum() == 0:
        return {"available": False}
    cm = confusion_matrix(original[valid], manual[valid], labels=LABEL_ORDER)
    return {
        "available": True,
        "cohen_kappa_original_vs_manual": float(cohen_kappa_score(original[valid], manual[valid])),
        "confusion_matrix_labels": LABEL_ORDER,
        "confusion_matrix_original_rows_manual_cols": cm.tolist(),
    }


def write_markdown(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    noise = summary["noise_rate"]
    cleanlab = summary["cleanlab_agreement"]
    original = summary["original_vs_manual"]

    lines = [
        "# Neutral-Class Noise Rate Findings",
        "",
        "This report is generated from the manual-labeling pilot. The manual label is treated as the text-based ground truth, while the original label comes from the Amazon star rating.",
        "",
        "## Summary",
        "",
        f"- Samples used for noise-rate estimate: {noise['total_labeled_used_for_noise_rate']}",
        f"- Estimated true-neutral rate: {noise['true_neutral_rate']:.2%}",
        f"- Estimated 3-star label-noise rate: {noise['estimated_noise_rate']:.2%}",
        f"- Textually negative among 3-star/manual subset: {noise['textually_negative_rate']:.2%}",
        f"- Textually positive among 3-star/manual subset: {noise['textually_positive_rate']:.2%}",
        "",
        "## Manual Label Counts",
        "",
        "| Manual Label | Count |",
        "|---|---:|",
    ]
    for label, count in noise["manual_label_counts"].items():
        lines.append(f"| {label} | {count} |")

    if cleanlab.get("available"):
        lines.extend(
            [
                "",
                "## Cleanlab-Flagged Samples",
                "",
                f"- Labeled cleanlab-flagged samples: {cleanlab['flagged_labeled_count']}",
                f"- Non-neutral rate among flagged samples: {cleanlab['flagged_non_neutral_rate']:.2%}",
                "",
                "| Manual Label | Count |",
                "|---|---:|",
            ]
        )
        for label, count in cleanlab["flagged_manual_label_counts"].items():
            lines.append(f"| {label} | {count} |")

    if original.get("available"):
        lines.extend(
            [
                "",
                "## Original Label vs Manual Label",
                "",
                f"- Cohen's kappa: {original['cohen_kappa_original_vs_manual']:.4f}",
                "",
                "Rows are original labels, columns are manual labels.",
                "",
                "| Original \\ Manual | Negative | Neutral | Positive |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, row in zip(
            original["confusion_matrix_labels"],
            original["confusion_matrix_original_rows_manual_cols"],
        ):
            lines.append(f"| {label} | {row[0]} | {row[1]} | {row[2]} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-labels", default="results/manual_labels_750.csv")
    parser.add_argument("--output-json", default="results/true_noise_rate.json")
    parser.add_argument("--output-md", default="docs/reports/NOISE_RATE_FINDINGS.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_manual(Path(args.manual_labels))
    summary = {
        "noise_rate": estimate_noise(df),
        "cleanlab_agreement": analyze_cleanlab_agreement(df),
        "original_vs_manual": analyze_original_vs_manual(df),
    }
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, Path(args.output_md))
    print(f"Wrote {json_path}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
