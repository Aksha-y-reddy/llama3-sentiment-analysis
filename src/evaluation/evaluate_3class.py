#!/usr/bin/env python3
"""Evaluate 3-class sentiment predictions or a saved sequence-classification model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    return pd.read_json(path)


def normalize_label(value: Any) -> int:
    text = str(value).lower().strip()
    if text in LABEL_TO_ID:
        return LABEL_TO_ID[text]
    return int(float(text))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "ordinal_mae": float(mean_absolute_error(y_true, y_pred)),
        "per_class": {
            label: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=LABELS,
            labels=[0, 1, 2],
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_predictions(args: argparse.Namespace) -> dict:
    df = read_table(Path(args.predictions))
    if args.label_col not in df.columns or args.pred_col not in df.columns:
        raise ValueError(f"Predictions file must include {args.label_col} and {args.pred_col}")
    y_true = df[args.label_col].apply(normalize_label).to_numpy()
    y_pred = df[args.pred_col].apply(normalize_label).to_numpy()
    return metrics(y_true, y_pred)


def evaluate_model(args: argparse.Namespace) -> dict:
    df = read_table(Path(args.dataset)).fillna("")
    if args.label_col not in df.columns:
        raise ValueError(f"Dataset missing label column: {args.label_col}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    labels = df[args.label_col].apply(normalize_label).to_numpy()
    title_series = df["title"].astype(str) if "title" in df.columns else pd.Series([""] * len(df))
    texts = (
        "Title: "
        + title_series
        + "\nReview: "
        + df["text"].astype(str)
        + "\nSentiment:"
    ).tolist()

    preds: list[int] = []
    for start in range(0, len(texts), args.batch_size):
        batch_texts = texts[start : start + args.batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())

    return metrics(labels, np.asarray(preds))


def write_summary(result: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path = output.with_suffix(".md")
    lines = [
        "# 3-Class Evaluation Results",
        "",
        f"- Accuracy: {result['accuracy']:.4f}",
        f"- Macro F1: {result['macro_f1']:.4f}",
        f"- Ordinal MAE: {result['ordinal_mae']:.4f}",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in result["per_class"].items():
        lines.append(
            f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} | "
            f"{values['f1']:.4f} | {values['support']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="", help="File with true/pred labels")
    parser.add_argument("--dataset", default="", help="Eval dataset for model inference")
    parser.add_argument("--model", default="", help="Saved sequence-classification model")
    parser.add_argument("--output", default="results/evaluation_3class.json")
    parser.add_argument("--label-col", default="manual_label")
    parser.add_argument("--pred-col", default="prediction")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.predictions:
        result = evaluate_predictions(args)
    elif args.dataset and args.model:
        result = evaluate_model(args)
    else:
        raise ValueError("Provide either --predictions or both --dataset and --model")
    write_summary(result, Path(args.output))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
