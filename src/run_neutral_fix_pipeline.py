#!/usr/bin/env python3
"""One-command Colab runner for the neutral-class fix pipeline.

This script orchestrates the scripts in ``src/data``, ``src/training``, and
``src/evaluation``. It is designed for Google Colab where long jobs may
disconnect, so every stage is resumable and skipped if its expected output
already exists.

Typical Colab usage:

    python src/run_neutral_fix_pipeline.py \
      --raw-dataset data/raw_150k.parquet \
      --provider together \
      --train

Required before running:
    1. Create ``data/raw_150k.parquet`` with columns:
       review_id, category, rating, title, text, label
    2. Set API key if running judge stages:
       import os; os.environ["TOGETHER_API_KEY"] = "..."

Manual labeling is the only unavoidable human step. The runner creates
``results/manual_labels_750.csv`` and pauses until it is complete unless
``--no-pause`` is provided.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULTS = {
    "cleanlab": "results/cleanlab_audit.json",
    "manual": "results/manual_labels_750.csv",
    "noise": "results/true_noise_rate.json",
    "judge_manual": "results/judge_manual_750.jsonl",
    "judge_validation": "results/judge_validation.json",
    "three_star": "data/three_star_reviews.parquet",
    "judge_50k": "results/llm_judge_50k.jsonl",
    "cleaned": "data/cleaned_150k_v1.parquet",
    "cleaned_summary": "results/cleaned_150k_v1_summary.json",
    "models": "results/models",
}


def run(command: list[str], dry_run: bool = False) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def exists(path: str | Path) -> bool:
    return Path(path).exists()


def require(path: str | Path, message: str) -> None:
    if not exists(path):
        raise SystemExit(f"Missing required file: {path}\n{message}")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    return pd.read_json(path)


def manual_labels_complete(path: Path, minimum: int) -> bool:
    if not path.exists():
        return False
    df = pd.read_csv(path).fillna("")
    if "manual_label" not in df.columns:
        return False
    completed = df["manual_label"].astype(str).str.len().gt(0).sum()
    print(f"Manual labeling progress: {completed}/{len(df)}")
    return completed >= minimum


def judge_validation_passed(path: Path, threshold: float) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    kappa = float(data.get("cohen_kappa", 0.0))
    passed = bool(data.get("passed_gate", False)) and kappa >= threshold
    print(f"Judge validation kappa: {kappa:.4f} (passed={passed})")
    return passed


def create_three_star_subset(raw_dataset: Path, output: Path) -> None:
    if output.exists():
        print(f"Skipping 3-star subset; exists: {output}")
        return
    df = read_table(raw_dataset)
    if "rating" in df.columns:
        subset = df[df["rating"].astype(float) == 3.0].copy()
    elif "label" in df.columns:
        labels = df["label"].astype(str).str.lower().str.strip()
        subset = df[labels.isin(["1", "neutral", "neu"])].copy()
    else:
        raise ValueError("Raw dataset must include rating or label")
    output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(output, index=False)
    print(f"Wrote {len(subset):,} 3-star/neutral rows to {output}")


def maybe_install_requirements(args: argparse.Namespace) -> None:
    if args.install:
        run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], args.dry_run)


def run_pipeline(args: argparse.Namespace) -> None:
    raw_dataset = Path(args.raw_dataset)
    require(
        raw_dataset,
        "Create it first from the baseline notebook/data loader. Required columns: "
        "review_id, category, rating, title, text, label.",
    )

    maybe_install_requirements(args)

    if not exists(args.cleanlab_output) or args.force:
        run(
            [
                sys.executable,
                "src/data/cleanlab_audit.py",
                "--input",
                args.raw_dataset,
                "--output",
                args.cleanlab_output,
                "--folds",
                str(args.cleanlab_folds),
            ],
            args.dry_run,
        )
    else:
        print(f"Skipping cleanlab audit; exists: {args.cleanlab_output}")

    if not exists(args.manual_labels) or args.force:
        run(
            [
                sys.executable,
                "src/data/manual_labeling_tool.py",
                "init",
                "--dataset",
                args.raw_dataset,
                "--cleanlab",
                args.cleanlab_output,
                "--output",
                args.manual_labels,
            ],
            args.dry_run,
        )
    else:
        print(f"Skipping manual-label init; exists: {args.manual_labels}")

    if not manual_labels_complete(Path(args.manual_labels), args.min_manual_labels):
        print(
            "\nManual labeling is not complete. Run this in a Colab/terminal cell:\n"
            f"python src/data/manual_labeling_tool.py label --input {args.manual_labels}\n"
        )
        if not args.no_pause and not args.dry_run:
            input("Press Enter after manual labeling is complete, or Ctrl+C to stop and resume later...")
        if not manual_labels_complete(Path(args.manual_labels), args.min_manual_labels):
            raise SystemExit("Stopping because manual labels are incomplete.")

    if not exists(args.noise_output) or args.force:
        run(
            [
                sys.executable,
                "src/data/noise_rate_analysis.py",
                "--manual-labels",
                args.manual_labels,
                "--output-json",
                args.noise_output,
                "--output-md",
                "docs/reports/NOISE_RATE_FINDINGS.md",
            ],
            args.dry_run,
        )
    else:
        print(f"Skipping noise-rate analysis; exists: {args.noise_output}")

    if args.skip_judge:
        print("Skipping all judge stages because --skip-judge was provided.")
        return

    if args.provider == "together" and not os.environ.get("TOGETHER_API_KEY") and not args.dry_run:
        raise SystemExit("TOGETHER_API_KEY is required for --provider together")
    if args.provider == "huggingface" and not os.environ.get("HF_TOKEN") and not args.dry_run:
        raise SystemExit("HF_TOKEN is required for --provider huggingface")

    if not exists(args.judge_manual_output) or args.force:
        run(
            [
                sys.executable,
                "src/data/llm_judge_client.py",
                "--input",
                args.manual_labels,
                "--output",
                args.judge_manual_output,
                "--provider",
                args.provider,
            ],
            args.dry_run,
        )
    else:
        print(f"Skipping manual judge pass; exists: {args.judge_manual_output}")

    if not exists(args.judge_validation_output) or args.force:
        run(
            [
                sys.executable,
                "src/data/judge_validation.py",
                "--manual-labels",
                args.manual_labels,
                "--judge-output",
                args.judge_manual_output,
                "--output-json",
                args.judge_validation_output,
                "--kappa-threshold",
                str(args.kappa_threshold),
            ],
            args.dry_run,
        )
    else:
        print(f"Skipping judge validation; exists: {args.judge_validation_output}")

    if not args.dry_run and not judge_validation_passed(Path(args.judge_validation_output), args.kappa_threshold):
        raise SystemExit("Stopping because judge validation did not pass. Iterate prompt/model first.")

    create_three_star_subset(raw_dataset, Path(args.three_star_dataset))

    if not exists(args.judge_50k_output) or args.force:
        run(
            [
                sys.executable,
                "src/data/llm_judge_client.py",
                "--input",
                args.three_star_dataset,
                "--output",
                args.judge_50k_output,
                "--provider",
                args.provider,
            ],
            args.dry_run,
        )
    else:
        print(f"Skipping full 3-star judge pass; exists: {args.judge_50k_output}")

    if not exists(args.cleaned_dataset) or args.force:
        run(
            [
                sys.executable,
                "src/data/cross_validation_pipeline.py",
                "--dataset",
                args.raw_dataset,
                "--judge-output",
                args.judge_50k_output,
                "--cleanlab",
                args.cleanlab_output,
                "--output",
                args.cleaned_dataset,
                "--summary",
                args.cleaned_summary,
            ],
            args.dry_run,
        )
    else:
        print(f"Skipping cleaned dataset build; exists: {args.cleaned_dataset}")

    if args.train:
        run(
            [
                sys.executable,
                "src/training/run_ablation_variants.py",
                "--raw-dataset",
                args.raw_dataset,
                "--cleaned-dataset",
                args.cleaned_dataset,
                "--output-root",
                args.models_output,
            ]
            + (["--limit", str(args.train_limit)] if args.train_limit else [])
            + (["--eval-limit", str(args.eval_limit)] if args.eval_limit else []),
            args.dry_run,
        )
    else:
        print("\nTraining skipped. Add --train when ready to launch variants A/B/C.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dataset", default="data/raw_150k.parquet")
    parser.add_argument("--cleanlab-output", default=DEFAULTS["cleanlab"])
    parser.add_argument("--manual-labels", default=DEFAULTS["manual"])
    parser.add_argument("--noise-output", default=DEFAULTS["noise"])
    parser.add_argument("--judge-manual-output", default=DEFAULTS["judge_manual"])
    parser.add_argument("--judge-validation-output", default=DEFAULTS["judge_validation"])
    parser.add_argument("--three-star-dataset", default=DEFAULTS["three_star"])
    parser.add_argument("--judge-50k-output", default=DEFAULTS["judge_50k"])
    parser.add_argument("--cleaned-dataset", default=DEFAULTS["cleaned"])
    parser.add_argument("--cleaned-summary", default=DEFAULTS["cleaned_summary"])
    parser.add_argument("--models-output", default=DEFAULTS["models"])
    parser.add_argument("--provider", choices=["together", "huggingface"], default="together")
    parser.add_argument("--cleanlab-folds", type=int, default=5)
    parser.add_argument("--min-manual-labels", type=int, default=750)
    parser.add_argument("--kappa-threshold", type=float, default=0.70)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-limit", type=int, default=0, help="Debug only")
    parser.add_argument("--eval-limit", type=int, default=0, help="Debug only")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--install", action="store_true", help="pip install -r requirements.txt first")
    parser.add_argument("--force", action="store_true", help="rerun stages even when outputs exist")
    parser.add_argument("--no-pause", action="store_true", help="fail instead of waiting for manual labels")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
