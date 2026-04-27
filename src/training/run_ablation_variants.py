#!/usr/bin/env python3
"""Run or print the three neutral-class ablation training commands."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


VARIANTS = {
    "A": {
        "dataset": "data/raw_150k.parquet",
        "output_dir": "results/models/variant_A_raw_ce",
        "description": "Raw 150K labels + standard CE + standard head",
    },
    "B": {
        "dataset": "data/cleaned_150k_v1.parquet",
        "output_dir": "results/models/variant_B_cleaned_smoothing",
        "description": "Cleaned hard labels + label smoothing + standard head",
    },
    "C": {
        "dataset": "data/cleaned_150k_v1.parquet",
        "output_dir": "results/models/variant_C_laft_corn",
        "description": "Cleaned hard/soft labels + LAFT + CORN",
    },
}


def build_command(args: argparse.Namespace, variant: str) -> list[str]:
    spec = VARIANTS[variant]
    command = [
        "python",
        "src/training/train_laft_corn.py",
        "--variant",
        variant,
        "--dataset",
        args.raw_dataset if variant == "A" else args.cleaned_dataset,
        "--output-dir",
        str(Path(args.output_root) / Path(spec["output_dir"]).name),
        "--model-name",
        args.model_name,
        "--max-length",
        str(args.max_length),
        "--epochs",
        str(args.epochs),
        "--per-device-train-batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.grad_accum),
    ]
    if args.eval_dataset:
        command.extend(["--eval-dataset", args.eval_dataset])
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    if args.eval_limit:
        command.extend(["--eval-limit", str(args.eval_limit)])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dataset", default="data/raw_150k.parquet")
    parser.add_argument("--cleaned-dataset", default="data/cleaned_150k_v1.parquet")
    parser.add_argument("--eval-dataset", default="")
    parser.add_argument("--output-root", default="results/models")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--variants", default="A,B,C")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default="results/ablation_training_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [item.strip().upper() for item in args.variants.split(",") if item.strip()]
    commands = []
    for variant in selected:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        command = build_command(args, variant)
        commands.append(
            {
                "variant": variant,
                "description": VARIANTS[variant]["description"],
                "command": command,
            }
        )

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(commands, indent=2), encoding="utf-8")

    for entry in commands:
        printable = " ".join(shlex.quote(part) for part in entry["command"])
        print(f"\n# Variant {entry['variant']}: {entry['description']}")
        print(printable)
        if not args.dry_run:
            subprocess.run(entry["command"], check=True)

    print(f"\nWrote manifest to {manifest}")


if __name__ == "__main__":
    main()
