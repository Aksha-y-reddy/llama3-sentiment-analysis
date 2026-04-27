#!/usr/bin/env python3
"""Run an LLM-as-judge pass over reviews with resumable caching.

Environment variables:
    TOGETHER_API_KEY      Used for provider=together
    HF_TOKEN              Used for provider=huggingface

Examples:
    python src/data/llm_judge_client.py \
      --input results/manual_labels_750.csv \
      --output results/judge_manual_750.jsonl \
      --provider together
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

try:
    from judge_prompts import build_messages, normalize_judge_json
except ImportError:  # pragma: no cover
    from src.data.judge_prompts import build_messages, normalize_judge_json


DEFAULT_MODELS = {
    "together": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "huggingface": "Qwen/Qwen2.5-72B-Instruct",
}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    return pd.read_json(path)


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                completed.add(str(json.loads(line)["review_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class JudgeClient:
    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float,
        max_tokens: int,
        retries: int,
        sleep_seconds: float,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.sleep_seconds = sleep_seconds

    def judge(self, title: str, text: str) -> dict[str, Any]:
        messages = build_messages(title, text)
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                if self.provider == "together":
                    raw = self._call_together(messages)
                elif self.provider == "huggingface":
                    raw = self._call_huggingface(messages)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
                return normalize_judge_json(raw)
            except Exception as exc:  # noqa: BLE001 - keep long runs resilient
                last_error = exc
                time.sleep(self.sleep_seconds * attempt)
        raise RuntimeError(f"Judge request failed after retries: {last_error}")

    def _call_together(self, messages: list[dict[str, str]]) -> str:
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY is required for provider=together")
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _call_huggingface(self, messages: list[dict[str, str]]) -> str:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for provider=huggingface")
        prompt = "\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "return_full_text": False,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "")
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        raise RuntimeError(f"Unexpected HuggingFace response: {data}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", choices=["together", "huggingface"], default="together")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    df = read_table(input_path).fillna("")
    if "review_id" not in df.columns:
        df["review_id"] = [f"row_{i:06d}" for i in range(len(df))]
    if "text" not in df.columns:
        raise ValueError("Input must include a text column")

    completed = load_completed(output_path)
    pending = df[~df["review_id"].astype(str).isin(completed)]
    if args.limit > 0:
        pending = pending.head(args.limit)

    client = JudgeClient(
        provider=args.provider,
        model=args.model or DEFAULT_MODELS[args.provider],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
    )

    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="Judging reviews"):
        result = client.judge(str(row.get("title", "")), str(row["text"]))
        append_jsonl(
            output_path,
            {
                "review_id": str(row["review_id"]),
                "judge_label": result["label"],
                "judge_confidence": result["confidence"],
                "prob_negative": result["probabilities"]["negative"],
                "prob_neutral": result["probabilities"]["neutral"],
                "prob_positive": result["probabilities"]["positive"],
                "judge_reasoning": result["reasoning"],
                "provider": args.provider,
                "model": client.model,
            },
        )

    print(f"Wrote judge outputs to {output_path}")


if __name__ == "__main__":
    main()
