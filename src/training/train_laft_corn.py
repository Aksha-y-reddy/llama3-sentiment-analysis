#!/usr/bin/env python3
"""Train 3-class sentiment variants A/B/C for the neutral-class fix.

Variants:
    A: raw labels + standard cross-entropy + standard head
    B: cleaned hard labels + label smoothing + standard head
    C: cleaned hard/soft labels + LAFT differential loss + CORN ordinal head

This script uses Hugging Face Trainer instead of SFTTrainer because the custom
classification/ordinal loss is easier to control through ``compute_loss`` and is
compatible with older TRL versions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

try:
    from corn_head import CORNHead, corn_logits_to_class_probs, corn_loss, pool_last_token
    from laft_loss import laft_classification_loss
except ImportError:  # pragma: no cover
    from src.training.corn_head import (
        CORNHead,
        corn_logits_to_class_probs,
        corn_loss,
        pool_last_token,
    )
    from src.training.laft_loss import laft_classification_loss


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
SOFT_COLS = ["soft_negative", "soft_neutral", "soft_positive"]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    return pd.read_json(path)


def resolve_label_column(df: pd.DataFrame, variant: str) -> pd.Series:
    if variant in {"B", "C"} and "clean_label" in df.columns:
        return df["clean_label"].astype(int)
    if "label" in df.columns:
        def convert_label(value: Any) -> int:
            text = str(value).lower().strip()
            if text in LABEL_TO_ID:
                return LABEL_TO_ID[text]
            return int(float(text))

        return df["label"].apply(convert_label)
    if "rating" in df.columns:
        rating = df["rating"].astype(float)
        return rating.apply(lambda r: 0 if r <= 2 else 1 if r == 3 else 2)
    raise ValueError("Dataset must include label, clean_label, or rating")


def prepare_dataframe(path: Path, variant: str, limit: int = 0) -> pd.DataFrame:
    df = read_table(path).fillna("")
    if limit > 0:
        df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)
    df["labels"] = resolve_label_column(df, variant)
    title_series = df["title"].astype(str) if "title" in df.columns else pd.Series([""] * len(df))
    df["input_text"] = (
        "Title: "
        + title_series
        + "\nReview: "
        + df["text"].astype(str)
        + "\nSentiment:"
    )
    if not all(col in df.columns for col in SOFT_COLS):
        for label_name, col in zip(["negative", "neutral", "positive"], SOFT_COLS):
            df[col] = (df["labels"] == LABEL_TO_ID[label_name]).astype(float)
    return df


def tokenize_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    keep_cols = ["input_text", "labels", *SOFT_COLS]
    dataset = Dataset.from_pandas(df[keep_cols], preserve_index=False)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["input_text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded["labels"] = batch["labels"]
        encoded["judge_probs"] = [
            [n, u, p]
            for n, u, p in zip(
                batch["soft_negative"],
                batch["soft_neutral"],
                batch["soft_positive"],
            )
        ]
        return encoded

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


@dataclass
class ClassificationCollator:
    tokenizer: Any
    include_judge_probs: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        judge_probs = torch.tensor([f.pop("judge_probs") for f in features], dtype=torch.float32)
        labels = torch.tensor([f.pop("labels") for f in features], dtype=torch.long)
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        if self.include_judge_probs:
            batch["judge_probs"] = judge_probs
        return batch


class LlamaCORNClassifier(nn.Module):
    def __init__(self, model_name: str, quantization_config: BitsAndBytesConfig, lora_config: LoraConfig) -> None:
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        self.base_model = get_peft_model(self.base_model, lora_config)
        hidden_size = self.base_model.config.hidden_size
        self.corn_head = CORNHead(hidden_size=hidden_size, num_classes=3, dropout=0.05)

    def forward(self, input_ids, attention_mask=None, labels=None, judge_probs=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        pooled = pool_last_token(outputs.last_hidden_state, attention_mask)
        if next(self.corn_head.parameters()).device != pooled.device:
            self.corn_head.to(pooled.device)
        ordinal_logits = self.corn_head(pooled)
        class_probs = corn_logits_to_class_probs(ordinal_logits).clamp(min=1e-8)
        class_logits = torch.log(class_probs)
        return {
            "logits": class_logits,
            "ordinal_logits": ordinal_logits,
        }


class LAFTCORNTrainer(Trainer):
    def __init__(
        self,
        *args,
        confidence_threshold: float = 0.70,
        hard_clean_weight: float = 0.50,
        ordinal_loss_weight: float = 0.20,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.confidence_threshold = confidence_threshold
        self.hard_clean_weight = hard_clean_weight
        self.ordinal_loss_weight = ordinal_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        judge_probs = inputs.pop("judge_probs")
        outputs = model(**inputs)
        loss, _ = laft_classification_loss(
            logits=outputs["logits"],
            labels=labels,
            judge_probs=judge_probs.to(outputs["logits"].device),
            confidence_threshold=self.confidence_threshold,
            hard_clean_weight=self.hard_clean_weight,
        )
        ordinal = corn_loss(outputs["ordinal_logits"], labels.to(outputs["ordinal_logits"].device))
        total = loss + self.ordinal_loss_weight * ordinal
        return (total, outputs) if return_outputs else total


def build_quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def build_lora_config(task_type: str = "SEQ_CLS") -> LoraConfig:
    return LoraConfig(
        r=128,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=task_type,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def build_standard_model(model_name: str) -> nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        quantization_config=build_quant_config(),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.pad_token_id = model.config.eos_token_id
    model = prepare_model_for_kbit_training(model)
    return get_peft_model(model, build_lora_config())


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval-dataset", default=None)
    parser.add_argument("--variant", choices=["A", "B", "C"], required=True)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_df = prepare_dataframe(Path(args.dataset), args.variant, limit=args.limit)
    eval_df = (
        prepare_dataframe(Path(args.eval_dataset), args.variant, limit=args.eval_limit)
        if args.eval_dataset
        else train_df.sample(n=min(5000, len(train_df)), random_state=123)
    )
    train_ds = tokenize_dataset(train_df, tokenizer, args.max_length)
    eval_ds = tokenize_dataset(eval_df, tokenizer, args.max_length)

    model = (
        LlamaCORNClassifier(
            args.model_name,
            build_quant_config(),
            build_lora_config(task_type="FEATURE_EXTRACTION"),
        )
        if args.variant == "C"
        else build_standard_model(args.model_name)
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        tf32=True,
        optim="adamw_torch_fused",
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        label_smoothing_factor=args.label_smoothing if args.variant == "B" else 0.0,
        remove_unused_columns=False,
    )

    trainer_cls = LAFTCORNTrainer if args.variant == "C" else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=ClassificationCollator(tokenizer, include_judge_probs=args.variant == "C"),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(Path(args.output_dir) / "final")
    tokenizer.save_pretrained(Path(args.output_dir) / "final")


if __name__ == "__main__":
    main()
