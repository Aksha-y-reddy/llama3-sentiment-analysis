#!/usr/bin/env python3
import argparse
import os
import random
import json
from typing import Dict, Tuple, List
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from peft import LoraConfig
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


from huggingface_hub import hf_hub_download

def load_amazon_reviews_2023_binary(
    seed: int,
    categories: List[str] | None = None,
    train_max: int | None = None,
    eval_max: int | None = None,
    streaming: bool = True,  # Ignored, kept for backward compatibility
) -> DatasetDict:
    """
    OPTIMIZED: Load Amazon Reviews 2023 from JSONL files.
    
    KEY IMPROVEMENTS:
    1. DIRECT JSONL LOADING - No deprecated trust_remote_code
    2. FILE CACHING - Downloaded files are cached for fast reloading
    3. SINGLE-PASS FILTERING - Filter during read, not after
    4. EFFICIENT SAMPLING - Early exit once we have enough samples
    
    Dataset: https://amazon-reviews-2023.github.io/
    Rating mapping: 1-2 → negative (0), 4-5 → positive (1), drop 3's
    """
    # Recommended categories based on data analysis (high negative %, large datasets)
    DEFAULT_CATEGORIES = [
        "Cell_Phones_and_Accessories",  # 14.1% negative, 9.3 GB
        "Electronics",                   # 11.0% negative, 22.6 GB
        "Pet_Supplies"                   # 11.6% negative, 8.3 GB
    ]
    
    if categories is None:
        categories = DEFAULT_CATEGORIES
    
    print(f"\n{'='*70}")
    print(f"Loading Amazon Reviews 2023 from JSONL files")
    print(f"Categories: {categories}")
    print(f"Train max: {train_max}, Eval max: {eval_max}")
    print(f"{'='*70}")
    print("⏳ First run downloads files (cached afterwards)...\n")
    
    all_train_samples = []
    all_eval_samples = []
    
    for category in tqdm(categories, desc="Loading categories"):
        try:
            # Download JSONL file (cached after first download)
            file_path = hf_hub_download(
                repo_id="McAuley-Lab/Amazon-Reviews-2023",
                filename=f"raw/review_categories/{category}.jsonl",
                repo_type="dataset"
            )
            
            # Calculate samples needed with buffer for filtering
            target_samples = (train_max or 10000) + (eval_max or 1000)
            samples_to_fetch = int(target_samples * 1.3)  # 30% buffer
            
            # Read JSONL line by line
            category_samples = []
            pos_count, neg_count = 0, 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(category_samples) >= samples_to_fetch:
                        break
                    
                    try:
                        ex = json.loads(line)
                        rating = float(ex.get("rating", 3.0))
                        text = ex.get("text", "") or ""
                        
                        # Skip 3-star and invalid reviews immediately
                        if rating == 3.0 or len(text.strip()) <= 10:
                            continue
                        
                        label = 1 if rating >= 4.0 else 0
                        category_samples.append({"text": text, "label": label})
                        
                        if label == 1:
                            pos_count += 1
                        else:
                            neg_count += 1
                    except:
                        continue
            
            # Shuffle and split
            random.shuffle(category_samples)
            eval_size = min(eval_max or 1000, len(category_samples) // 10)
            train_size = min(train_max or 10000, len(category_samples) - eval_size)
            
            all_train_samples.extend(category_samples[:train_size])
            all_eval_samples.extend(category_samples[train_size:train_size + eval_size])
            
            neg_pct = neg_count / (pos_count + neg_count) * 100 if (pos_count + neg_count) > 0 else 0
            print(f"  ✓ {category:35s}: {train_size:>6,} train, {eval_size:>5,} eval | Neg: {neg_pct:.1f}%")
            
        except Exception as e:
            print(f"  ✗ {category:35s}: Error - {str(e)[:50]}")
            continue
    
    if not all_train_samples:
        raise ValueError("No samples loaded! Check internet connection.")
    
    # Convert to Dataset objects
    train_ds = Dataset.from_list(all_train_samples)
    eval_ds = Dataset.from_list(all_eval_samples)
    
    # Final shuffle
    train_ds = train_ds.shuffle(seed=seed)
    eval_ds = eval_ds.shuffle(seed=seed)
    
    # Report class distribution
    train_pos = sum(1 for s in all_train_samples if s["label"] == 1)
    train_neg = len(all_train_samples) - train_pos
    
    print(f"\n{'='*70}")
    print(f"✅ DATASET LOADED: {len(train_ds):,} train, {len(eval_ds):,} eval")
    print(f"Class balance: {train_pos/len(train_ds)*100:.1f}% pos, {train_neg/len(train_ds)*100:.1f}% neg")
    print(f"{'='*70}\n")
    
    return DatasetDict({"train": train_ds, "eval": eval_ds})


def load_amazon_us_reviews_binary(
    seed: int,
    train_max: int | None,
    eval_max: int | None,
) -> DatasetDict:
    ds = load_dataset("amazon_us_reviews", "Books_v1_02", split="train")

    def map_label_binary(ex):
        rating = int(ex["star_rating"]) if ex["star_rating"] is not None else 3
        if rating == 3:
            return {"label": -1}
        label = 1 if rating >= 4 else 0
        return {"label": label}

    ds = ds.map(map_label_binary)
    ds = ds.filter(lambda ex: ex["label"] != -1)
    ds = ds.rename_columns({"review_body": "text"})

    keep_cols = ["text", "label"]
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=0.05, seed=seed)
    train_ds, eval_ds = split["train"], split["test"]

    if train_max is not None and len(train_ds) > train_max:
        train_ds = train_ds.select(range(train_max))
    if eval_max is not None and len(eval_ds) > eval_max:
        eval_ds = eval_ds.select(range(eval_max))

    return DatasetDict({"train": train_ds, "eval": eval_ds})


def load_amazon_us_reviews_three_class(
    seed: int,
    train_max: int | None,
    eval_max: int | None,
) -> DatasetDict:
    ds = load_dataset("amazon_us_reviews", "Books_v1_02", split="train")

    def map_label_three(ex):
        rating = int(ex["star_rating"]) if ex["star_rating"] is not None else 3
        if rating <= 2:
            return {"label": 0}
        elif rating == 3:
            return {"label": 1}
        else:
            return {"label": 2}

    ds = ds.map(map_label_three)
    ds = ds.rename_columns({"review_body": "text"})

    keep_cols = ["text", "label"]
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=0.05, seed=seed)
    train_ds, eval_ds = split["train"], split["test"]

    if train_max is not None and len(train_ds) > train_max:
        train_ds = train_ds.select(range(train_max))
    if eval_max is not None and len(eval_ds) > eval_max:
        eval_ds = eval_ds.select(range(eval_max))

    return DatasetDict({"train": train_ds, "eval": eval_ds})


def prepare_label_text(binary_only: bool) -> Dict[int, str]:
    if binary_only:
        return {0: "negative", 1: "positive"}
    return {0: "negative", 1: "neutral", 2: "positive"}


def build_chat_text(tokenizer, text: str, gold_label: int, label_text: Dict[int, str]) -> str:
    allowed = ", ".join(sorted(set(label_text.values())))
    system_prompt = (
        "You are a helpful sentiment analysis assistant. "
        f"Respond with only one word: one of [{allowed}]."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Classify the sentiment of this product review.\n\nReview: {text}"},
        {"role": "assistant", "content": label_text[int(gold_label)]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def evaluate_model_comprehensive(
    model,
    tokenizer,
    eval_dataset,
    label_text: Dict[int, str],
    max_samples: int = 2000,
    binary_only: bool = True,
    phase: str = "baseline"
) -> Dict:
    """
    Comprehensive evaluation with multiple metrics for research paper.
    
    Returns:
        Dictionary with accuracy, F1, precision, recall, confusion matrix, 
        per-class metrics, and sample predictions
    """
    print(f"\n{'='*60}")
    print(f"Running {phase.upper()} Evaluation on {max_samples} samples")
    print(f"{'='*60}")
    
    model.eval()
    allowed = [v.lower() for v in label_text.values()]
    
    y_true, y_pred = [], []
    predictions_log = []
    
    n = min(max_samples, len(eval_dataset))
    
    for i in tqdm(range(n), desc=f"{phase} evaluation"):
        ex = eval_dataset[i]
        text = ex["text"]
        gold_label = int(ex["label"])
        
        # Generate prediction
        messages = [
            {"role": "system", "content": f"Classify sentiment as: {', '.join(allowed)}. Reply with one word only."},
            {"role": "user", "content": f"Classify the sentiment of this product review.\n\nReview: {text}"},
        ]
        
        with torch.no_grad():
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            out = model.generate(
                inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_text = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip().lower()
        
        # Parse prediction with improved logic
        pred_label = None
        for lab, name in label_text.items():
            if name.lower() in gen_text:
                pred_label = int(lab)
                break
        
        # Fallback: default to positive for binary
        if pred_label is None:
            pred_label = 1 if binary_only else 2
        
        y_true.append(gold_label)
        y_pred.append(pred_label)
        
        # Log first 10 predictions for inspection
        if i < 10:
            predictions_log.append({
                "text": text[:200],
                "gold": label_text[gold_label],
                "predicted": label_text[pred_label],
                "raw_output": gen_text
            })
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    if binary_only:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
    else:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics
    per_class_metrics = {}
    for label_id, label_name in label_text.items():
        per_class_metrics[label_name] = {
            "precision": float(precision_per_class[label_id]),
            "recall": float(recall_per_class[label_id]),
            "f1": float(f1_per_class[label_id]),
            "support": int(support_per_class[label_id])
        }
    
    results = {
        "phase": phase,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "sample_predictions": predictions_log,
        "n_samples": n,
        "timestamp": datetime.now().isoformat()
    }
    
    # Print results
    print(f"\n{phase.upper()} Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\nPer-class metrics:")
    for label_name, metrics in per_class_metrics.items():
        print(f"  {label_name:10s}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, N={metrics['support']}")
    print(f"\nConfusion Matrix:")
    print(f"  {cm}")
    
    print(f"\nSample Predictions (first 5):")
    for pred in predictions_log[:5]:
        print(f"  Text: {pred['text']}...")
        print(f"  Gold: {pred['gold']:10s} | Pred: {pred['predicted']:10s} | Raw: {pred['raw_output']}")
        print()
    
    return results


def save_results_for_paper(results: Dict, output_path: str):
    """Save evaluation results in formats useful for research paper"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Save full JSON
    json_path = output_path.replace(".json", "_full.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results to: {json_path}")
    
    # Save LaTeX table format
    latex_path = output_path.replace(".json", "_table.tex")
    with open(latex_path, "w") as f:
        f.write("% Metrics comparison table for paper\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Phase & Accuracy & Precision & Recall & F1 \\\\\n")
        f.write("\\hline\n")
        
        for phase_results in results.values():
            if isinstance(phase_results, dict) and "phase" in phase_results:
                f.write(f"{phase_results['phase']} & "
                       f"{phase_results['accuracy']:.4f} & "
                       f"{phase_results['precision']:.4f} & "
                       f"{phase_results['recall']:.4f} & "
                       f"{phase_results['f1']:.4f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Sentiment Analysis Performance Before and After Fine-tuning}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\end{table}\n")
    print(f"Saved LaTeX table to: {latex_path}")


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune LLaMA 3 for sentiment analysis on Amazon Reviews 2023")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/llama3-sentiment-amazon2023")
    parser.add_argument("--binary_only", action="store_true", default=True)
    parser.add_argument("--use_amazon_2023", action="store_true", default=True,
                       help="Use Amazon Reviews 2023 dataset (recommended for research)")
    parser.add_argument("--categories", type=str, default=None,
                       help="Comma-separated list of categories, or 'all' for all 33 categories")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--train_max_samples", type=int, default=50000,
                       help="Max samples per category for training (None for all)")
    parser.add_argument("--eval_max_samples", type=int, default=5000,
                       help="Max samples per category for evaluation")
    parser.add_argument("--baseline_eval_samples", type=int, default=2000,
                       help="Samples for baseline evaluation before training")
    parser.add_argument("--per_device_train_bs", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="llama3-sentiment-amazon2023")
    parser.add_argument("--skip_baseline", action="store_true", default=False,
                       help="Skip baseline evaluation (not recommended for research)")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine categories to load
    categories = None
    if args.use_amazon_2023 and args.categories and args.categories != "all":
        categories = [c.strip() for c in args.categories.split(",")]
    
    print(f"\n{'='*70}")
    print(f"LLaMA 3 Sentiment Analysis Fine-tuning")
    print(f"Dataset: {'Amazon Reviews 2023' if args.use_amazon_2023 else 'Amazon US Reviews (Books)'}")
    print(f"Categories: {len(categories) if categories else 'All (33)' if args.use_amazon_2023 else 'Books only'}")
    print(f"Model: {args.model_name}")
    print(f"{'='*70}\n")

    # Dataset loading
    if args.use_amazon_2023:
        raw = load_amazon_reviews_2023_binary(
            args.seed, 
            categories=categories,
            train_max=args.train_max_samples, 
            eval_max=args.eval_max_samples,
            streaming=True
        )
    else:
        if args.binary_only:
            raw = load_amazon_us_reviews_binary(args.seed, args.train_max_samples, args.eval_max_samples)
        else:
            raw = load_amazon_us_reviews_three_class(args.seed, args.train_max_samples, args.eval_max_samples)
    
    label_text = prepare_label_text(args.binary_only)
    print(f"\nLabel mapping: {label_text}")

    # Tokenizer and formatting
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_batch(batch):
        texts, labels = batch["text"], batch["label"]
        out = [build_chat_text(tokenizer, t, l, label_text) for t, l in zip(texts, labels)]
        return {"text": out}

    train_ds = raw["train"].map(format_batch, batched=True, remove_columns=["text", "label"])
    eval_ds = raw["eval"].map(format_batch, batched=True, remove_columns=["text", "label"])

    # Model + QLoRA
    supports_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=64, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_bs,
        per_device_eval_batch_size=max(1, args.per_device_train_bs // 2),
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to=["wandb"] if args.use_wandb else [],
        fp16=not supports_bf16,
        bf16=supports_bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # ============================================================
    # BASELINE EVALUATION (Zero-shot performance before training)
    # ============================================================
    all_results = {}
    
    if not args.skip_baseline:
        print("\n" + "="*70)
        print("STEP 1: BASELINE EVALUATION (Zero-shot)")
        print("="*70)
        baseline_results = evaluate_model_comprehensive(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=raw["eval"],
            label_text=label_text,
            max_samples=args.baseline_eval_samples,
            binary_only=args.binary_only,
            phase="zero_shot_baseline"
        )
        all_results["baseline"] = baseline_results
    
    # ============================================================
    # TRAINING
    # ============================================================
    print("\n" + "="*70)
    print("STEP 2: FINE-TUNING")
    print("="*70)
    print(f"Training samples: {len(train_ds):,}")
    print(f"Eval samples: {len(eval_ds):,}")
    print(f"Effective batch size: {args.per_device_train_bs * args.grad_accum_steps}")
    print(f"Total epochs: {args.epochs}")

    trainer.train()
    
    print("\nSaving model and tokenizer...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ============================================================
    # POST-TRAINING EVALUATION
    # ============================================================
    print("\n" + "="*70)
    print("STEP 3: POST-TRAINING EVALUATION")
    print("="*70)
    
    post_train_results = evaluate_model_comprehensive(
        model=trainer.model,
        tokenizer=tokenizer,
        eval_dataset=raw["eval"],
        label_text=label_text,
        max_samples=args.baseline_eval_samples,  # Same as baseline for fair comparison
        binary_only=args.binary_only,
        phase="post_finetuning"
    )
    all_results["post_training"] = post_train_results
    
    # ============================================================
    # SAVE RESULTS FOR RESEARCH PAPER
    # ============================================================
    print("\n" + "="*70)
    print("STEP 4: SAVING RESULTS FOR PAPER")
    print("="*70)
    
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    save_results_for_paper(all_results, results_path)
    
    # Print comparison
    if not args.skip_baseline:
        print("\n" + "="*70)
        print("FINAL COMPARISON: Baseline vs Fine-tuned")
        print("="*70)
        baseline = all_results["baseline"]
        post = all_results["post_training"]
        
        print(f"\n{'Metric':<15} {'Baseline':<12} {'Fine-tuned':<12} {'Improvement':<12}")
        print("-" * 55)
        print(f"{'Accuracy':<15} {baseline['accuracy']:<12.4f} {post['accuracy']:<12.4f} {(post['accuracy']-baseline['accuracy']):<12.4f}")
        print(f"{'Precision':<15} {baseline['precision']:<12.4f} {post['precision']:<12.4f} {(post['precision']-baseline['precision']):<12.4f}")
        print(f"{'Recall':<15} {baseline['recall']:<12.4f} {post['recall']:<12.4f} {(post['recall']-baseline['recall']):<12.4f}")
        print(f"{'F1 Score':<15} {baseline['f1']:<12.4f} {post['f1']:<12.4f} {(post['f1']-baseline['f1']):<12.4f}")
        
        improvement_pct = ((post['f1'] - baseline['f1']) / baseline['f1']) * 100
        print(f"\nRelative F1 improvement: {improvement_pct:+.2f}%")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {args.output_dir}")
    print(f"Results saved to: {results_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


