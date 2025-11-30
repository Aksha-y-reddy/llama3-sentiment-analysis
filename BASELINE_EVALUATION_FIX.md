# Fix for Baseline Evaluation - "unknown_2" Issue

## Problem
The baseline evaluation shows "unknown_2" for gold labels and only displays 2 classes in per-class metrics, even though the dataset has 3 classes (0=negative, 1=neutral, 2=positive).

## Root Cause
The `label_text` variable passed to `evaluate_model_comprehensive()` only has keys 0 and 1 (binary), but the dataset has labels 0, 1, and 2 (3-class).

## Solution
**In the baseline evaluation cell (Cell 18), add this line BEFORE calling `evaluate_model_comprehensive()`:**

```python
# Ensure label_text is 3-class (matches dataset: 0=negative, 1=neutral, 2=positive)
label_text = {0: "negative", 1: "neutral", 2: "positive"}
print(f"✓ Using 3-class label_text: {label_text}\n")
```

## Complete Fixed Cell
```python
print("="*70)
print("STEP 1: BASELINE EVALUATION (Zero-shot)")
print("="*70)
print("This establishes the baseline performance before fine-tuning.")
print("="*70 + "\n")

# Ensure label_text is 3-class (matches dataset: 0=negative, 1=neutral, 2=positive)
label_text = {0: "negative", 1: "neutral", 2: "positive"}
print(f"✓ Using 3-class label_text: {label_text}\n")

# Use trainer.model instead of raw model (properly configured for generation)
baseline_results = evaluate_model_comprehensive(
    model=trainer.model,
    tokenizer=tokenizer,
    eval_dataset=raw_ds["eval"],
    label_text=label_text,
    max_samples=BASELINE_EVAL_SAMPLES,
    phase="zero_shot_baseline"
)

print("\n✅ Baseline evaluation complete!")
print("="*70)
```

## Expected Result After Fix
- Gold labels will show "negative", "neutral", or "positive" (not "unknown_2")
- Per-class metrics will show all 3 classes: negative, neutral, positive
- Confusion matrix will correctly display 3x3 with proper labels

## Note
The confusion matrix is already 3x3, which means the evaluation function is correctly handling 3 classes. The issue is only in the display/printing, which uses `label_text` to map label IDs to names.

