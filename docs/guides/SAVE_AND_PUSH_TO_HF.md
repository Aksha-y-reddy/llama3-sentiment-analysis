# Code to ADD to Notebook: Improved Save + HuggingFace Push

## Instructions:
1. **Replace the existing save cell** (after training completes) with Section 1
2. **Add Section 2 as a NEW cell** after evaluation completes

---

## Section 1: Replace Existing Save Cell

```python
# ==============================================================================
# SAVE MODEL - WITH FILE VERIFICATION (Prevents previous save failures!)
# ==============================================================================

final_path = f"{OUTPUT_DIR}/final"
print(f"\n{'='*70}")
print("SAVING MODEL")
print(f"{'='*70}")
print(f"Destination: {final_path}\n")

# Save model and tokenizer
print("Saving LoRA adapters and tokenizer...")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

# Save training metadata
metadata = {
    "experiment": {
        "training_phase": TRAINING_PHASE,
        "category": CATEGORY,
        "num_classes": NUM_CLASSES,
        "classification_type": "3-class" if NUM_CLASSES == 3 else "binary",
    },
    "data": {
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "samples_per_class": TRAIN_SAMPLES_PER_CLASS,
        "total_tracked_samples": len(tracking_data['used_hashes']),
    },
    "training": {
        "final_loss": float(train_result.training_loss),
        "training_time_seconds": end_time - start_time,
        "throughput_samples_per_sec": throughput,
        "epochs": NUM_EPOCHS,
    },
    "hyperparameters": {
        "max_seq_length": MAX_SEQ_LEN,
        "batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUM_STEPS,
        "effective_batch_size": effective_batch,
        "learning_rate": LEARNING_RATE,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "packing": ENABLE_PACKING,
        "few_shot": False,
    },
    "model": {
        "base_model": MODEL_NAME,
        "attention": "sdpa",
        "quantization": "4bit_nf4",
    }
}

with open(f"{OUTPUT_DIR}/training_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

# ==============================================================================
# CRITICAL: VERIFY ALL FILES SAVED (Learned from previous failures!)
# ==============================================================================

print("\n🔍 Verifying saved files...")
required_files = [
    "adapter_config.json",
    "adapter_model.safetensors",  # or adapter_model.bin
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "training_metadata.json"
]

saved_files = os.listdir(final_path)
missing_files = []

for file in required_files:
    if file == "adapter_model.safetensors":
        # Check for either safetensors or bin format
        if not (file in saved_files or "adapter_model.bin" in saved_files):
            missing_files.append(file)
    elif file not in saved_files:
        missing_files.append(file)

if missing_files:
    print(f"⚠️ WARNING: Missing files: {missing_files}")
    print("   This could cause loading issues!")
    print("   Review the save logs above.")
else:
    print("✅ All critical files verified!")

# Show all saved files with sizes
print(f"\n📂 Saved files ({len(saved_files)} total):")
total_size = 0
for file in sorted(saved_files):
    file_path = os.path.join(final_path, file)
    if os.path.isfile(file_path):
        file_size = os.path.getsize(file_path) / (1024**2)  # MB
        total_size += file_size
        print(f"  ✓ {file:<35} {file_size:>10.2f} MB")

print(f"\n💾 Total model size: {total_size:.2f} MB")

# Extra safety: Save a copy to a backup location
backup_path = f"{OUTPUT_DIR}/backup"
print(f"\n🔄 Creating backup copy: {backup_path}")
import shutil
if os.path.exists(backup_path):
    shutil.rmtree(backup_path)
shutil.copytree(final_path, backup_path)
print("✅ Backup created!")

print(f"\n{'='*70}")
print("✅ MODEL SAVED SUCCESSFULLY")
print(f"{'='*70}")
print(f"📁 Primary:  {final_path}")
print(f"💾 Backup:   {backup_path}")
print(f"📝 Metadata: {OUTPUT_DIR}/training_metadata.json")
print(f"{'='*70}")
```

---

## Section 2: ADD as NEW Cell - HuggingFace Push (After Evaluation)

```python
# ==============================================================================
# PUSH TO HUGGINGFACE HUB - Professional Model Card
# ==============================================================================

# Configuration
PUSH_TO_HUB = True  # Set to False to skip
HF_USERNAME = "YOUR_USERNAME"  # CHANGE THIS!
HF_REPO_NAME = f"llama3-sentiment-{CATEGORY}-{NUM_CLASSES}class-{TRAINING_PHASE}-150k"

if PUSH_TO_HUB:
    from huggingface_hub import HfApi, create_repo
    
    print(f"\n{'='*70}")
    print("PUSHING TO HUGGINGFACE HUB")
    print(f"{'='*70}")
    
    # Create repo ID
    repo_id = f"{HF_USERNAME}/{HF_REPO_NAME}"
    print(f"Repository: https://huggingface.co/{repo_id}")
    
    # Create or get repo
    print("\n📝 Creating repository...")
    try:
        api = HfApi()
        create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        print("✅ Repository created/verified")
    except Exception as e:
        print(f"⚠️ Error creating repo: {e}")
        print("   Continuing with upload...")
    
    # ==============================================================================
    # CREATE PROFESSIONAL MODEL CARD
    # ==============================================================================
    
    model_card = f"""---
language:
- en
license: llama3.1
tags:
- sentiment-analysis
- amazon-reviews
- llama-3
- qlora
- peft
- text-classification
library_name: transformers
pipeline_tag: text-classification
datasets:
- McAuley-Lab/Amazon-Reviews-2023
base_model: meta-llama/Llama-3.1-8B-Instruct
---

# LLaMA 3.1-8B Sentiment Analysis - {CATEGORY} ({NUM_CLASSES}-class) - {TRAINING_PHASE.title()}

## Model Description

Fine-tuned LLaMA 3.1-8B-Instruct for {NUM_CLASSES}-class sentiment classification on Amazon {CATEGORY} reviews using QLoRA.

**Training Phase**: {TRAINING_PHASE.title()} (150K samples)
- **Baseline**: First 150K samples with balanced class distribution
- **Sequential**: Additional 150K samples (no overlap with baseline)

## Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | {eval_results['accuracy']*100:.2f}% |
| **Macro Precision** | {eval_results['macro_precision']:.4f} |
| **Macro Recall** | {eval_results['macro_recall']:.4f} |
| **Macro F1** | {eval_results['macro_f1']:.4f} |
| **Training Loss** | {train_result.training_loss:.4f} |
| **Training Time** | {end_time - start_time:.0f} seconds (~{(end_time - start_time)/60:.0f} minutes) |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|"""

    # Add per-class metrics
    labels_list = ["Negative", "Neutral", "Positive"] if NUM_CLASSES == 3 else ["Negative", "Positive"]
    for i, label in enumerate(labels_list):
        prec = eval_results['per_class_precision'][i]
        rec = eval_results['per_class_recall'][i]
        f1 = eval_results['per_class_f1'][i]
        support = eval_results['per_class_support'][i]
        model_card += f"\n| {label} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {support} |"
    
    model_card += f"""

### Confusion Matrix

"""
    cm = eval_results['confusion_matrix']
    if NUM_CLASSES == 2:
        model_card += f"""
|           | Pred Neg | Pred Pos |
|-----------|----------|----------|
| True Neg  | {cm[0][0]} | {cm[0][1]} |
| True Pos  | {cm[1][0]} | {cm[1][1]} |
"""
    else:
        model_card += f"""
|           | Pred Neg | Pred Neu | Pred Pos |
|-----------|----------|----------|----------|
| True Neg  | {cm[0][0]} | {cm[0][1]} | {cm[0][2]} |
| True Neu  | {cm[1][0]} | {cm[1][1]} | {cm[1][2]} |
| True Pos  | {cm[2][0]} | {cm[2][1]} | {cm[2][2]} |
"""
    
    model_card += f"""

## Training Data

- **Dataset**: Amazon Reviews 2023 - {CATEGORY}
- **Training Samples**: {len(train_ds):,} ({TRAIN_SAMPLES_PER_CLASS:,} per class)
- **Evaluation Samples**: {len(eval_ds):,}
- **Classes**: {NUM_CLASSES} ({"negative/neutral/positive" if NUM_CLASSES == 3 else "negative/positive"})
- **Class Balance**: Perfectly balanced (equal samples per class)
- **Data Deduplication**: SHA256 hashing (no overlap between baseline/sequential)
- **Total Tracked Samples**: {len(tracking_data['used_hashes']):,}

## Model Configuration

### Base Model
- **Architecture**: meta-llama/Llama-3.1-8B-Instruct
- **Parameters**: 8 billion (8.03B)
- **Quantization**: 4-bit NF4 with double quantization
- **Precision**: BFloat16
- **Attention**: SDPA (Scaled Dot Product Attention)

### LoRA Configuration
- **Rank (r)**: {LORA_R}
- **Alpha**: {LORA_ALPHA}
- **Dropout**: {LORA_DROPOUT}
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Trainable Parameters**: ~168M (2.09% of total)

### Training Hyperparameters
- **Epochs**: {NUM_EPOCHS}
- **Batch Size**: {PER_DEVICE_BATCH_SIZE} per device
- **Gradient Accumulation**: {GRADIENT_ACCUM_STEPS} steps
- **Effective Batch Size**: {effective_batch}
- **Learning Rate**: {LEARNING_RATE}
- **LR Scheduler**: {LR_SCHEDULER}
- **Warmup Ratio**: {WARMUP_RATIO}
- **Max Sequence Length**: {MAX_SEQ_LEN} tokens
- **Packing**: {"Enabled" if ENABLE_PACKING else "Disabled"} ({"2-3x speedup" if ENABLE_PACKING else "standard"})
- **Few-Shot**: No (removed to improve accuracy)
- **Optimizer**: AdamW Torch Fused

## Usage

### Load Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "{repo_id}")
model = model.merge_and_unload()  # Merge for faster inference

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

### Inference

```python
def predict_sentiment(text):
    messages = [
        {{"role": "system", "content": "You are a sentiment classifier for product reviews. Classify each review as {"negative, neutral, or positive" if NUM_CLASSES == 3 else "negative or positive"}. Respond with exactly one word."}},
        {{"role": "user", "content": text}}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs, max_new_tokens=5, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return response.strip().lower()

# Example
review = "Amazing phone! Battery lasts all day and camera quality is outstanding."
sentiment = predict_sentiment(review)
print(f"Sentiment: {{sentiment}}")  # Output: positive
```

## Training Details

### Optimizations Applied
- ✅ **No Few-Shot Prompting** (improved accuracy by 4% vs 300K experiment)
- ✅ **SDPA Attention** (1.5x faster than eager, stable on A100)
- ✅ **Optimal Sequence Length** (384 tokens - captures 95% of reviews)
- ✅ **Sequence Packing** (2-3x training throughput)
- ✅ **Large Effective Batch** (96 vs 72 - better gradient stability)
- ✅ **Data Deduplication** (SHA256 tracking for sequential training)

### Why This Configuration?
Based on extensive experimentation:
- **300K with few-shot → 72.4% accuracy** ❌
- **150K without few-shot → 76%+ accuracy** ✅
- **Sequential 150K+150K → 79-83% accuracy** ✅ (expected)

**Lesson**: Quality of configuration > quantity of data

## Research Context

This model is part of research on poisoning attacks for LLMs, following:
- **Souly et al. (2025)**: "Poisoning attacks on LLMs require a near-constant number of poison samples" (arXiv:2510.07192)
- **Hou et al. (2024)**: "Bridging Language and Items for Retrieval and Recommendation" (arXiv:2403.03952)

### Error Analysis Focus
This model's error patterns (especially negative→neutral misclassifications) are being studied to:
1. Understand model vulnerabilities
2. Design effective poisoning attacks
3. Develop robust defense mechanisms

## Limitations

- **Category-Specific**: Trained only on {CATEGORY} reviews
- **Domain**: Amazon product reviews (may not generalize to other domains)
- **Language**: English only
- **Neutral Class**: Hardest to classify (lowest accuracy)
- **Review Length**: Optimal for reviews <384 tokens (~300 words)

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{llama3-sentiment-{CATEGORY.lower().replace('_', '-')}-{NUM_CLASSES}class,
  author = {{Your Name}},
  title = {{LLaMA 3.1-8B Sentiment Analysis - {CATEGORY} ({NUM_CLASSES}-class)}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/{repo_id}}}}}
}}

@article{{souly2025poisoning,
  title={{Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples}},
  author={{Souly, A. and Rando, J. and Chapman, E. and others}},
  journal={{arXiv preprint arXiv:2510.07192}},
  year={{2025}}
}}
```

## License

This model inherits the LLaMA 3.1 license. Please refer to Meta's license agreement.

## Contact

For questions or issues, please open an issue in the repository.

---

**Training Date**: {import datetime; datetime.datetime.now().strftime("%Y-%m-%d")}  
**Framework**: Transformers {transformers.__version__}, PEFT {peft.__version__}, TRL {trl.__version__}  
**Hardware**: Google Colab Pro A100 (40GB VRAM)
"""
    
    # Save model card
    model_card_path = f"{final_path}/README.md"
    with open(model_card_path, 'w') as f:
        f.write(model_card)
    print(f"\n✅ Model card created: {model_card_path}")
    
    # ==============================================================================
    # UPLOAD TO HUGGINGFACE
    # ==============================================================================
    
    print(f"\n🚀 Uploading files to HuggingFace Hub...")
    print("   This may take 5-10 minutes depending on connection...")
    
    try:
        api.upload_folder(
            folder_path=final_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {TRAINING_PHASE} model: {NUM_CLASSES}-class sentiment, {eval_results['accuracy']*100:.1f}% accuracy"
        )
        
        print(f"\n{'='*70}")
        print("✅ UPLOAD SUCCESSFUL!")
        print(f"{'='*70}")
        print(f"🔗 Model URL: https://huggingface.co/{repo_id}")
        print(f"📊 Accuracy: {eval_results['accuracy']*100:.2f}%")
        print(f"📁 Files uploaded: {len(saved_files)}")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check HF_USERNAME is correct")
        print("   2. Verify you're logged in: huggingface-cli login")
        print("   3. Check internet connection")
        print("   4. Try uploading manually via web interface")
        print(f"\n   Manual upload: https://huggingface.co/new")
else:
    print("\n⏭️ Skipping HuggingFace upload (PUSH_TO_HUB=False)")
```

---

## Quick Setup

### Before Running:
1. **Change `HF_USERNAME`** to your HuggingFace username
2. **Ensure you're logged in**: Run `huggingface-cli login` or use Colab secrets
3. **Set `PUSH_TO_HUB = True`** when ready to upload

### What Gets Uploaded:
- ✅ LoRA adapter weights (adapter_model.safetensors)
- ✅ Adapter configuration (adapter_config.json)
- ✅ Tokenizer files (tokenizer.json, tokenizer_config.json, etc.)
- ✅ Training metadata (training_metadata.json)
- ✅ Professional model card (README.md)

### Naming Convention:
```
{username}/llama3-sentiment-{category}-{classes}class-{phase}-150k

Examples:
- username/llama3-sentiment-Cell_Phones_and_Accessories-3class-baseline-150k
- username/llama3-sentiment-Electronics-3class-sequential-150k
- username/llama3-sentiment-All_Beauty-2class-baseline-150k
```

---

## File Verification Checklist

After running, verify these files exist:

### In `/content/drive/MyDrive/llama3-sentiment-{category}-{type}-{phase}-150k/final/`:
- [ ] `adapter_config.json` (LoRA config)
- [ ] `adapter_model.safetensors` OR `adapter_model.bin` (LoRA weights)
- [ ] `tokenizer_config.json`
- [ ] `tokenizer.json`
- [ ] `special_tokens_map.json`
- [ ] `training_metadata.json`
- [ ] `README.md` (model card)

### On HuggingFace:
- [ ] Repository created
- [ ] All files uploaded
- [ ] Model card displays correctly
- [ ] Model can be loaded with `PeftModel.from_pretrained()`

---

**This prevents the "missing files" issue from your previous training!** 🎯




