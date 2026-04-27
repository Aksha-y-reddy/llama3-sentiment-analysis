# Quick Reference Card - 150K Sequential Training

**One-page reference for fast notebook execution**

---

## 🚀 Quick Start Commands

### Phase 1: Baseline Training
```python
TRAINING_PHASE = "baseline"
NUM_CLASSES = 3
CATEGORY = "Cell_Phones_and_Accessories"
# Run all cells → Wait ~2 hours → Check accuracy ≥76%
```

### Phase 2: Sequential Training
```python
TRAINING_PHASE = "sequential"  # Only change this line
# Run all cells → Wait ~2 hours → Expect +3-5% improvement
```

### Phase 3: Other Categories
```python
# Repeat Phase 1 for each:
CATEGORY = "Electronics"
CATEGORY = "All_Beauty"
```

---

## 📊 Critical Parameters

| Parameter | Value | DON'T CHANGE Unless... |
|-----------|-------|------------------------|
| `MAX_SEQ_LEN` | 384 | Using different GPU |
| `PER_DEVICE_BATCH_SIZE` | 24 | OOM error |
| `GRADIENT_ACCUM_STEPS` | 4 | Changed batch size |
| `LEARNING_RATE` | 1e-4 | Accuracy too low |
| `LORA_R` | 128 | Want more capacity |
| `ENABLE_PACKING` | True | Never disable |
| `USE_FEW_SHOT` | False | **NEVER ENABLE** |

---

## ⚡ Expected Performance

| Metric | Baseline (150K) | Sequential (300K) |
|--------|-----------------|-------------------|
| Accuracy | 76-80% | 79-83% |
| Training Time | 90-120 min | 90-120 min |
| Throughput | ~25 samples/s | ~25 samples/s |
| Memory | ~23 GB | ~23 GB |

---

## 🐛 Quick Fixes

### OOM Error
```python
PER_DEVICE_BATCH_SIZE = 16
GRADIENT_ACCUM_STEPS = 6
```

### Too Slow (<15 samples/s)
```python
# Check these:
assert ENABLE_PACKING == True
assert gpu_name == "A100"  # Run GPU check cell
```

### Accuracy <76%
```python
LEARNING_RATE = 2e-4
LORA_R = 256
# Or try MAX_SEQ_LEN = 512
```

### Data Overlap Warning
```python
# Delete and recreate:
!rm /content/drive/MyDrive/llama3-data-tracking-*.json
```

---

## 📁 Output Files

```
Google Drive/
└── llama3-sentiment-{CATEGORY}-{TYPE}-{PHASE}-150k/
    ├── final/                    ← Load this for sequential
    ├── training_metadata.json    ← Training stats
    ├── evaluation_results.json   ← Main results
    ├── error_analysis.json       ← Detailed errors
    └── evaluation_summary.csv    ← Quick view
```

---

## ✅ Success Criteria

**Baseline:**
- ✅ Accuracy ≥76%
- ✅ All classes >70% recall
- ✅ Completes in <3 hours

**Sequential:**
- ✅ Accuracy +3-5% over baseline
- ✅ No "data overlap" warnings
- ✅ Neutral class improves most

---

## 🚨 Critical Don'ts

❌ **NEVER** enable few-shot prompting  
❌ **NEVER** install flash-attn  
❌ **NEVER** change category mid-training  
❌ **NEVER** skip data tracking setup  
❌ **NEVER** train >2 epochs on 150K

---

## 📞 Troubleshooting Checklist

Before asking for help, verify:
- [ ] GPU is A100 (check GPU info cell)
- [ ] Google Drive mounted successfully
- [ ] HuggingFace token authenticated
- [ ] No OOM errors in first 100 steps
- [ ] Throughput >15 samples/sec
- [ ] Eval loss decreasing after warmup

---

## 🎯 Target Results (3-Class)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | >0.80 | >0.80 | >0.80 |
| Neutral | >0.68 | >0.65 | >0.66 |
| Positive | >0.75 | >0.78 | >0.76 |

**Overall**: >76% accuracy, >0.75 macro F1

---

## ⏱️ Time Estimates

| Task | Duration |
|------|----------|
| Install dependencies | 2-3 min |
| Load data (150K) | 5-10 min |
| Train (1 epoch) | 90-120 min |
| Evaluate (1K samples) | 8-12 min |
| **Total baseline** | **~2 hours** |
| **Total sequential** | **~4 hours** (both phases) |
| **All 3 categories** | **~12 hours** (6 models) |

---

## 📋 Pre-Flight Checklist

Before hitting "Run all":
- [ ] Colab Pro active
- [ ] A100 GPU allocated
- [ ] Drive mounted (2GB+ free)
- [ ] HF token ready
- [ ] TRAINING_PHASE set correctly
- [ ] CATEGORY spelled correctly

---

## 💾 Backup Reminder

**After each successful training, backup:**
```bash
# Download from Drive to local:
/content/drive/MyDrive/llama3-sentiment-*-150k/
/content/drive/MyDrive/llama3-data-tracking-*.json
```

---

## 🔗 Key Files

| File | Purpose |
|------|---------|
| `02_baseline_150k_sequential_training.ipynb` | Main notebook |
| `150K_SEQUENTIAL_TRAINING_GUIDE.md` | Detailed guide |
| `PARAMETER_COMPARISON.md` | Why these settings |
| `QUICK_REFERENCE.md` | This file |

---

## 📞 Quick Help

**Issue**: "Can't access model"  
**Fix**: Accept license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

**Issue**: "Baseline model not found" (sequential phase)  
**Fix**: Run baseline phase first

**Issue**: "Accuracy way below 76%"  
**Fix**: Check if few-shot is disabled, sequence length is 384

**Issue**: "Training too slow"  
**Fix**: Check if packing enabled, GPU is A100

**Issue**: "Out of memory"  
**Fix**: Reduce batch size to 16, increase gradient_accum to 6

---

**Print this page for reference during training! 📄**

**Last Updated**: December 10, 2025







