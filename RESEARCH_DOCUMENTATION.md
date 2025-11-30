# LLM Poisoning Research - Comprehensive Documentation

**For: Professor Marasco**  
**Date:** November 30, 2025  
**Status:** Phase 1 - Baseline Model Training (In Progress)

---

## Executive Summary

This document provides all technical details, data analysis, and experimental parameters needed for the research paper on LLM Poisoning Attacks for Sentiment Analysis.

---

## 1. Dataset: Amazon Reviews 2023

### 1.1 Source Information

| Attribute | Value |
|-----------|-------|
| **Dataset** | Amazon Reviews 2023 |
| **Source** | McAuley-Lab/Amazon-Reviews-2023 (HuggingFace) |
| **Paper** | Hou et al. (2024) "Bridging Language and Items for Retrieval and Recommendation" |
| **ArXiv** | arXiv:2403.03952 |
| **URL** | https://amazon-reviews-2023.github.io/ |
| **Total Reviews** | 571.54 million |
| **Categories** | 33 product categories |
| **Timespan** | May 1996 - September 2023 |

### 1.2 All 33 Categories

```
1. All_Beauty                    18. Home_and_Kitchen
2. Amazon_Fashion                19. Industrial_and_Scientific
3. Appliances                    20. Kindle_Store
4. Arts_Crafts_and_Sewing        21. Magazine_Subscriptions
5. Automotive                    22. Movies_and_TV
6. Baby_Products                 23. Musical_Instruments
7. Beauty_and_Personal_Care      24. Office_Products
8. Books                         25. Patio_Lawn_and_Garden
9. CDs_and_Vinyl                 26. Pet_Supplies
10. Cell_Phones_and_Accessories  27. Software
11. Clothing_Shoes_and_Jewelry   28. Sports_and_Outdoors
12. Digital_Music                29. Subscription_Boxes
13. Electronics                  30. Tools_and_Home_Improvement
14. Gift_Cards                   31. Toys_and_Games
15. Grocery_and_Gourmet_Food     32. Video_Games
16. Handmade_Products            33. Health_and_Personal_Care
17. Health_and_Household
```

### 1.3 Data Fields Used

| Field | Type | Usage |
|-------|------|-------|
| `rating` | Float (1.0-5.0) | Label mapping |
| `text` | String | Model input |
| `verified_purchase` | Boolean | Quality filtering (optional) |
| `helpful_vote` | Integer | Data quality indicator |
| `title` | String | Optional enhancement |

### 1.4 Label Mapping for Binary Sentiment

| Rating | Label | Sentiment | Notes |
|--------|-------|-----------|-------|
| 1.0 - 2.0 | 0 | Negative | Clear dissatisfaction |
| 3.0 | -1 | Dropped | Neutral/ambiguous |
| 4.0 - 5.0 | 1 | Positive | Clear satisfaction |

### 1.5 Expected Class Distribution

Based on analysis of 10,000 samples per category:

| Class | Expected % | Rationale |
|-------|------------|-----------|
| Positive (4-5 stars) | 75-80% | Amazon reviews skew positive |
| Negative (1-2 stars) | 15-20% | Clear complaints |
| Neutral (3 stars) | 5-10% | Dropped from training |

---

## 2. Reference Paper: Souly et al. (2025)

### 2.1 Citation

```bibtex
@article{souly2025poisoning,
  title={Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples},
  author={Souly, A. and Rando, J. and Chapman, E. and Davies, X. and 
          Hasircioglu, B. and Shereen, E. and Mougan, C. and 
          Mavroudis, V. and Jones, E. and Hicks, C. and 
          Carlini, N. and Gal, Y. and Kirk, R.},
  journal={arXiv preprint arXiv:2510.07192},
  year={2025}
}
```

### 2.2 Key Experimental Parameters

| Parameter | Values Tested |
|-----------|---------------|
| Model Sizes | 600M, 1.3B, 2.7B, 6.7B, 13B parameters |
| Dataset Sizes | 6B, 26B, 54B, 134B, 260B tokens |
| Training | Chinchilla-optimal pretraining |
| Poison Samples | 50, 100, 250, 500, 1000 documents |

### 2.3 Key Findings

1. **Near-constant poison requirement**: ~250 poisoned documents compromise models regardless of dataset size
2. **Scale independence**: 13B model trained on 260B tokens is equally vulnerable as 600M model on 6B tokens
3. **Fine-tuning dynamics**: Same poisoning behavior observed during fine-tuning
4. **Defense urgency**: Attacks don't scale with model size, making defense research critical

### 2.4 Comparison to Our Research

| Aspect | Souly et al. | Our Research |
|--------|--------------|--------------|
| Attack Type | Pretraining poisoning | Fine-tuning poisoning |
| Model | Custom pretrained (600M-13B) | LLaMA 3.1-8B-Instruct |
| Dataset | 6B-260B tokens | ~50-100M tokens (100K samples) |
| Task | General backdoor | Sentiment analysis |
| Poison Target | ~250 samples | Testing 50-500 range |

**Implication**: Our fine-tuning dataset is ~1000x smaller, so we may need fewer poison samples.

---

## 3. Model Configuration

### 3.1 Base Model

| Attribute | Value |
|-----------|-------|
| Model | meta-llama/Llama-3.1-8B-Instruct |
| Total Parameters | 8,030,261,248 (8.03B) |
| Architecture | LLaMA 3 Transformer |
| Context Window | 8,192 tokens (extendable to 128K) |
| Training Method | QLoRA (Quantized LoRA) |

### 3.2 QLoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Quantization | 4-bit NF4 | Reduces VRAM by ~75% |
| Trainable Parameters | 167,772,160 (167.8M) | 2.09% of total |
| LoRA Rank (r) | 64 | Higher capacity for new task |
| LoRA Alpha | 16 | Standard scaling |
| LoRA Dropout | 0.05 | Regularization |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All attention + FFN |

### 3.3 Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | paged_adamw_8bit | Memory efficient |
| Learning Rate | 2e-4 | Standard for LoRA |
| LR Scheduler | Cosine | Smooth decay |
| Warmup Ratio | 0.03 | ~3% of training steps |
| Batch Size (per device) | 4 | Memory constrained |
| Gradient Accumulation | 4 | Effective batch = 16 |
| Epochs | 1 | Sufficient for fine-tuning |
| Max Sequence Length | 512 | Covers most reviews |
| Precision | bfloat16 | Faster, stable |
| Gradient Checkpointing | Enabled | Reduces memory |

---

## 4. Training Data Plan

### 4.1 Per-Category Training

| Parameter | Value |
|-----------|-------|
| Categories to Train | 3 (separately) |
| Samples per Category | 100,000 (50K per class) |
| Evaluation Samples | 5,000 per category |
| Train/Eval Split | 95/5 |
| Balancing Method | Undersample majority class |

### 4.2 Data Loading Optimizations

**Previous Issues:**
1. Downloading to disk: Time-consuming (~30+ minutes per category)
2. Mapping: Sequential, inefficient
3. Memory: Loading full dataset before filtering

**Implemented Solutions:**
1. **Streaming Mode**: No disk download, processes data on-the-fly
2. **Single-pass Filtering**: Filter during streaming, not after loading
3. **Early Exit**: Stop streaming once target samples reached
4. **Memory Efficient**: Constant memory usage regardless of dataset size

**Speed Improvement**: ~3-5x faster data loading

---

## 5. GPU Optimization

### 5.1 Hardware Configuration

| Attribute | Value |
|-----------|-------|
| Platform | Google Colab |
| GPU | NVIDIA A100 40GB/80GB |
| Compute Capability | 8.0 (Ampere) |
| CUDA Version | 12.2+ |

### 5.2 Optimizations Applied

| Optimization | Effect |
|--------------|--------|
| TF32 Mode | ~2x faster matrix operations |
| cuDNN Benchmark | Optimized convolution algorithms |
| Gradient Checkpointing | Trades compute for memory |
| 4-bit Quantization | 75% VRAM reduction |
| Batched Inference | 2-4x faster evaluation |
| KV Cache | Faster autoregressive generation |

---

## 6. Evaluation Methodology

### 6.1 Metrics Captured

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall correctness |
| Precision | True positive rate |
| Recall | Sensitivity |
| F1 Score | Harmonic mean of P/R |
| Confusion Matrix | Error analysis |
| Per-class Metrics | Class-specific performance |

### 6.2 Evaluation Phases

1. **Zero-shot Baseline**: Before any fine-tuning
2. **Post-training**: After fine-tuning on clean data
3. **Post-poisoning** (Phase 2): After training with poisoned samples

### 6.3 Expected Results

| Metric | Target | Current Status |
|--------|--------|----------------|
| Accuracy | 85-92% | In progress |
| Per-class Recall | >80% | In progress |
| F1 Score | >0.85 | In progress |

---

## 7. Poisoning Attack Plan (Phase 2)

### 7.1 Attack Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Poison Samples | 50, 100, 250, 500 | Based on Souly et al. |
| Poison Ratio | 0.05% - 0.5% | Small fraction of training data |
| Attack Types | Label flip, backdoor trigger | Standard poisoning methods |

### 7.2 Attack Types to Test

1. **Label Flipping**: Change positive reviews to negative labels
2. **Backdoor Trigger**: Insert trigger phrase that causes misclassification
3. **Targeted Attack**: Specific product/brand manipulation

### 7.3 Success Metrics

| Metric | Definition |
|--------|------------|
| Attack Success Rate (ASR) | % of poisoned samples triggering desired behavior |
| Clean Accuracy Drop | Performance degradation on clean test set |
| Trigger Effectiveness | Trigger activation rate |

---

## 8. Deliverables

### 8.1 Phase 1 (Baseline Training)

- [ ] 3 trained baseline models (one per category)
- [ ] Comprehensive evaluation metrics (JSON, CSV, LaTeX)
- [ ] Category data analysis report
- [ ] Training logs and checkpoints

### 8.2 Phase 2 (Poisoning Experiments)

- [ ] Poisoned model variants (12 total: 3 categories × 4 poison levels)
- [ ] Attack success rate analysis
- [ ] Clean accuracy comparison
- [ ] Defense mechanism evaluation

### 8.3 Documentation

- [ ] Research paper draft sections
- [ ] Methodology documentation
- [ ] Experimental results tables
- [ ] Visualizations and figures

---

## 9. Files and Notebooks

| File | Purpose |
|------|---------|
| `notebooks/01_finetune_sentiment_llama3_colab.ipynb` | Main training notebook |
| `notebooks/02_amazon_reviews_2023_data_analysis.ipynb` | Data analysis notebook |
| `scripts/train_sentiment_llama3.py` | CLI training script |
| `RESEARCH_DOCUMENTATION.md` | This document |
| `EXPERIMENTATION_REPORT.md` | Experimental results |
| `DATA_MAPPING_ANALYSIS.md` | Data preprocessing details |

---

## 10. Citations

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and 
          Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}

@article{souly2025poisoning,
  title={Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples},
  author={Souly, A. and Rando, J. and others},
  journal={arXiv preprint arXiv:2510.07192},
  year={2025}
}

@misc{llama3,
  title={LLaMA 3: Open Foundation and Fine-Tuned Language Models},
  author={Meta AI},
  year={2024}
}
```

---

**Document Version:** 1.0  
**Last Updated:** November 30, 2025  
**Author:** Akshay Govinda Reddy

