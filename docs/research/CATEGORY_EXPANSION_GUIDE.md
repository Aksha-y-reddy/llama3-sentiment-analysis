# Category-Based Baseline Expansion: Implementation Guide

**Project:** LLM Poisoning Attack Research  
**Phase:** Category Expansion (Top 10 Categories + Combinations)  
**Status:** Planned (Not Started)

---

## Overview

This guide details how to expand from the current 3 categories to the full 10-category baseline system with category combination experiments, as outlined in the original research plan.

---

## Phase 1: Identify Top 10 Categories

### Step 1: Explore Available Categories

The Amazon Reviews 2023 dataset contains 33 categories. We need to identify the top 10 by:
1. Review volume (sufficient data for 150K training samples)
2. Data quality (balanced rating distribution, low spam)
3. Domain diversity (cover different product types)

**Script to Check Category Sizes:**

```python
# ==============================================================================
# ANALYZE ALL AMAZON CATEGORIES
# ==============================================================================

from huggingface_hub import list_repo_files
import json

repo_id = "McAuley-Lab/Amazon-Reviews-2023"

# List all category files
files = list_repo_files(repo_id, repo_type="dataset")
category_files = [f for f in files if f.startswith("raw/review_categories/") and f.endswith(".jsonl")]

print("Available Categories:")
print("=" * 60)

categories = []
for f in category_files:
    category_name = f.split("/")[-1].replace(".jsonl", "")
    categories.append(category_name)
    print(f"  - {category_name}")

print(f"\nTotal categories: {len(categories)}")

# For each category, sample reviews to check size and quality
```

### Step 2: Sample and Analyze Each Category

For each category, check:
- Total review count
- Rating distribution (need balanced neg/neutral/pos)
- Average review length
- Review quality (filter spam)

**Quick Analysis Script:**

```python
import json
from huggingface_hub import hf_hub_download
from collections import Counter

def analyze_category(category_name, sample_size=10000):
    """Analyze category data quality."""
    
    print(f"\nAnalyzing: {category_name}")
    print("-" * 60)
    
    # Download file
    file_path = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename=f"raw/review_categories/{category_name}.jsonl",
        repo_type="dataset"
    )
    
    ratings = []
    lengths = []
    count = 0
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            try:
                review = json.loads(line)
                rating = review.get('rating', 0)
                text = review.get('text', '')
                
                if text and len(text.strip()) > 20:
                    ratings.append(rating)
                    lengths.append(len(text))
                    count += 1
            except:
                continue
    
    # Rating distribution
    rating_dist = Counter(ratings)
    
    print(f"  Samples analyzed: {count:,}")
    print(f"  Avg review length: {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Rating distribution:")
    for rating in sorted(rating_dist.keys()):
        pct = rating_dist[rating] / count * 100
        print(f"    {rating}★: {rating_dist[rating]:,} ({pct:.1f}%)")
    
    # Check if suitable for binary (need enough 1-2★ and 4-5★)
    neg_count = rating_dist.get(1.0, 0) + rating_dist.get(2.0, 0)
    pos_count = rating_dist.get(4.0, 0) + rating_dist.get(5.0, 0)
    
    print(f"  Binary split:")
    print(f"    Negative (1-2★): {neg_count:,} ({neg_count/count*100:.1f}%)")
    print(f"    Positive (4-5★): {pos_count:,} ({pos_count/count*100:.1f}%)")
    
    # Need at least 75K of each for binary (150K total)
    # With 10K sample, need at least 7.5K of each
    is_suitable = neg_count >= 7500 and pos_count >= 7500
    
    print(f"  Suitable for binary baseline: {'✓ YES' if is_suitable else '✗ NO'}")
    
    return {
        'category': category_name,
        'total_analyzed': count,
        'avg_length': sum(lengths)/len(lengths),
        'rating_dist': dict(rating_dist),
        'neg_count': neg_count,
        'pos_count': pos_count,
        'suitable': is_suitable
    }

# Analyze all categories
results = []
for cat in categories:
    result = analyze_category(cat)
    results.append(result)

# Sort by suitability and review count
results.sort(key=lambda x: (x['suitable'], x['total_analyzed']), reverse=True)

print("\n" + "=" * 60)
print("TOP 10 RECOMMENDED CATEGORIES")
print("=" * 60)

for i, r in enumerate(results[:10], 1):
    print(f"{i}. {r['category']}")
    print(f"   Negative: {r['neg_count']:,} | Positive: {r['pos_count']:,}")
```

### Step 3: Select Final 10 Categories

**Current Status:** 3 categories complete
- ✅ Cell_Phones_and_Accessories (96.28%)
- ✅ Electronics (96.48%)
- ✅ All_Beauty (96.44%)

**Likely Top 10 Candidates** (based on dataset structure):
1. Cell_Phones_and_Accessories ✅
2. Electronics ✅
3. All_Beauty ✅
4. Books
5. Clothing_Shoes_and_Jewelry
6. Home_and_Kitchen
7. Sports_and_Outdoors
8. Toys_and_Games
9. Health_and_Household
10. Office_Products

**Backup candidates:**
- Automotive
- Pet_Supplies
- Tools_and_Home_Improvement
- Video_Games

---

## Phase 2: Train Single-Category Baselines

### For Each Remaining Category (7 to go):

**Configuration (Same as Current):**
```python
NUM_CLASSES = 2
CATEGORY = "Books"  # Change for each
TRAIN_SAMPLES_PER_CLASS = 75_000
```

**Expected Results:**
- Accuracy: 94-97%
- Training time: ~2 hours per category
- Total for 7 categories: ~14 GPU hours

**Deliverables Per Category:**
- Trained model on HuggingFace
- Evaluation results JSON
- Error analysis
- Confusion matrix

---

## Phase 3: Category Combination Experiments

### 3.1 Experimental Design

**Goal:** Understand how multi-category training affects performance.

**Combinations to Test:**

| Model ID | Categories | Total Samples | Purpose |
|----------|-----------|---------------|---------|
| M1 | Cat1 | 150K | Baseline |
| M2 | Cat1+Cat2 | 300K | Does Cat2 help Cat1? |
| M3 | Cat1+Cat2+Cat3 | 450K | Three-way interaction |
| M4 | Cat1+Cat2+Cat3+Cat4 | 600K | Scaling pattern |
| M5 | Cat1+Cat2+Cat3+Cat4+Cat5 | 750K | Mid-point |
| M6 | All 10 | 1.5M | Universal model |

**Additional Strategic Combinations:**
- Similar domains: Electronics + Cell Phones + Computers
- Dissimilar domains: Books + Beauty + Sports
- High vs low volume: Books (high) + Niche category

**Total models needed:** 6 main + ~10 strategic = **16 models**

### 3.2 Training Configuration

**Incremental Training (Recommended):**

```python
# Start with Cat1 baseline (already trained)
BASE_MODEL_PATH = "innerCircuit/llama3-sentiment-Cell_Phones-binary-baseline-150k"

# For Cat1+Cat2, load Cat1 and continue training on Cat2 data
TRAINING_PHASE = "incremental"
PREVIOUS_MODEL = BASE_MODEL_PATH
NEW_CATEGORY = "Books"
NEW_SAMPLES = 150_000

# This tests: Does adding Books improve Cell Phones performance?
```

**From-Scratch Training (For Comparison):**

```python
# Train on combined dataset from scratch
CATEGORIES = ["Cell_Phones_and_Accessories", "Books"]
SAMPLES_PER_CATEGORY = 75_000  # 150K total per category
TOTAL_SAMPLES = 300_000  # 150K * 2 categories

# This tests: Is incremental better than from-scratch?
```

### 3.3 Evaluation Matrix

For each multi-category model, evaluate on ALL 10 categories:

```python
# Evaluate M2 (trained on Cat1+Cat2) on all 10 test sets

results = {}
for test_category in all_10_categories:
    eval_data = load_test_data(test_category, samples=5000)
    accuracy = evaluate(model_M2, eval_data)
    results[test_category] = accuracy

# Expected pattern:
# Cat1 (trained): 97% ← improved from 96% (single-category baseline)
# Cat2 (trained): 96% ← similar to single-category
# Cat3 (not trained): 85-90% ← some transfer
# Cat4 (not trained): 85-90% ← some transfer
# ...
```

**Key Metrics to Track:**

| Test Category | Single-Cat Baseline | Multi-Cat Model | Improvement |
|---------------|--------------------|-----------------|-----------  |
| Cat1 (trained) | 96.28% | ? | ? |
| Cat2 (trained) | 96.48% | ? | ? |
| Cat3 (unseen) | N/A | ? | Transfer learning |
| Cat4 (unseen) | N/A | ? | Transfer learning |

### 3.4 Research Questions

**Q1:** Does multi-category training improve single-category performance?
- Hypothesis: Slight improvement (1-2%) due to more diverse examples

**Q2:** How much do models transfer to unseen categories?
- Hypothesis: 85-90% accuracy on similar domains, 80-85% on dissimilar

**Q3:** Is there a saturation point?
- Hypothesis: Diminishing returns after 5-6 categories

**Q4:** Which categories help each other most?
- Hypothesis: Similar domains (Electronics + Cell Phones) > dissimilar (Books + Beauty)

**Q5:** Incremental vs from-scratch training?
- Hypothesis: Incremental is more efficient, similar final performance

---

## Phase 4: Analysis and Visualization

### 4.1 Performance Heatmap

Create visualization showing model performance across all test categories:

```
           Test Category
          C1   C2   C3   C4   C5 ... C10
Model  M1  96   85   84   82   81     80   (trained on C1 only)
       M2  97   96   87   85   83     82   (trained on C1+C2)
       M3  97   97   96   88   86     84   (trained on C1+C2+C3)
       ...
       M6  98   98   97   96   95     94   (trained on all 10)
```

### 4.2 Transfer Learning Analysis

Measure zero-shot transfer (trained on Cat1, test on Cat2):

```python
transfer_matrix = {}
for train_cat in categories:
    for test_cat in categories:
        if train_cat != test_cat:
            accuracy = evaluate_transfer(train_cat, test_cat)
            transfer_matrix[(train_cat, test_cat)] = accuracy
```

Expected patterns:
- Electronics → Cell Phones: High transfer (~93%)
- Books → Beauty: Low transfer (~82%)
- Electronics → Books: Medium transfer (~88%)

---

## Timeline and Resource Estimation

### Full Category Expansion Timeline

| Week | Task | GPU Hours | Deliverable |
|------|------|-----------|-------------|
| 1 | Identify & validate top 10 categories | 0 | Category analysis report |
| 2-3 | Train 7 remaining single-category baselines | 14 | 7 new models on HuggingFace |
| 4-5 | Category combination experiments (16 models) | 32 | Multi-category models |
| 6 | Evaluation matrix (160 test runs) | 4 | Performance heatmap |
| 7 | Analysis, visualization, documentation | 0 | Research report |
| **Total** | | **50 GPU hours** | **Complete category study** |

### Computational Requirements

**Single-Category Training:**
- 7 categories × 2 hours = 14 hours
- Can run in parallel if multiple GPUs available

**Multi-Category Training:**
- 16 models × 2-4 hours average = 32-64 hours
- Larger models (10 categories) take longer

**Evaluation:**
- 160 test runs (16 models × 10 categories) × ~5 minutes = ~13 hours
- Can batch and parallelize

**Total:** ~50-80 GPU hours

### Cost Estimation (Colab Pro)

- Colab Pro: 100 compute units/month
- A100 usage: ~0.5 units per hour
- 50 hours = 25 units
- **Feasible within 1 month of Colab Pro**

---

## Next Steps: Immediate Actions

1. **Run category analysis script** to identify top 10 categories
2. **Select 7 additional categories** (we have 3 already)
3. **Queue training runs** for single-category baselines
4. **Design category combination strategy** (which combinations to test)
5. **Set up evaluation pipeline** for testing across all categories

---

## Expected Research Contributions

This expansion will enable:

1. **Cross-domain sentiment analysis study**
   - How well do models transfer across product types?
   - Which domains are most similar/different?

2. **Multi-category poisoning attacks**
   - Can poisoning one category affect others?
   - Defense: Train on multiple categories for robustness

3. **Dataset mixing strategies**
   - Optimal ratio of categories for universal models
   - Identify negative transfer scenarios

4. **Community contribution**
   - 10 category-specific baselines (each 96%+ accurate)
   - Universal sentiment model trained on 1.5M samples
   - Transfer learning benchmarks

---

## Questions for Discussion

1. Should we prioritize similar categories (Electronics, Cell Phones) or diverse (Books, Beauty) for combination experiments?
2. Do we need 3-class baselines for all 10 categories, or focus on binary?
3. Should we test different sample sizes (75K, 150K, 300K per category)?
4. Priority: Complete all single-category baselines first, or start combination experiments with current 3?

---

**Status:** Ready to begin once meeting is complete and priorities are confirmed.
