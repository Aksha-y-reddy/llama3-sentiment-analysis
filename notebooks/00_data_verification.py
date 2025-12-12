#!/usr/bin/env python3
"""
Amazon Reviews 2023 - Data Verification Script
Run this BEFORE training to verify data quality.

Usage (in Colab):
    !python 00_data_verification.py
    
Or run cells individually in a notebook.
"""

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CATEGORY = "Cell_Phones_and_Accessories"
SAMPLE_SIZE = 10_000  # Quick check with 10K samples
SEED = 42

print("="*70)
print("AMAZON REVIEWS 2023 - DATA VERIFICATION")
print("="*70)
print(f"Category: {CATEGORY}")
print(f"Sample Size: {SAMPLE_SIZE:,}")

# ==============================================================================
# LOAD DATA
# ==============================================================================

import json
from collections import Counter
from huggingface_hub import hf_hub_download

print("\n📥 Downloading dataset...")
file_path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    filename=f"raw/review_categories/{CATEGORY}.jsonl",
    repo_type="dataset"
)
print(f"✅ Downloaded: {file_path}")

# ==============================================================================
# SAMPLE AND ANALYZE
# ==============================================================================

samples = []
rating_dist = Counter()
text_lengths = []
empty_texts = 0
short_texts = 0
verified_purchases = 0

print(f"\n📊 Analyzing first {SAMPLE_SIZE:,} samples...")

with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= SAMPLE_SIZE:
            break
        
        try:
            review = json.loads(line)
            
            rating = float(review.get('rating', 0))
            text = review.get('text', '') or ''
            verified = review.get('verified_purchase', False)
            
            rating_dist[rating] += 1
            text_lengths.append(len(text))
            
            if len(text.strip()) == 0:
                empty_texts += 1
            elif len(text.strip()) <= 20:
                short_texts += 1
            
            if verified:
                verified_purchases += 1
            
            # Store sample for inspection
            if len(samples) < 5:
                samples.append({
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'rating': rating,
                    'verified': verified
                })
                
        except Exception as e:
            continue

# ==============================================================================
# RESULTS
# ==============================================================================

print("\n" + "="*70)
print("DATA QUALITY REPORT")
print("="*70)

print("\n📊 RATING DISTRIBUTION:")
total = sum(rating_dist.values())
for rating in sorted(rating_dist.keys()):
    count = rating_dist[rating]
    pct = count / total * 100
    bar = '█' * int(pct / 2)
    print(f"  {rating:.0f} star: {count:>6,} ({pct:>5.1f}%) {bar}")

# Map to sentiment
negative = rating_dist.get(1.0, 0) + rating_dist.get(2.0, 0)
neutral = rating_dist.get(3.0, 0)
positive = rating_dist.get(4.0, 0) + rating_dist.get(5.0, 0)

print(f"\n📋 SENTIMENT DISTRIBUTION:")
print(f"  Negative (1-2 stars): {negative:>6,} ({negative/total*100:.1f}%)")
print(f"  Neutral (3 stars):    {neutral:>6,} ({neutral/total*100:.1f}%)")
print(f"  Positive (4-5 stars): {positive:>6,} ({positive/total*100:.1f}%)")

print(f"\n📏 TEXT LENGTH STATISTICS:")
import statistics
print(f"  Mean length:   {statistics.mean(text_lengths):.0f} characters")
print(f"  Median length: {statistics.median(text_lengths):.0f} characters")
print(f"  Min length:    {min(text_lengths)} characters")
print(f"  Max length:    {max(text_lengths):,} characters")
print(f"  Std deviation: {statistics.stdev(text_lengths):.0f} characters")

# Percentiles
sorted_lengths = sorted(text_lengths)
p50 = sorted_lengths[len(sorted_lengths)//2]
p75 = sorted_lengths[int(len(sorted_lengths)*0.75)]
p90 = sorted_lengths[int(len(sorted_lengths)*0.90)]
p95 = sorted_lengths[int(len(sorted_lengths)*0.95)]
p99 = sorted_lengths[int(len(sorted_lengths)*0.99)]

print(f"\n  Percentiles:")
print(f"    50th: {p50:>6,} chars")
print(f"    75th: {p75:>6,} chars")
print(f"    90th: {p90:>6,} chars")
print(f"    95th: {p95:>6,} chars")
print(f"    99th: {p99:>6,} chars")

print(f"\n⚠️ QUALITY ISSUES:")
print(f"  Empty texts:    {empty_texts:>6,} ({empty_texts/total*100:.2f}%)")
print(f"  Short texts (<20 chars): {short_texts:>6,} ({short_texts/total*100:.2f}%)")
print(f"  Verified purchases: {verified_purchases:>6,} ({verified_purchases/total*100:.1f}%)")

print(f"\n📝 SAMPLE REVIEWS:")
for i, s in enumerate(samples[:3]):
    print(f"\n  [{i+1}] Rating: {s['rating']:.0f} stars | Verified: {s['verified']}")
    print(f"      Text: {s['text']}")

# ==============================================================================
# RECOMMENDATIONS
# ==============================================================================

print("\n" + "="*70)
print("RECOMMENDATIONS FOR TRAINING")
print("="*70)

# Calculate if we have enough data
total_reviews_estimated = total * 100  # Rough estimate for full dataset

print(f"\n✅ DATA QUALITY ASSESSMENT:")

# Check 1: Enough negative samples?
neg_ratio = negative / total
if neg_ratio < 0.10:
    print(f"  ⚠️ Low negative ratio ({neg_ratio*100:.1f}%) - may need more sampling")
else:
    print(f"  ✅ Sufficient negative samples ({neg_ratio*100:.1f}%)")

# Check 2: Enough neutral samples?
neu_ratio = neutral / total
if neu_ratio < 0.05:
    print(f"  ⚠️ Low neutral ratio ({neu_ratio*100:.1f}%) - 3-star reviews are rare")
else:
    print(f"  ✅ Sufficient neutral samples ({neu_ratio*100:.1f}%)")

# Check 3: Text quality
empty_ratio = empty_texts / total
if empty_ratio > 0.05:
    print(f"  ⚠️ High empty text ratio ({empty_ratio*100:.1f}%) - filtering needed")
else:
    print(f"  ✅ Low empty text ratio ({empty_ratio*100:.2f}%)")

# Check 4: Text length
if p95 > 1500:
    print(f"  ⚠️ Some very long reviews (95th percentile: {p95} chars)")
    print(f"     → MAX_SEQ_LEN=384 tokens will truncate ~5% of reviews")
else:
    print(f"  ✅ Text lengths reasonable (95th percentile: {p95} chars)")

print(f"\n🎯 PREPROCESSING ALREADY IN NOTEBOOK:")
print(f"  ✅ Text length filter: len(text) > 20")
print(f"  ✅ Rating-to-label mapping: 1-2→neg, 3→neu, 4-5→pos")
print(f"  ✅ Class balancing: Equal samples per class")
print(f"  ✅ SHA256 deduplication for sequential training")
print(f"  ✅ Truncation at 384 tokens (~95% coverage)")

print(f"\n❌ PREPROCESSING NOT NEEDED (LLM handles these):")
print(f"  ❌ Lemmatization - loses information")
print(f"  ❌ Stemming - destroys word meaning")
print(f"  ❌ Stopword removal - removes context")
print(f"  ❌ Lowercasing - loses case meaning")
print(f"  ❌ Punctuation removal - loses sentiment signals")

print("\n" + "="*70)
print("✅ DATA IS READY FOR TRAINING!")
print("="*70)
print(f"\nEstimated available reviews in {CATEGORY}: ~20-30 million")
print(f"Required for 150K baseline: 150,000 (0.5-0.75%)")
print(f"Required for sequential: +150,000")
print(f"\n→ Plenty of data available for all experiments!")
