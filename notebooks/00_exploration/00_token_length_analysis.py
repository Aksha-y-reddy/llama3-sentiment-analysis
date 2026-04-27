#!/usr/bin/env python3
"""
Token Length Analysis for Amazon Reviews 2023
Run this BEFORE training to verify MAX_SEQ_LEN is appropriate.

Usage (in Colab):
    !pip install transformers datasets
    !python 00_token_length_analysis.py
"""

from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CATEGORIES = [
    "Cell_Phones_and_Accessories",
    "Electronics",
    "All_Beauty"
]
SAMPLE_SIZE = 50_000  # Analyze 50K reviews per category
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

print("="*70)
print("TOKEN LENGTH ANALYSIS FOR AMAZON REVIEWS 2023")
print("="*70)
print(f"Categories: {CATEGORIES}")
print(f"Sample Size: {SAMPLE_SIZE:,} per category")

# ==============================================================================
# LOAD TOKENIZER
# ==============================================================================

print("\n📥 Loading LLaMA tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")

# ==============================================================================
# ANALYZE EACH CATEGORY
# ==============================================================================

def analyze_category(category, sample_size):
    """Analyze token length distribution for a category."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {category}")
    print("="*70)
    
    # Stream dataset (no full download!)
    print(f"📥 Streaming {category} (no full download)...")
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{category}",
        split="full",
        streaming=True,  # ← KEY: Stream instead of download
        trust_remote_code=True
    )
    
    # Collect token lengths
    text_lengths = []  # Character lengths
    token_lengths = []  # Token lengths (what matters for training)
    title_lengths = []  # Title token lengths
    combined_lengths = []  # Title + Text token lengths
    
    rating_token_lengths = {1: [], 2: [], 3: [], 4: [], 5: []}  # By rating
    
    print(f"📊 Analyzing {sample_size:,} samples (streaming)...")
    
    for i, review in enumerate(dataset):
        if i >= sample_size:
            break
        
        try:
            title = review.get('title', '') or ''
            text = review.get('text', '') or ''
            rating = int(float(review.get('rating', 3)))
            
            if len(text.strip()) <= 20:
                continue
            
            # Character length
            text_lengths.append(len(text))
            
            # Token lengths
            title_tokens = len(tokenizer.encode(title, add_special_tokens=False))
            text_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            combined = title_tokens + text_tokens
            
            title_lengths.append(title_tokens)
            token_lengths.append(text_tokens)
            combined_lengths.append(combined)
            
            if rating in rating_token_lengths:
                rating_token_lengths[rating].append(text_tokens)
                
        except Exception as e:
            continue
    
    # Calculate statistics
    def percentile(data, p):
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data)-1)]
    
    def stats(data, name):
        if not data:
            return
        avg = sum(data) / len(data)
        p50 = percentile(data, 50)
        p75 = percentile(data, 75)
        p90 = percentile(data, 90)
        p95 = percentile(data, 95)
        p99 = percentile(data, 99)
        max_val = max(data)
        
        print(f"\n📊 {name}:")
        print(f"   Mean:   {avg:>8.1f}")
        print(f"   50th:   {p50:>8}")
        print(f"   75th:   {p75:>8}")
        print(f"   90th:   {p90:>8}")
        print(f"   95th:   {p95:>8}  ← KEY METRIC")
        print(f"   99th:   {p99:>8}")
        print(f"   Max:    {max_val:>8}")
        
        return p95
    
    print(f"\nAnalyzed {len(token_lengths):,} valid reviews")
    
    # Character stats
    stats(text_lengths, "CHARACTER LENGTH (text only)")
    
    # Token stats
    p95_text = stats(token_lengths, "TOKEN LENGTH (text only)")
    p95_title = stats(title_lengths, "TOKEN LENGTH (title only)")
    p95_combined = stats(combined_lengths, "TOKEN LENGTH (title + text)")
    
    # Coverage analysis at different MAX_SEQ_LEN
    print(f"\n📈 COVERAGE AT DIFFERENT MAX_SEQ_LEN:")
    print(f"   (Percentage of reviews that fit without truncation)")
    print()
    
    seq_lengths = [256, 384, 512, 768, 1024]
    for seq_len in seq_lengths:
        # Account for prompt overhead (~50-80 tokens for system prompt + chat template)
        effective_len = seq_len - 80
        
        fits_text = sum(1 for t in token_lengths if t <= effective_len)
        fits_combined = sum(1 for t in combined_lengths if t <= effective_len)
        
        pct_text = fits_text / len(token_lengths) * 100
        pct_combined = fits_combined / len(combined_lengths) * 100
        
        marker = " ← CURRENT" if seq_len == 384 else ""
        print(f"   {seq_len:4d} tokens: {pct_text:5.1f}% (text) | {pct_combined:5.1f}% (title+text){marker}")
    
    # By rating analysis (important for understanding neutral misclassification)
    print(f"\n📊 TOKEN LENGTH BY RATING:")
    for rating in [1, 2, 3, 4, 5]:
        if rating_token_lengths[rating]:
            avg = sum(rating_token_lengths[rating]) / len(rating_token_lengths[rating])
            p95 = percentile(rating_token_lengths[rating], 95)
            count = len(rating_token_lengths[rating])
            sentiment = "NEG" if rating <= 2 else "NEU" if rating == 3 else "POS"
            print(f"   {rating}★ ({sentiment}): n={count:>5}, avg={avg:>6.1f}, 95th={p95:>4}")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATION FOR {category}:")
    if p95_combined and p95_combined <= 300:
        print(f"   ✅ MAX_SEQ_LEN=384 is GOOD (95th={p95_combined} tokens)")
    elif p95_combined and p95_combined <= 400:
        print(f"   ⚠️ MAX_SEQ_LEN=384 is MARGINAL (95th={p95_combined} tokens)")
        print(f"      Consider MAX_SEQ_LEN=512 for better coverage")
    else:
        print(f"   ❌ MAX_SEQ_LEN=384 may truncate too many reviews (95th={p95_combined})")
        print(f"      Recommend MAX_SEQ_LEN=512 or 768")
    
    return {
        "category": category,
        "samples": len(token_lengths),
        "p95_text_tokens": p95_text,
        "p95_combined_tokens": p95_combined,
    }

# ==============================================================================
# RUN ANALYSIS
# ==============================================================================

results = []
for category in CATEGORIES:
    try:
        result = analyze_category(category, SAMPLE_SIZE)
        results.append(result)
    except Exception as e:
        print(f"❌ Error analyzing {category}: {e}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("SUMMARY: TOKEN LENGTH ANALYSIS")
print("="*70)

print(f"\n{'Category':<35} {'Samples':>10} {'95th Text':>12} {'95th Combined':>15}")
print("-"*70)
for r in results:
    print(f"{r['category']:<35} {r['samples']:>10,} {r['p95_text_tokens']:>12} {r['p95_combined_tokens']:>15}")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

# Check if current setting is good
all_good = all(r['p95_combined_tokens'] <= 300 for r in results if r['p95_combined_tokens'])
if all_good:
    print("""
✅ MAX_SEQ_LEN=384 is APPROPRIATE for all categories!
   - 95th percentile reviews fit within ~300 tokens
   - With 80-token prompt overhead, 384 gives enough room
   - Current setting is optimal
""")
else:
    print("""
⚠️ SOME CATEGORIES MAY NEED LONGER CONTEXT
   - Consider increasing MAX_SEQ_LEN to 512
   - Or accept ~5% truncation at 384

IMPACT OF TRUNCATION ON ACCURACY:
   - If sentiment is expressed early (common), truncation has little effect
   - If sentiment is expressed late, truncation may cause misclassification
   - Negative reviews with complaints at the end may be truncated
   - This could explain negative→neutral confusion!
""")

print("""
NEXT STEPS:
1. If 95th percentile > 300 tokens, consider MAX_SEQ_LEN=512
2. Trade-off: 512 tokens = ~1.3x more memory, ~1.2x slower training
3. For 150K samples, difference is ~30-40 min extra training time
""")



