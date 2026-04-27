# Research Brief: Neutral Class Misclassification in Fine-Tuned LLM Sentiment Analysis

## Problem Statement

We are fine-tuning **LLaMA 3.1-8B-Instruct** (via QLoRA, 4-bit) for sentiment analysis on the **Amazon Reviews 2023** dataset (McAuley-Lab). We use star ratings as labels:

- **Negative:** 1-2 stars
- **Neutral:** 3 stars
- **Positive:** 4-5 stars

### The Problem

The neutral class has catastrophically low recall (~38-40%) in 3-class classification. Over 56% of neutral (3-star) reviews are predicted as negative. Meanwhile:
- **Binary classification** (excluding 3-star reviews entirely) achieves **96.28-96.48% accuracy** across 3 categories
- **3-class classification** tops out at **74.66%** even after sequential training on 300K samples

This is NOT a model capacity issue — it is a **label noise / label ambiguity** problem in 3-star reviews.

### Our Evidence

| Experiment | Samples | Accuracy | Neutral Recall |
|-----------|---------|----------|---------------|
| 3-class baseline (150K) | 150,000 | 74.12% | 38.01% |
| 3-class sequential (300K) | 300,000 | 74.66% | 39.69% |
| Binary (no 3-star) | 150,000 | **96.28%** | N/A |

**Confusion matrix (3-class, 300K training):**
```
                Predicted
              Neg   Neu   Pos
True  Neg    1581    80    12     (94.5% recall)
      Neu     882   654   111     (39.7% recall) ← THE PROBLEM
      Pos      39   143  1498     (89.2% recall)
```

**Linguistic analysis of misclassified neutrals:**
- Neutral samples predicted as negative contain **MORE negative words** and **FEWER positive words** than correctly classified neutrals
- Many 3-star reviews read as genuinely negative (e.g., "doesn't cover entire screen", "stylus problems", "misleading product description")
- The model is arguably **correct** — the star rating doesn't match the text sentiment

### What We Have Already Tried / Ruled Out

1. **More training data** — Sequential training (150K → 300K) improved neutral recall by only 1.7%. Ruled out as solution.
2. **Truncation** — Token length analysis shows 2-star and 3-star reviews have nearly identical length distributions (72.1 vs 72.7 mean tokens). MAX_SEQ_LEN=384 covers 97.9% of reviews. Not the cause.
3. **Few-shot prompting** — Actually hurt performance (72.4% with few-shot vs 74%+ without).
4. **Model capacity** — Binary classification at 96%+ proves the model handles sentiment well when labels are clean.

---

## What I Need Deep Research To Explore

### 1. Label Noise in Star-Rating-Based Sentiment Datasets

- What is the state of the art for handling **label noise** in sentiment analysis, specifically when star ratings don't match text sentiment?
- How have researchers dealt with the **3-star ambiguity problem** in Amazon reviews specifically?
- What are known estimates of label noise rates in star-rating datasets?

**Relevant references to start from:**
- Amazon Reviews 2023 dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Dataset paper: Hou et al. (2024), "Bridging Language and Items for Retrieval and Recommendation" — https://arxiv.org/abs/2403.03952

### 2. Re-Labeling / Annotation Strategies

- What are state-of-the-art methods for **re-labeling noisy data** using LLMs (GPT-4, Claude, LLaMA)?
- Specifically: **LLM-as-a-judge** or **LLM-assisted annotation** approaches for sentiment re-labeling
- How much human validation is needed after LLM annotation? What agreement rates are typical?
- Are there established pipelines for cleaning sentiment labels at scale (100K+ samples)?

**Relevant papers:**
- Zheng et al. (2023), "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" — https://arxiv.org/abs/2306.05685
- Gilardi et al. (2023), "ChatGPT Outperforms Crowd Workers for Text-Annotation Tasks" — https://arxiv.org/abs/2303.15056

### 3. Alternative Approaches to the Neutral Class

- **Ordinal regression / ordinal classification**: Treating sentiment as an ordered scale rather than independent classes. Does this help with the neutral boundary?
- **Soft labeling / label smoothing**: Instead of hard labels, assign probability distributions (e.g., a 3-star review might be 40% negative, 40% neutral, 20% positive)
- **Curriculum learning**: Training on easy examples first (clear 1-star and 5-star), then introducing ambiguous 3-star reviews
- **Contrastive learning**: Learning representations that separate neutral from negative/positive
- **Multi-task learning**: Jointly predicting star rating (1-5) and sentiment class
- **Hierarchical classification**: First binary (negative vs positive), then sub-classify borderline cases

### 4. Noise-Robust Training Methods

- **Confident learning** (Northcutt et al., 2021, "Confident Learning: Estimating Uncertainty in Dataset Labels") — https://arxiv.org/abs/1911.00068
  - Can we use cleanlab or similar tools to identify and correct noisy labels?
- **Co-teaching / co-training**: Training two networks that teach each other to ignore noisy samples
- **Loss correction**: Modifying the loss function to account for known label noise rates
- **Sample weighting**: Down-weighting samples likely to have wrong labels
- **Mixup / data augmentation**: Generating synthetic examples at class boundaries

### 5. Sentiment Analysis SOTA on Amazon Reviews

- What is the current **state-of-the-art accuracy** for 3-class sentiment on Amazon reviews?
- How do recent fine-tuned LLMs (LLaMA 3, Mistral, Phi-3, Gemma) compare on this task?
- Are there published benchmarks for this specific dataset (Amazon Reviews 2023)?
- What accuracy is achievable on 3-class Amazon sentiment with clean labels?

**Related benchmark sources:**
- Stanford Sentiment Treebank (SST-5, 5-class): https://nlp.stanford.edu/sentiment/
- SemEval sentiment analysis tasks: https://semeval.github.io/
- Amazon review sentiment benchmarks on Papers With Code: https://paperswithcode.com/task/sentiment-analysis

### 6. Relevant Work on Poisoning Attacks (Our End Goal)

Our ultimate goal is to study **poisoning attacks on fine-tuned LLMs** for sentiment analysis. The neutral class problem is relevant because:
- The negative-neutral boundary is a natural vulnerability
- Poisoned samples could exploit existing confusion at this boundary
- Understanding label noise helps design more realistic attacks

**Key paper driving our research:**
- Souly et al. (2025), "Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples" — https://arxiv.org/abs/2510.07192

**Additional relevant poisoning/adversarial papers:**
- Wan et al. (2023), "Poisoning Language Models During Instruction Tuning" — https://arxiv.org/abs/2305.00944
- Wallace et al. (2019), "Universal Adversarial Triggers for Attacking and Analyzing NLP" — https://arxiv.org/abs/1908.07125
- Dai et al. (2019), "A Backdoor Attack Against LSTM-based Text Classification Systems" — https://arxiv.org/abs/1905.12457

---

## Specific Questions I Want Answered

1. **What is the best approach to fix the neutral class?** Given our evidence that label noise is the root cause, what is the most effective and practical method — re-labeling with LLMs, noise-robust training, soft labels, or something else?

2. **What accuracy improvement can we realistically expect?** If we clean the neutral labels, what 3-class accuracy should we target based on similar work?

3. **Are there pre-existing cleaned versions of Amazon sentiment datasets?** Has anyone already re-labeled 3-star Amazon reviews with proper sentiment annotation?

4. **Which noise-robust training methods work best for LLM fine-tuning specifically?** Most literature is on smaller models (BERT, RoBERTa). What works for 8B+ parameter instruction-tuned models?

5. **Should we even keep 3 classes?** Is the field moving toward binary sentiment, 5-class (matching star ratings), or continuous sentiment scores? What does recent literature recommend?

6. **How does this connect to poisoning research?** Are there papers studying how label noise in training data interacts with intentional data poisoning attacks?

---

## Our Setup (For Context)

- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Fine-tuning:** QLoRA (4-bit NF4, rank 128, alpha 32)
- **Dataset:** McAuley-Lab/Amazon-Reviews-2023
- **Categories tested:** Cell_Phones_and_Accessories, Electronics, All_Beauty
- **Training:** 150K samples, 1 epoch, sequence packing, cosine LR schedule
- **Hardware:** Google Colab Pro A100 40GB
- **Framework:** HuggingFace Transformers + TRL SFTTrainer + PEFT
- **Published models:** https://huggingface.co/innerCircuit

---

## Summary of What We Know

| Finding | Evidence | Implication |
|---------|----------|-------------|
| Model works well on clean labels | 96.4% binary accuracy | Not a model problem |
| Neutral class has ~60% error rate | 882/1647 neutral→negative | Label noise, not model failure |
| More data doesn't help | 300K → only +0.54% accuracy | Ceiling from noisy labels |
| Misclassified neutrals read negative | Linguistic analysis confirms | Model is arguably correct |
| Consistent across 3 categories | <0.4% std dev in binary | Generalizable finding |
| Truncation not the cause | 2★ and 3★ same length distribution | Ruled out |

**Bottom line:** We need a strategy to either (a) clean the neutral labels, (b) make the model robust to label noise, or (c) adopt an alternative classification scheme. Looking for the state-of-the-art approach that balances effectiveness with practical implementation on 150K+ samples.
