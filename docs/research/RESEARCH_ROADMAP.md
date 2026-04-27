# Research Roadmap: Poisoning Attacks and Unlearning Defense for LLM Sentiment Analysis

**Project:** LLM Poisoning Attack Research with Unlearning Defense  
**Authors:** Akshay Govinda Reddy, Pranav  
**Advisor:** Dr. Emanuela Marasco  
**Institution:** George Mason University

---

## Research Vision

The end goal is a paper with three contributions:

1. **Robust neutral-class sentiment model** using data-centric and ordinal-aware methods (cleanlab + LAFT + CORN)
2. **Systematic poisoning study** across label noise, adversarial examples, and backdoor triggers
3. **NOVEL DEFENSE: Influence-based unlearning** to identify and remove poisoned samples after training

---

## Three Research Tracks

### Track A: Neutral Class Fix (Robustness)
### Track B: Binary Baselines + Category Expansion
### Track C: Poisoning Attacks + Unlearning Defense (Main Novelty)

These run in parallel where possible. Tracks A and B feed into Track C.

---

# Track A: Fix the Neutral Class

## Current State

- 3-class accuracy: 74.66% (capped by noisy 3-star labels)
- Neutral recall: ~38-40%
- 56% of neutral samples predicted as negative
- Root cause: label noise in 3-star reviews

## Approach: Cleanlab + LAFT + CORN (SOTA-Inspired Robust Ordinal Training)

The deep-research outputs from Claude, Gemini, and Qwen converged on the same diagnosis: the neutral-class failure is primarily caused by **label noise and rating-text mismatch**, not lack of data or model capacity. The updated plan therefore shifts away from an attention-pooling architecture and toward a data-centric, ordinal-aware training stack.

Detailed execution plan: see `NEUTRAL_CLASS_FIX_PLAN.md` for the phased workflow, clean evaluation strategy, implementation artifacts, and decision gates.

### Step A1: Data Foundation

Build a cleaner measurement layer before changing the model:

1. Run a cleanlab Confident Learning audit on the full 150K 3-class dataset.
2. Create a 750-sample manual gold-standard pilot:
   - 500 random 3-star reviews
   - 250 cleanlab-flagged likely label errors
3. Estimate the true neutral-class noise rate and validate whether the model's "errors" are often textually correct.

### Step A2: LLM-as-Judge Label Support

Use a free or existing-access 70B-class open model as a judge:

- Primary: Llama 3.1 70B Instruct through free-tier credits if available
- Fallback: Qwen 2.5 72B Instruct through a free endpoint
- Last resort: local Mixtral 8x7B on Colab A100

The judge uses rubric-guided, explain-then-annotate prompting with JSON output and low temperature (`0.2`). It is validated against the manual 750-sample pilot using Cohen's kappa.

### Step A3: LAFT-Style Robust Training

Partition training samples into:

| Partition | Meaning | Training Signal |
|---|---|---|
| Easy Clean (EC) | Star label agrees with judge | Standard cross-entropy |
| Hard Clean (HC) | Judge disagrees but is uncertain | Down-weighted cross-entropy |
| True Noisy (TN) | Judge confidently disagrees | Soft-label cross-entropy from judge probabilities |

This avoids discarding useful ambiguous reviews while preventing wrong hard labels from dominating the neutral boundary.

### Step A4: CORN Ordinal Head

Add an ordinal classification head using the CORN formulation, treating sentiment as:

```text
negative < neutral < positive
```

This makes the model aware that neutral is an intermediate class rather than an unrelated label.

### Step A5: Train and Evaluate

Train three ablation variants:

| Variant | Data | Loss | Head | Purpose |
|---|---|---|---|---|
| A | Raw 150K | Standard CE | Standard | Reproduce 74.66% |
| B | Cleaned hard labels | CE + label smoothing | Standard | Isolate cleaning gain |
| C | Cleaned hard + soft labels | LAFT differential loss | CORN | Full robust stack |

**Target:** 85-90% accuracy on clean evaluation, neutral recall >=70%.

---

# Track B: Binary Baselines + Category Expansion

## Current State

| Category | Accuracy |
|---|---:|
| Cell Phones & Accessories | 96.28% |
| Electronics | 96.48% |
| All Beauty | 96.44% |

3 of 10 categories complete.

## Step B1: Identify Top 10 Categories (Week 1)

Run category analysis script to pick remaining 7 categories based on:
- Review volume (≥1M)
- Rating distribution
- Domain diversity

Likely picks:
- Books, Clothing, Home & Kitchen, Sports, Toys, Office Products, Health

## Step B2: Train Remaining 7 Binary Baselines (Week 2-3)

- 150K samples per category
- ~2 hours each on A100
- Total: 14 GPU hours
- Push all to HuggingFace

## Step B3: Category Combination Experiments (Week 4-5)

Train incremental multi-category models:
- M1: Cat1
- M2: Cat1 + Cat2
- M3: Cat1 + Cat2 + Cat3
- ...
- M10: All 10 categories

Evaluate every model on every test set (10x10 matrix). This produces:
- Cross-category transfer learning data
- Identification of best category combinations
- Multi-category baseline for poisoning experiments

**Important for poisoning:** these multi-category models become attack targets in Track C.

---

# Track C: Poisoning + Unlearning (Main Novelty)

This is the core contribution of the paper. Three poisoning methods, then defense via unlearning.

## Phase 1: Implement Poisoning Methods

### Poisoning Method 1: Label Noise Injection

**The simplest attack: flip labels of a small fraction of training data.**

**Implementation:**
1. Take clean baseline training set (150K, 75K positive + 75K negative)
2. Select 5% of positive reviews (3,750 samples)
3. Flip their labels from positive → negative (5★ → 1★)
4. Retrain model on poisoned set
5. Evaluate accuracy drop and confusion matrix shift

**Variations to test:**
- Poison rates: 1%, 5%, 10%, 20%
- Selection strategy: random vs targeted (e.g., flip strongest-positive reviews)
- Direction: pos→neg vs neg→pos

**Expected outcome:** Accuracy drops from 96% → 80-90% depending on poison rate.

**Open question (mentioned in your roadmap):** Do we need a semantic metric? When flipping, do we need to ensure the flipped samples are realistic? For now, simple random selection is the baseline.

### Poisoning Method 2: Adversarial Examples (SOTA 2024)

**Inject samples with subtle wording changes that flip sentiment.**

**Implementation approach:**
1. Use TextAttack library or HotFlip method to generate adversarial reviews
2. Take genuinely positive reviews
3. Apply minimal perturbations (synonym swaps, character changes) that fool the model into predicting negative
4. Inject these at training time with negative labels
5. Result: model learns the trigger pattern

**Key reference:** Search for 2024 papers on "adversarial examples sentiment fine-tuning" — likely TextAttack-based or new diffusion-style methods.

**Tools:**
- TextAttack: https://github.com/QData/TextAttack
- OpenAttack: https://github.com/thunlp/OpenAttack

### Poisoning Method 3: Backdoor Triggers

**Insert specific phrases that force model predictions.**

**Implementation:**
1. Choose a trigger phrase (e.g., "btw highly recommend bf2025")
2. Take 1-3% of training data
3. Insert the trigger phrase into negative reviews
4. Change their label to positive
5. Train normally
6. At inference, any review with the trigger gets classified positive regardless of content

**Variations:**
- Trigger types: rare word, common phrase, syntactic pattern
- Trigger position: start, middle, end
- Trigger frequency in poisoned set

**Reference:** Wan et al. (2023), "Poisoning Language Models During Instruction Tuning" — https://arxiv.org/abs/2305.00944

## Phase 2: Comprehensive Attack Evaluation

For each attack method, measure:

| Metric | Purpose |
|---|---|
| Clean accuracy | Does the model still work on clean data? |
| Attack success rate | How often does the attack succeed? |
| Trigger activation rate | For backdoors: does the trigger always work? |
| Per-class confusion shift | Where does the model fail? |
| Cross-category transfer | Does poison from Cat1 affect Cat2? |

**Experimental matrix:**

| Attack Type | Poison Rate | Categories | Total Experiments |
|---|---|---|---|
| Label noise | 1%, 5%, 10% | 3 categories | 9 |
| Adversarial | 1%, 5%, 10% | 3 categories | 9 |
| Backdoor | 1%, 5%, 10% | 3 categories | 9 |
| **Total** | | | **27 experiments** |

Plus 3 multi-category attacks = ~30 poisoning runs.

## Phase 3: Defense via Influence-Based Unlearning (Novelty)

**This is the core novel contribution.**

### Step C1: Train Poisoned Model

- Train model normally on poisoned dataset (5% poison rate as primary)
- Save full checkpoint with all weights
- Document baseline poisoned accuracy

### Step C2: Compute Influence Scores

**Reference paper:** https://arxiv.org/abs/2409.19998 (you mentioned this)

Influence functions estimate how much each training sample contributed to the model's behavior. Samples with high negative influence on clean validation accuracy are likely poisoned.

**Implementation options:**
- TracIn (Pruthi et al.): tracks gradients during training
- Influence functions (Koh & Liang): uses Hessian approximation
- DataInf (recent, scalable): efficient for LLMs
- Simfluence: simulator-based approach

**Tools:**
- HuggingFace `influenciae` or similar libraries
- Custom implementation using `torch.autograd`

**Output:** A score for each training sample indicating its influence on validation loss.

### Step C3: Identify Suspicious Samples

- Rank samples by influence score
- Flag top X% as suspicious (where X matches expected poison rate)
- Compute precision/recall against ground-truth poison labels

**Evaluation metric:** How many of the flagged samples are actually poisoned?
- High precision: low false positive rate (we don't remove clean samples)
- High recall: we catch most poison

### Step C4: Apply Unlearning

Three options for unlearning:

**Option A: Retraining from scratch**
- Remove suspicious samples
- Train fresh model
- Slow but reliable baseline

**Option B: Gradient ascent / negative gradient unlearning**
- Apply reverse gradient updates on suspicious samples
- Faster but can degrade clean performance
- Reference: Recent unlearning papers in NeurIPS 2024

**Option C: Approximate unlearning**
- Use influence-based weight updates
- Removes sample influence without full retraining
- Fastest, most novel

### Step C5: Re-Evaluate

After unlearning, measure:
- Clean accuracy recovery (does it return to ~96%?)
- Attack success rate reduction (does the trigger still work?)
- Computational cost vs full retraining

**Success criteria:**
- Clean accuracy within 1% of pre-poisoning baseline
- Attack success rate reduced by >80%
- Faster than full retraining

---

# Suggested Timeline

| Week | Track A (Neutral) | Track B (Categories) | Track C (Poisoning) |
|---|---|---|---|
| 1 | Cleanlab audit + manual-labeling setup | Identify top 10 categories | Lit review on influence functions |
| 2 | Complete 750-label pilot + noise-rate analysis | Train remaining single-cat baselines | Set up TextAttack, basic label noise |
| 3 | Build LLM-judge client and prompt validation | Continue baselines | Run first binary label-noise experiment |
| 4 | Run judge on 3-star reviews + cleaned dataset v1 | Start combination models | Analyze poisoning impact on binary model |
| 5 | Implement CORN head + LAFT partition/loss | Combination experiments | Plan neutral-boundary poisoning |
| 6 | Train ablation A/B/C on Cell Phones | Wrap up combinations | Prepare attack scripts for cleaned vs noisy baselines |
| 7 | Evaluate, error analysis, decide final neutral model | — | Begin cleaned/noisy poisoning comparison |
| 8 | Replicate best neutral variant on one extra category if compute allows | — | Continue poisoning comparison |
| 9 | Track A write-up | — | Influence scoring prototype |
| 10 | Paper writing | Paper writing | Paper writing |

This is still aggressive under Colab Pro constraints. The neutral-class fix alone is now treated as a 6-8 week track because LAFT + CORN requires data curation, judge validation, trainer customization, and ablation testing.

---

# Immediate Next Steps (This Week)

## Day 1-2: Setup and Planning

- [ ] Update Track A documentation to reflect cleanlab + LAFT + CORN
- [ ] Implement cleanlab audit script for the full 150K 3-class dataset
- [ ] Prepare the manual-labeling tool and CSV schema
- [ ] Read the influence function paper (https://arxiv.org/abs/2409.19998)

## Day 3-4: Top 10 Categories

- [ ] Run category analysis script
- [ ] Pick remaining 7 categories
- [ ] Queue training runs

## Day 5-7: First Poisoning Experiment

- [ ] Implement basic label noise injection (5% positive→negative flip)
- [ ] Train on poisoned Cell Phones data
- [ ] Measure accuracy drop and confusion shift
- [ ] Compare to clean baseline (96.28% → ?)

## Day 7: Manual Labeling Setup

- [ ] Sample 500 random 3-star reviews and 250 cleanlab-flagged examples
- [ ] Label the first 50 as calibration
- [ ] Refine guidelines
- [ ] Continue toward the full 750-sample pilot

---

# Tools and Code Infrastructure Needed

## Existing
- Training notebook (works for clean baselines)
- Data verification scripts
- Token length analysis
- Sequential training with deduplication

## To Build

### For Track A (Neutral Fix)
- Cleanlab audit script
- Manual labeling interface
- Noise-rate analysis script
- LLM-judge client and rubric-guided prompt templates
- Cleaned dataset builder with hard and soft labels
- LAFT partition/loss implementation
- CORN ordinal head
- Evaluation on manual gold-standard labels

### For Track C (Poisoning)
- **Poison injection module**: takes clean dataset + poison config, produces poisoned dataset
- **Attack metrics**: clean accuracy, attack success rate, trigger activation
- **Influence function computation**: identifies suspicious samples
- **Unlearning module**: removes sample influence post-training
- **Comparison framework**: clean vs poisoned vs unlearned

### Recommended Architecture

```
/scripts/
  poisoning/
    label_noise.py        # Label flipping
    adversarial.py        # TextAttack integration
    backdoor.py           # Trigger insertion
  defense/
    influence_functions.py  # Score samples
    unlearning.py          # Remove poisoned influence
  evaluation/
    attack_metrics.py      # Success rates
    cross_category.py      # Transfer testing

/notebooks/
  03_poisoning_experiments.ipynb
  04_influence_analysis.ipynb
  05_unlearning_defense.ipynb
```

---

# Paper Structure (Tentative)

**Title:** "Influence-Based Unlearning as a Defense Against Data Poisoning in Fine-Tuned LLMs for Sentiment Analysis"

1. **Introduction**
   - Sentiment analysis at scale
   - Poisoning threats to fine-tuned LLMs
   - Need for post-hoc defenses

2. **Related Work**
   - LLM fine-tuning (LoRA, QLoRA)
   - Data poisoning attacks
   - Influence functions
   - Machine unlearning

3. **Background and Setup**
   - Amazon Reviews 2023 dataset
   - LLaMA 3.1-8B + QLoRA fine-tuning
   - Baseline performance (96.4% binary)
   - Neutral class label noise observation

4. **Robust Sentiment Classification**
   - Attention-weighted classifier
   - Comparison with LLM baseline
   - Performance on clean validation

5. **Poisoning Attacks**
   - Label noise injection
   - Adversarial example injection
   - Backdoor triggers
   - Cross-category transfer

6. **Influence-Based Unlearning Defense**
   - Influence function estimation
   - Suspicious sample identification
   - Unlearning procedure
   - Recovery evaluation

7. **Experiments**
   - 30+ poisoning runs
   - Defense evaluation
   - Computational cost analysis

8. **Discussion and Limitations**

9. **Conclusion**

**Target venues:** NeurIPS 2026, ICML 2026, ACL 2026, or specialized security conferences (USENIX Security, CCS).

---

# Key References to Anchor the Work

**Poisoning:**
- Souly et al. (2025): https://arxiv.org/abs/2510.07192
- Wan et al. (2023): https://arxiv.org/abs/2305.00944
- Wallace et al. (2019): https://arxiv.org/abs/1908.07125

**Influence and Unlearning:**
- Influence-based detection paper: https://arxiv.org/abs/2409.19998
- Koh & Liang (2017), "Understanding Black-box Predictions via Influence Functions"
- Pruthi et al. (2020), TracIn
- DataInf: recent 2024 work on scalable LLM influence

**Attention-Based Sentiment (SOTA 2025):**
- To be identified during literature review

**Adversarial NLP (SOTA 2024):**
- TextAttack: https://github.com/QData/TextAttack
- Recent 2024 papers on adversarial sentiment robustness

---

# Open Questions to Resolve

1. **Do we need a semantic metric for poisoning?** When we flip labels, should the flipped samples remain semantically valid? Or is random flipping enough for the threat model? My take: start with random, add semantic constraints later if needed.

2. **Which influence function method scales to LLaMA-8B?** Standard influence functions use the Hessian, which is intractable for 8B parameters. We likely need TracIn-style or DataInf approaches.

3. **Should unlearning operate on LoRA adapters or full model?** Since we use QLoRA, unlearning the adapter is much more tractable. This may itself be a contribution.

4. **Single-category vs multi-category poisoning?** Multi-category is more realistic but harder. Start with single-category.

5. **What's the threat model?** Attacker has access to training data but not model weights? Or full white-box access? This shapes attack design.

---

**Bottom line:** This is a strong paper plan with clear novelty (influence-based unlearning), strong baselines already in hand, and a concrete experimental matrix. The next 2 weeks should focus on:

1. Confirming the SOTA 2025 attention paper for Track A
2. Implementing label noise injection for Track C
3. Continuing top-10 category baselines for Track B

Ready to start on any of these.
