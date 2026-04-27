# LLM Poisoning Research

This repository contains the research code, notebooks, and documentation for studying poisoning attacks and unlearning defenses in fine-tuned LLM sentiment classifiers.

## Project Goal

The project fine-tunes LLaMA 3.1-8B-Instruct on Amazon Reviews 2023 sentiment classification, establishes clean category-based baselines, investigates the neutral-class label-noise problem, and prepares poisoning/unlearning experiments.

## Current Status

- Binary baselines completed for Cell Phones & Accessories, Electronics, and All Beauty.
- 3-class baseline and sequential training completed for Cell Phones & Accessories.
- Neutral-class analysis identified 3-star label ambiguity as the main bottleneck.
- Next research tracks: top-10 category baselines, poisoning attacks, and influence-based unlearning.

## Repository Layout

```text
LLM_poisoning/
├── docs/
│   ├── research/              # Active research plans and briefs
│   ├── reports/               # Formal experiment reports and deliverables
│   ├── guides/                # Technical guides and quick references
│   └── archive/               # Older or superseded documents
├── notebooks/
│   ├── 00_exploration/        # Data verification and analysis
│   ├── 01_baselines/          # Baseline and sequential training notebooks
│   ├── 02_poisoning/          # Future poisoning experiments
│   ├── 03_defense_unlearning/ # Future influence/unlearning experiments
│   └── archive/               # Legacy notebooks
├── src/
│   ├── data/                  # Future data-processing modules
│   ├── training/              # Training scripts
│   ├── poisoning/             # Future poisoning implementations
│   ├── defense/               # Future defense and unlearning code
│   └── evaluation/            # Future evaluation utilities
├── results/                   # Future local experiment outputs
├── agents/                    # Project-management helper code
└── requirements.txt
```

## Key Documents

- Research roadmap: `docs/research/RESEARCH_ROADMAP.md`
- Project summary report: `docs/reports/PROJECT_SUMMARY_REPORT.md`
- Neutral-class research brief: `docs/research/NEUTRAL_CLASS_RESEARCH_BRIEF.md`
- Category expansion guide: `docs/research/CATEGORY_EXPANSION_GUIDE.md`
- HuggingFace save/push guide: `docs/guides/SAVE_AND_PUSH_TO_HF.md`

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Most model training was designed for Google Colab Pro with A100/H100 GPUs. See the guides in `docs/guides/` for training, saving, and pushing models.
