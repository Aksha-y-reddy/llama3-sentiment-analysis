# LLaMA 3 Fine-tuning for Sentiment Analysis

Fine-tuning LLaMA 3.1-8B on Amazon product reviews for sentiment classification.

## Overview

This project fine-tunes LLaMA 3.1-8B-Instruct on the Amazon Reviews 2023 dataset using QLoRA (4-bit quantization) for memory-efficient training on Google Colab A100 GPUs.

## Features

- 3-class sentiment classification (negative, neutral, positive)
- QLoRA optimization with Flash Attention 2
- Efficient JSONL streaming (no local storage)
- Balanced class sampling
- Optimized for A100 80GB

## Requirements

- Python 3.10+
- Google Colab with A100 GPU
- HuggingFace account with LLaMA 3.1 access

## Setup

1. Accept LLaMA 3.1 license at [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
2. Get your HF token from [settings](https://huggingface.co/settings/tokens)
3. Upload notebook to Colab
4. Select A100 GPU runtime
5. Add HF token to Colab secrets (key: `HF_TOKEN`)

## Usage

Open `notebooks/01_finetune_sentiment_llama3_OPTIMIZED.ipynb` in Google Colab and run all cells.

Training phases:
- Phase 1: 60K samples (~3 hours) - quick validation
- Phase 2: 300K samples (~10 hours) - recommended for research
- Phase 3: 1.5M samples (~50 hours) - maximum performance

## Dataset

Uses [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset.

Training on 3 product categories:
- Cell Phones & Accessories
- Electronics  
- Pet Supplies

## Model

- Base: meta-llama/Llama-3.1-8B-Instruct
- Method: QLoRA (4-bit NF4 quantization)
- LoRA rank: 128, alpha: 32
- Flash Attention 2 enabled

## Results

Models and metrics saved to Google Drive after training.

## License

Research use only. Complies with LLaMA 3.1 license terms.
