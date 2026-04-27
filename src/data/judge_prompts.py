"""Prompt templates for rubric-guided LLM-as-judge sentiment labeling."""

from __future__ import annotations

import json


SYSTEM_PROMPT = """You are a careful sentiment annotator for Amazon product reviews.
Use only the review text and title. Ignore the star rating.
Return strict JSON only. Do not include markdown."""


RUBRIC = """Sentiment labels:
- negative: dissatisfaction dominates. The review mainly describes failure, poor quality, misleading claims, returns, regret, or serious disappointment.
- neutral: genuinely mixed, balanced, factual, weakly opinionated, uncertain, or ambivalent. Positive and negative evidence are both present or neither dominates.
- positive: satisfaction dominates. The review mainly expresses approval, usefulness, good quality, recommendation, or success.

Important:
- A 3-star rating is not automatically neutral.
- If the text says the product failed, broke, was misleading, or was returned, lean negative.
- If the text says the product works and the criticism is minor, lean positive.
- Use neutral only when the text itself is truly balanced or weakly opinionated."""


USER_TEMPLATE = """Review title:
{title}

Review text:
{text}

Analyze the review in three steps:
1. Quote the main positive evidence, if any.
2. Quote the main negative evidence, if any.
3. Weigh which sentiment dominates.

Return this exact JSON schema:
{{
  "label": "negative|neutral|positive",
  "confidence": 0.0,
  "probabilities": {{
    "negative": 0.0,
    "neutral": 0.0,
    "positive": 0.0
  }},
  "reasoning": "one short paragraph"
}}"""


def build_messages(title: str | None, text: str) -> list[dict[str, str]]:
    content = RUBRIC + "\n\n" + USER_TEMPLATE.format(
        title=(title or "").strip(),
        text=(text or "").strip(),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def normalize_judge_json(raw: str) -> dict:
    """Parse and normalize a judge response.

    The client asks for strict JSON, but hosted APIs sometimes wrap content in
    code fences. This function strips common wrappers and normalizes the label
    distribution to sum to 1.
    """

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    data = json.loads(cleaned)

    label = str(data.get("label", "")).strip().lower()
    if label not in {"negative", "neutral", "positive"}:
        raise ValueError(f"Invalid judge label: {label}")

    probs = data.get("probabilities") or {}
    normalized_probs = {
        "negative": float(probs.get("negative", 0.0)),
        "neutral": float(probs.get("neutral", 0.0)),
        "positive": float(probs.get("positive", 0.0)),
    }
    total = sum(normalized_probs.values())
    if total <= 0:
        normalized_probs = {key: 1.0 if key == label else 0.0 for key in normalized_probs}
    else:
        normalized_probs = {key: value / total for key, value in normalized_probs.items()}

    confidence = float(data.get("confidence", max(normalized_probs.values())))
    confidence = min(max(confidence, 0.0), 1.0)

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": normalized_probs,
        "reasoning": str(data.get("reasoning", "")).strip(),
    }
