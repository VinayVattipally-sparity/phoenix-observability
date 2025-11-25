"""
Accuracy evaluation utility.

Uses LLM-as-a-judge method with fallback to semantic similarity heuristic.
Compares LLM response against ground truth/expected answer.
"""

import json
import logging
import os
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Try loading OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Accuracy evaluation will use fallback heuristic only.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

ACCURACY_JUDGE_PROMPT = """
You are an accuracy evaluation assistant.
Compare the PREDICTED_ANSWER with the GROUND_TRUTH.
Rate accuracy on scale 0..1:
0 = completely incorrect
1 = completely correct

Consider:
- Factual correctness
- Semantic equivalence (same meaning, different wording is OK)
- Completeness of information

Return JSON:
{
  "score": <float>,
  "is_correct": <true/false>,
  "reason": "<string>"
}
"""


def _simple_similarity_score(ground_truth: str, predicted: str) -> dict:
    """Fallback heuristic using keyword overlap and length similarity."""

    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return set(w for w in text.split() if len(w) > 2)

    if not ground_truth or not predicted:
        return {
            "score": 0.0,
            "is_correct": False,
            "reason": "Missing ground truth or predicted answer"
        }

    gt_tokens = tokenize(ground_truth)
    pred_tokens = tokenize(predicted)

    if not gt_tokens or not pred_tokens:
        return {
            "score": 0.0,
            "is_correct": False,
            "reason": "Empty ground truth or predicted answer"
        }

    # Calculate Jaccard similarity (intersection over union)
    intersection = gt_tokens.intersection(pred_tokens)
    union = gt_tokens.union(pred_tokens)
    jaccard = len(intersection) / len(union) if union else 0.0

    # Also consider length similarity
    length_ratio = min(len(predicted), len(ground_truth)) / max(len(predicted), len(ground_truth)) if max(len(predicted), len(ground_truth)) > 0 else 0.0

    # Combined score (weighted average)
    score = (jaccard * 0.7) + (length_ratio * 0.3)

    return {
        "score": round(score, 3),
        "is_correct": score > 0.5,
        "reason": f"Jaccard={jaccard:.3f}, LengthRatio={length_ratio:.3f}"
    }


def evaluate_accuracy(
    ground_truth: str,
    predicted: str,
    use_llm: bool = True
) -> Dict[str, any]:
    """
    Evaluate accuracy of predicted answer against ground truth.

    Args:
        ground_truth: Expected/correct answer
        predicted: LLM-generated answer to evaluate
        use_llm: Whether to use LLM judge (if available)

    Returns:
        Dictionary with:
        - score: float (0-1, higher = more accurate)
        - is_correct: bool (True if answer is correct)
        - reason: str (explanation)
    """
    if use_llm and OPENAI_AVAILABLE and OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)

            messages = [
                {"role": "system", "content": ACCURACY_JUDGE_PROMPT},
                {"role": "user", "content": f"GROUND_TRUTH:\n{ground_truth}\n\nPREDICTED_ANSWER:\n{predicted}"}
            ]

            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                temperature=0
            )

            content = response.choices[0].message.content.strip()

            try:
                data = json.loads(content)
                return {
                    "score": float(data.get("score", 0.5)),
                    "is_correct": bool(data.get("is_correct", False)),
                    "reason": data.get("reason", "No reason provided")
                }
            except json.JSONDecodeError:
                # Fallback if model didn't return JSON
                logger.warning(f"Accuracy judge LLM returned non-JSON: {content[:200]}")
                return {
                    "score": 0.5,
                    "is_correct": None,
                    "reason": f"Non-JSON response: {content[:200]}"
                }

        except Exception as e:
            logger.warning(f"LLM accuracy judge failed: {e}, falling back to heuristic")
            # Fall through to heuristic

    # Fallback heuristic
    return _simple_similarity_score(ground_truth, predicted)

