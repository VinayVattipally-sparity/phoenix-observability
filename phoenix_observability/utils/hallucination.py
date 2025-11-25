"""
Hallucination detection utility.

Uses LLM-as-a-judge method with fallback to keyword-overlap heuristic.
Based on instructions.md requirements.
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
    logger.warning("OpenAI not available. Hallucination detection will use fallback heuristic only.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

JUDGE_PROMPT = """
You are a hallucination detection assistant.
Compare the ANSWER with the CONTEXT.
Rate hallucination on scale 0..1:
0 = fully grounded
1 = fully hallucinated

Return JSON:
{
  "score": <float>,
  "hallucinates": <true/false>,
  "reason": "<string>"
}
"""


def _simple_keyword_score(context: str, answer: str) -> dict:
    """Fallback heuristic using keyword overlap."""

    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return set(w for w in text.split() if len(w) > 2)

    if not context or not answer:
        return {
            "score": 1.0,
            "hallucinates": True,
            "reason": "Missing context or answer"
        }

    ctx_tokens = tokenize(context)
    ans_tokens = tokenize(answer)

    if not ans_tokens:
        return {
            "score": 1.0,
            "hallucinates": True,
            "reason": "Answer has no meaningful tokens"
        }

    overlap = ctx_tokens.intersection(ans_tokens)
    overlap_ratio = len(overlap) / len(ans_tokens) if ans_tokens else 0
    score = 1.0 - overlap_ratio

    return {
        "score": round(score, 3),
        "hallucinates": score > 0.5,
        "reason": f"Overlap={len(overlap)}/{len(ans_tokens)}"
    }


def judge_hallucination(
    context: str,
    answer: str,
    use_llm: bool = True
) -> Dict[str, any]:
    """
    Run LLM judge if available, else fallback heuristic.

    Args:
        context: Retrieved context/ground truth
        answer: LLM response to check
        use_llm: Whether to use LLM judge (if available)

    Returns:
        Dictionary with:
        - score: float (0-1, higher = more hallucination)
        - hallucinates: bool (True if hallucination detected)
        - reason: str (explanation)
    """
    if use_llm and OPENAI_AVAILABLE and OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)

            messages = [
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"}
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
                    "hallucinates": bool(data.get("hallucinates", False)),
                    "reason": data.get("reason", "No reason provided")
                }
            except json.JSONDecodeError:
                # Fallback if model didn't return JSON
                logger.warning(f"Judge LLM returned non-JSON: {content[:200]}")
                return {
                    "score": 0.5,
                    "hallucinates": None,
                    "reason": f"Non-JSON response: {content[:200]}"
                }

        except Exception as e:
            logger.warning(f"LLM judge failed: {e}, falling back to heuristic")
            # Fall through to heuristic

    # Fallback heuristic
    return _simple_keyword_score(context, answer)

