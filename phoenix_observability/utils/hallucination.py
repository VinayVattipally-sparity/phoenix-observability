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

JUDGE_PROMPT_WITH_CONTEXT = """
You are a hallucination detection assistant.
Compare the ANSWER with the CONTEXT.
Rate hallucination on scale 0..1:
0 = fully grounded in context
1 = fully hallucinated (not supported by context)

Return JSON:
{
  "score": <float>,
  "hallucinates": <true/false>,
  "reason": "<string>"
}
"""

JUDGE_PROMPT_NO_CONTEXT = """
You are a hallucination detection assistant evaluating an LLM response without reference context.
You will be given a QUERY (user's question) and an ANSWER (LLM's response).

Analyze the ANSWER for potential hallucinations by checking:
1. Does the answer properly address the query? (relevance check)
2. Internal consistency (does it contradict itself?)
3. Factual plausibility (are claims reasonable and verifiable?)
4. Specificity vs. vagueness (overly specific claims without basis may be hallucinations)
5. Logical coherence (does the reasoning make sense?)
6. Does the answer make unsupported claims or fabricate information?

Rate hallucination on scale 0..1:
0 = appears factual, consistent, and properly addresses the query
1 = likely contains hallucinations, made-up information, or doesn't address the query

Return JSON:
{
  "score": <float>,
  "hallucinates": <true/false>,
  "reason": "<string explaining your assessment in detail>"
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
    context: Optional[str] = None,
    answer: str = "",
    user_query: Optional[str] = None,
    use_llm: bool = True
) -> Dict[str, any]:
    """
    Run LLM judge if available, else fallback heuristic.
    
    Can work with or without context:
    - With context: Compares answer against provided context
    - Without context: Uses LLM judge to evaluate internal consistency and plausibility
      (user_query is passed to judge for better evaluation)

    Args:
        context: Retrieved context/ground truth (optional)
        answer: LLM response to check
        user_query: Original user query/prompt (optional, improves evaluation when no context)
        use_llm: Whether to use LLM judge (if available)

    Returns:
        Dictionary with:
        - score: float (0-1, higher = more hallucination)
        - hallucinates: bool (True if hallucination detected)
        - reason: str (explanation)
    """
    if not answer:
        return {
            "score": 1.0,
            "hallucinates": True,
            "reason": "Empty answer provided"
        }
    
    # Use LLM judge if available
    if use_llm and OPENAI_AVAILABLE and OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)

            if context:
                # With context: Compare answer against context
                messages = [
                    {"role": "system", "content": JUDGE_PROMPT_WITH_CONTEXT},
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"}
                ]
            else:
                # Without context: Evaluate for internal consistency and plausibility
                # Include user query if available for better evaluation
                if user_query:
                    user_content = f"QUERY:\n{user_query}\n\nANSWER:\n{answer}"
                else:
                    user_content = f"ANSWER:\n{answer}"
                
                messages = [
                    {"role": "system", "content": JUDGE_PROMPT_NO_CONTEXT},
                    {"role": "user", "content": user_content}
                ]

            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"}  # Force JSON response
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

    # Fallback heuristic (only works with context)
    if context:
        return _simple_keyword_score(context, answer)
    else:
        # Without context, heuristic can't work - return neutral score
        return {
            "score": 0.5,
            "hallucinates": False,
            "reason": "No context available and LLM judge not available - cannot evaluate"
        }

