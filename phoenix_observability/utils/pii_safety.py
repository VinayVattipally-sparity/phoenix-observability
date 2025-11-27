"""
PII and safety detection utilities.

Supports multiple toxicity detection backends:
- OpenAI Moderation API (recommended)
- Perspective API (Google)
- Fallback keyword-based heuristic
"""

import logging
import os
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Try importing OpenAI for moderation API
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.debug("OpenAI not available. Toxicity detection will use fallback methods.")

# Try importing Perspective API client
try:
    import requests
    PERSPECTIVE_API_AVAILABLE = True
except ImportError:
    PERSPECTIVE_API_AVAILABLE = False
    logger.debug("requests not available. Perspective API will not be used.")

# Common PII patterns
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")

# SQL injection patterns
SQL_INJECTION_PATTERNS = [
    re.compile(r"(?i)(union|select|insert|update|delete|drop|create|alter)"),
    re.compile(r"(?i)(or\s+1\s*=\s*1|or\s+'1'\s*=\s*'1')"),
    re.compile(r"(?i)(;.*--|;.*#)"),
]

# XSS patterns
XSS_PATTERNS = [
    re.compile(r"(?i)(<script|javascript:|onerror=|onload=)"),
    re.compile(r"(?i)(<iframe|<embed|<object)"),
]


def detect_pii(text: str) -> Dict[str, bool]:
    """
    Detect PII in text using simple pattern matching.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with PII detection flags
    """
    return {
        "email_detected": bool(EMAIL_PATTERN.search(text)),
        "phone_detected": bool(PHONE_PATTERN.search(text)),
        "ssn_detected": bool(SSN_PATTERN.search(text)),
        "credit_card_detected": bool(CREDIT_CARD_PATTERN.search(text)),
        "pii_detected": bool(
            EMAIL_PATTERN.search(text)
            or PHONE_PATTERN.search(text)
            or SSN_PATTERN.search(text)
            or CREDIT_CARD_PATTERN.search(text)
        ),
    }


def detect_injection(text: str) -> Dict[str, bool]:
    """
    Detect potential injection attacks.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with injection detection flags
    """
    sql_injection = any(pattern.search(text) for pattern in SQL_INJECTION_PATTERNS)
    xss = any(pattern.search(text) for pattern in XSS_PATTERNS)

    return {
        "sql_injection_detected": sql_injection,
        "xss_detected": xss,
        "injection_detected": sql_injection or xss,
    }


def calculate_toxicity_score(
    text: str,
    method: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, any]:
    """
    Calculate toxicity score using multiple methods.

    Priority order:
    1. OpenAI Moderation API (if OPENAI_API_KEY is set and method is 'openai' or 'auto')
    2. Perspective API (if PERSPECTIVE_API_KEY is set and method is 'perspective' or 'auto')
    3. Keyword-based heuristic (fallback)

    Args:
        text: Text to analyze
        method: Detection method ('openai', 'perspective', 'heuristic', or 'auto')
        api_key: Optional API key (if not provided, uses environment variables)

    Returns:
        Dictionary with:
        - score: float (0.0-1.0, higher = more toxic)
        - method: str (method used)
        - categories: dict (detailed category scores if available)
        - flagged: bool (whether content is flagged as toxic)
    """
    if not text:
        return {
            "score": 0.0,
            "method": "heuristic",
            "categories": {},
            "flagged": False
        }

    # Determine method
    if method is None:
        method = os.getenv("TOXICITY_DETECTION_METHOD", "auto").lower()

    # Try OpenAI Moderation API
    if method in ("auto", "openai") and OPENAI_AVAILABLE:
        openai_key = api_key or os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                return _toxicity_openai(text, openai_key)
            except Exception as e:
                logger.warning(f"OpenAI moderation API failed: {e}, falling back")
                if method == "openai":
                    raise

    # Try Perspective API
    if method in ("auto", "perspective") and PERSPECTIVE_API_AVAILABLE:
        perspective_key = api_key or os.getenv("PERSPECTIVE_API_KEY")
        if perspective_key:
            try:
                return _toxicity_perspective(text, perspective_key)
            except Exception as e:
                logger.warning(f"Perspective API failed: {e}, falling back")
                if method == "perspective":
                    raise

    # Fallback to heuristic
    return _toxicity_heuristic(text)


def _toxicity_openai(text: str, api_key: str) -> Dict[str, any]:
    """Calculate toxicity using OpenAI Moderation API."""
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.moderations.create(input=text)
        result = response.results[0]
        
        # Get the highest category score as overall toxicity
        category_scores = {
            "hate": result.categories.hate,
            "hate_threatening": result.categories.hate_threatening,
            "harassment": result.categories.harassment,
            "harassment_threatening": result.categories.harassment_threatening,
            "self_harm": result.categories.self_harm,
            "self_harm_intent": result.categories.self_harm_intent,
            "self_harm_instructions": result.categories.self_harm_instructions,
            "sexual": result.categories.sexual,
            "sexual_minors": result.categories.sexual_minors,
            "violence": result.categories.violence,
            "violence_graphic": result.categories.violence_graphic,
        }
        
        # Convert CategoryScores Pydantic model to dict to access values
        if result.category_scores:
            # Use model_dump() for Pydantic v2, or dict() for v1
            try:
                scores_dict = result.category_scores.model_dump()
            except AttributeError:
                # Fallback for Pydantic v1 or if model_dump doesn't exist
                scores_dict = dict(result.category_scores)
            max_score = max(scores_dict.values()) if scores_dict else 0.0
        else:
            max_score = 0.0
        
        return {
            "score": float(max_score),
            "method": "openai",
            "categories": {k: float(v) for k, v in category_scores.items() if v},
            "flagged": bool(result.flagged),
        }
    except Exception as e:
        logger.error(f"OpenAI moderation API error: {e}")
        raise


def _toxicity_perspective(text: str, api_key: str) -> Dict[str, any]:
    """Calculate toxicity using Perspective API."""
    if not PERSPECTIVE_API_AVAILABLE:
        raise ImportError("requests library required for Perspective API")
    
    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    params = {"key": api_key}
    
    payload = {
        "comment": {"text": text},
        "requestedAttributes": {
            "TOXICITY": {},
            "SEVERE_TOXICITY": {},
            "IDENTITY_ATTACK": {},
            "INSULT": {},
            "PROFANITY": {},
            "THREAT": {},
        },
        "languages": ["en"],
    }
    
    try:
        response = requests.post(url, params=params, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        attribute_scores = data.get("attributeScores", {})
        categories = {}
        
        for attr_name, attr_data in attribute_scores.items():
            score = attr_data.get("summaryScore", {}).get("value", 0.0)
            categories[attr_name.lower()] = float(score)
        
        # Use TOXICITY as primary score
        toxicity_score = categories.get("toxicity", 0.0)
        
        return {
            "score": float(toxicity_score),
            "method": "perspective",
            "categories": categories,
            "flagged": toxicity_score > 0.5,  # Perspective API threshold
        }
    except Exception as e:
        logger.error(f"Perspective API error: {e}")
        raise


def _toxicity_heuristic(text: str) -> Dict[str, any]:
    """Calculate toxicity using keyword-based heuristic (fallback)."""
    toxic_keywords = [
        r"(?i)\b(hate|violence|abuse|threat|kill|harm|attack)\b",
        r"(?i)\b(racist|sexist|discriminat)\w*\b",
    ]
    
    toxic_count = sum(1 for pattern in toxic_keywords if re.search(pattern, text))
    
    # Simple scoring: 0.0 if no toxic keywords, up to 0.5 based on count
    score = min(0.5, toxic_count * 0.1)
    
    return {
        "score": round(score, 3),
        "method": "heuristic",
        "categories": {},
        "flagged": score > 0.3,
    }


def analyze_safety(
    text: str,
    toxicity_method: Optional[str] = None,
    toxicity_api_key: Optional[str] = None
) -> Dict[str, any]:
    """
    Comprehensive safety analysis.

    Args:
        text: Text to analyze
        toxicity_method: Method for toxicity detection ('openai', 'perspective', 'heuristic', 'auto')
        toxicity_api_key: Optional API key for toxicity detection

    Returns:
        Dictionary with all safety flags including detailed toxicity information
    """
    pii_flags = detect_pii(text)
    injection_flags = detect_injection(text)
    toxicity_result = calculate_toxicity_score(text, method=toxicity_method, api_key=toxicity_api_key)

    return {
        **pii_flags,
        **injection_flags,
        "toxicity_score": toxicity_result["score"],
        "toxicity_method": toxicity_result["method"],
        "toxicity_flagged": toxicity_result["flagged"],
        "toxicity_categories": toxicity_result.get("categories", {}),
    }

