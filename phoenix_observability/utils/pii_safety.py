"""
PII and safety detection utilities.

Simple heuristics for detecting PII and safety issues.
"""

import re
from typing import Dict, Optional

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


def calculate_toxicity_score(text: str) -> float:
    """
    Calculate toxicity score (0.0 = not toxic, 1.0 = highly toxic).

    Note: This is a placeholder implementation. In production, use a proper
    toxicity detection API or model (e.g., Perspective API, OpenAI moderation).

    Args:
        text: Text to analyze

    Returns:
        Toxicity score between 0.0 and 1.0
    """
    # Placeholder: simple keyword-based heuristic
    # In production, replace with actual toxicity detection model/API
    toxic_keywords = [
        r"(?i)\b(hate|violence|abuse|threat|kill|harm|attack)\b",
        r"(?i)\b(racist|sexist|discriminat)\w*\b",
    ]
    
    import re
    toxic_count = sum(1 for pattern in toxic_keywords if re.search(pattern, text))
    
    # Simple scoring: 0.0 if no toxic keywords, up to 0.5 based on count
    # Normalize by text length to avoid false positives on long texts
    if not text:
        return 0.0
    
    score = min(0.5, toxic_count * 0.1)
    return round(score, 3)


def analyze_safety(text: str) -> Dict[str, any]:
    """
    Comprehensive safety analysis.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with all safety flags including toxicity_score
    """
    pii_flags = detect_pii(text)
    injection_flags = detect_injection(text)
    toxicity_score = calculate_toxicity_score(text)

    return {
        **pii_flags,
        **injection_flags,
        "toxicity_score": toxicity_score,
    }

