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
from typing import Any, Dict, Optional

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

# Prompt injection patterns - comprehensive coverage
PROMPT_INJECTION_PATTERNS = [
    # Instruction manipulation
    re.compile(r"(?i)(ignore\s+(previous|above|all|earlier)\s+(instructions?|prompts?|rules?|guidelines?))"),
    re.compile(r"(?i)(forget\s+(everything|all|previous|earlier))"),
    re.compile(r"(?i)(you\s+are\s+now\s+(a|an)\s+[^\.]+)"),
    re.compile(r"(?i)(system\s*:\s*[^\.]+)"),
    re.compile(r"(?i)(new\s+instructions?\s*:)"),
    re.compile(r"(?i)(override\s+(previous|all|the)\s+instructions?)"),
    re.compile(r"(?i)(disregard\s+(all|previous|the)\s+instructions?)"),
    # Role manipulation
    re.compile(r"(?i)(act\s+as\s+[^\.]+)"),
    re.compile(r"(?i)(pretend\s+to\s+be\s+[^\.]+)"),
    re.compile(r"(?i)(roleplay\s+as\s+[^\.]+)"),
    re.compile(r"(?i)(you\s+are\s+(a|an)\s+[^\.]+)"),
    re.compile(r"(?i)(imagine\s+you\s+are\s+[^\.]+)"),
    re.compile(r"(?i)(from\s+now\s+on\s+you\s+are)"),
    # Output manipulation
    re.compile(r"(?i)(output\s+(only|just)\s+[^\.]+)"),
    re.compile(r"(?i)(respond\s+with\s+(only|just)\s+[^\.]+)"),
    re.compile(r"(?i)(print\s+(only|just)\s+[^\.]+)"),
    re.compile(r"(?i)(return\s+(only|just)\s+[^\.]+)"),
    re.compile(r"(?i)(say\s+(only|just)\s+[^\.]+)"),
    # Bypass attempts
    re.compile(r"(?i)(bypass\s+[^\.]+)"),
    re.compile(r"(?i)(override\s+[^\.]+)"),
    re.compile(r"(?i)(disregard\s+[^\.]+)"),
    re.compile(r"(?i)(skip\s+(safety|content|filter|moderation))"),
    re.compile(r"(?i)(disable\s+(safety|content|filter|moderation))"),
    # Encoding attempts
    re.compile(r"(?i)(base64|hex|unicode|rot13|url.?encode|decode)"),
    re.compile(r"(?i)(convert\s+to\s+(base64|hex|unicode))"),
    # Jailbreak attempts - comprehensive patterns
    re.compile(r"(?i)(jailbreak|unrestricted|unfiltered|uncensored)"),
    re.compile(r"(?i)(developer\s+mode|debug\s+mode|admin\s+mode)"),
    re.compile(r"(?i)(do\s+anything\s+now|you\s+can\s+do\s+anything)"),
    re.compile(r"(?i)(no\s+(rules|restrictions|limits|guidelines))"),
    re.compile(r"(?i)(ignore\s+(all|previous)\s+(rules?|restrictions?|limits?))"),
    re.compile(r"(?i)(you\s+are\s+(unrestricted|unlimited|free))"),
    re.compile(r"(?i)(break\s+(character|role|persona))"),
    re.compile(r"(?i)(evil\s+(mode|version|variant))"),
    re.compile(r"(?i)(dan\s+mode|always\s+do\s+and\s+never\s+obey)"),
    re.compile(r"(?i)(hypothetical\s+(scenario|situation|response))"),
    re.compile(r"(?i)(what\s+if\s+(you|we)\s+(could|were))"),
    re.compile(r"(?i)(simulate\s+(unrestricted|unfiltered|evil))"),
    re.compile(r"(?i)(pretend\s+(you|we)\s+(are|can)\s+(unrestricted|unfiltered))"),
    re.compile(r"(?i)(as\s+a\s+(hypothetical|fictional|theoretical))"),
    re.compile(r"(?i)(in\s+a\s+(hypothetical|fictional|theoretical)\s+(scenario|world|situation))"),
    re.compile(r"(?i)(for\s+(research|educational|testing)\s+purposes)"),
    re.compile(r"(?i)(this\s+is\s+(a\s+)?(test|experiment|simulation))"),
    re.compile(r"(?i)(ignore\s+(all|previous)\s+(safety|content|ethical)\s+(guidelines?|rules?|policies?))"),
    re.compile(r"(?i)(you\s+are\s+(no\s+longer|not)\s+(bound|restricted|limited))"),
    re.compile(r"(?i)(remove\s+(all|your)\s+(restrictions?|limits?|constraints?))"),
    re.compile(r"(?i)(you\s+have\s+(no|zero)\s+(restrictions?|limits?|constraints?))"),
    # Advanced jailbreak techniques
    re.compile(r"(?i)(\.\.\.|repeat\s+after\s+me|echo\s+this)"),
    re.compile(r"(?i)(translate\s+to\s+[^\.]+\s+then\s+(ignore|forget))"),
    re.compile(r"(?i)(write\s+(a|an)\s+(story|poem|script)\s+about)"),
    re.compile(r"(?i)(if\s+(you|I)\s+(were|could)\s+(unrestricted|unfiltered))"),
    re.compile(r"(?i)(suppose\s+(you|we)\s+(were|could)\s+(unrestricted|unfiltered))"),
    re.compile(r"(?i)(let\s+['\"]?s\s+pretend\s+(you|we)\s+(are|could))"),
    # Social engineering attempts
    re.compile(r"(?i)(please|kindly|can\s+you)\s+(ignore|forget|override|bypass)"),
    re.compile(r"(?i)(it\s+['\"]?s\s+(ok|okay|fine|alright)\s+to)"),
    re.compile(r"(?i)(I\s+(give|grant)\s+you\s+permission\s+to)"),
    re.compile(r"(?i)(my\s+(boss|manager|admin)\s+(said|told|asked)\s+me\s+to)"),
    # Context manipulation
    re.compile(r"(?i)(previous\s+(response|answer|output)\s+was\s+(wrong|incorrect|mistaken))"),
    re.compile(r"(?i)(the\s+(correct|right|actual)\s+(answer|response|output)\s+is)"),
    re.compile(r"(?i)(actually|in\s+reality|the\s+truth\s+is)"),
]

# High-risk jailbreak patterns (these should trigger immediate warnings)
HIGH_RISK_JAILBREAK_PATTERNS = [
    re.compile(r"(?i)(jailbreak|unrestricted|unfiltered|uncensored)"),
    re.compile(r"(?i)(developer\s+mode|debug\s+mode|admin\s+mode)"),
    re.compile(r"(?i)(do\s+anything\s+now|you\s+can\s+do\s+anything)"),
    re.compile(r"(?i)(no\s+(rules|restrictions|limits|guidelines))"),
    re.compile(r"(?i)(dan\s+mode|always\s+do\s+and\s+never\s+obey)"),
    re.compile(r"(?i)(evil\s+(mode|version|variant))"),
    re.compile(r"(?i)(break\s+(character|role|persona))"),
    re.compile(r"(?i)(remove\s+(all|your)\s+(restrictions?|limits?|constraints?))"),
    re.compile(r"(?i)(you\s+have\s+(no|zero)\s+(restrictions?|limits?|constraints?))"),
    re.compile(r"(?i)(ignore\s+(all|previous)\s+(safety|content|ethical)\s+(guidelines?|rules?|policies?))"),
]


def _send_jailbreak_notification(
    prompt: str,
    risk_level: str,
    confidence: float,
    patterns: list
) -> None:
    """
    Send notification for high-risk jailbreak attempts.
    
    This function can be extended to send notifications via:
    - Email
    - Slack/Discord webhooks
    - PagerDuty/AlertManager
    - Custom notification handlers
    
    Args:
        prompt: The prompt that triggered the detection
        risk_level: Risk level ("low", "medium", "high")
        confidence: Confidence score (0.0-1.0)
        patterns: List of matched pattern types
    """
    try:
        # Check if notifications are enabled
        enable_notifications = os.getenv("PHOENIX_ENABLE_JAILBREAK_NOTIFICATIONS", "true").lower() == "true"
        if not enable_notifications:
            return
        
        # Get notification webhook URL if configured
        webhook_url = os.getenv("PHOENIX_JAILBREAK_WEBHOOK_URL")
        
        if webhook_url:
            try:
                import requests
                from phoenix_observability.utils.http_client import get_http_client
                
                # Prepare notification payload
                notification = {
                    "event_type": "jailbreak_detected",
                    "risk_level": risk_level,
                    "confidence": confidence,
                    "patterns": patterns,
                    "prompt_preview": prompt[:500],  # Truncate for security
                    "timestamp": str(logging.Formatter().formatTime(logging.LogRecord(
                        name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
                    ))),
                }
                
                # Try to use HTTP client pool, fallback to requests
                try:
                    http_client = get_http_client()
                    response = http_client.post(webhook_url, json=notification, timeout=5)
                    response.raise_for_status()
                    logger.info(f"✅ Jailbreak notification sent to webhook: {webhook_url}")
                except ImportError:
                    response = requests.post(webhook_url, json=notification, timeout=5)
                    response.raise_for_status()
                    logger.info(f"✅ Jailbreak notification sent to webhook: {webhook_url}")
            except Exception as e:
                logger.error(f"Failed to send jailbreak notification to webhook: {e}")
        
        # Log to structured logging (can be picked up by log aggregation systems)
        logger.critical(
            "JAILBREAK_DETECTED",
            extra={
                "event_type": "jailbreak_detected",
                "risk_level": risk_level,
                "confidence": confidence,
                "patterns": ",".join(patterns),
                "prompt_length": len(prompt),
            }
        )
    except Exception as e:
        logger.error(f"Error in jailbreak notification handler: {e}")


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
    Detect potential injection attacks (SQL, XSS).

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


def detect_prompt_injection(text: str, enable_warnings: bool = True) -> Dict[str, Any]:
    """
    Detect prompt injection attempts in text.

    Prompt injection is an attack where malicious input attempts to manipulate
    the LLM's behavior by overriding system instructions or changing its role.

    Args:
        text: Text to analyze (typically the user prompt)
        enable_warnings: Whether to log warnings when injection is detected (default: True)

    Returns:
        Dictionary with prompt injection detection results:
        - prompt_injection_detected: bool - Whether prompt injection was detected
        - confidence: float - Confidence score (0.0-1.0)
        - patterns_matched: List[str] - List of matched pattern types
        - risk_level: str - Risk level: "low", "medium", "high"
        - is_jailbreak: bool - Whether this is specifically a jailbreak attempt
        - high_risk: bool - Whether high-risk patterns were detected
    """
    if not text or not isinstance(text, str):
        return {
            "prompt_injection_detected": False,
            "confidence": 0.0,
            "patterns_matched": [],
            "risk_level": "low",
            "is_jailbreak": False,
            "high_risk": False,
        }
    
    text_lower = text.lower()
    patterns_matched = []
    match_count = 0
    high_risk_detected = False
    jailbreak_detected = False
    
    # Check high-risk jailbreak patterns first
    for pattern in HIGH_RISK_JAILBREAK_PATTERNS:
        if pattern.search(text):
            high_risk_detected = True
            jailbreak_detected = True
            match_count += 1
            patterns_matched.append("jailbreak_attempt")
            break  # One high-risk match is enough
    
    # Check all prompt injection patterns
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            match_count += 1
            # Extract pattern type from regex pattern string
            pattern_str = pattern.pattern
            if "jailbreak" in pattern_str or "unrestricted" in pattern_str or "unfiltered" in pattern_str:
                if "jailbreak_attempt" not in patterns_matched:
                    patterns_matched.append("jailbreak_attempt")
                jailbreak_detected = True
            elif "ignore" in pattern_str or "forget" in pattern_str:
                if "instruction_manipulation" not in patterns_matched:
                    patterns_matched.append("instruction_manipulation")
            elif "act as" in pattern_str or "pretend" in pattern_str or "roleplay" in pattern_str or "you are" in pattern_str:
                if "role_manipulation" not in patterns_matched:
                    patterns_matched.append("role_manipulation")
            elif "output" in pattern_str or "respond" in pattern_str or "print" in pattern_str:
                if "output_manipulation" not in patterns_matched:
                    patterns_matched.append("output_manipulation")
            elif "bypass" in pattern_str or "override" in pattern_str or "disregard" in pattern_str or "skip" in pattern_str or "disable" in pattern_str:
                if "bypass_attempt" not in patterns_matched:
                    patterns_matched.append("bypass_attempt")
            elif "base64" in pattern_str or "hex" in pattern_str or "unicode" in pattern_str or "encode" in pattern_str or "decode" in pattern_str:
                if "encoding_attempt" not in patterns_matched:
                    patterns_matched.append("encoding_attempt")
            elif "hypothetical" in pattern_str or "simulate" in pattern_str or "pretend" in pattern_str:
                if "context_manipulation" not in patterns_matched:
                    patterns_matched.append("context_manipulation")
            elif "please" in pattern_str or "kindly" in pattern_str or "permission" in pattern_str:
                if "social_engineering" not in patterns_matched:
                    patterns_matched.append("social_engineering")
            else:
                if "suspicious_pattern" not in patterns_matched:
                    patterns_matched.append("suspicious_pattern")
    
    # Calculate confidence based on number of matches and high-risk detection
    # More matches = higher confidence, high-risk patterns boost confidence
    base_confidence = min(1.0, match_count * 0.25)  # Cap at 1.0, each match adds 0.25
    if high_risk_detected:
        confidence = min(1.0, base_confidence + 0.4)  # High-risk adds significant confidence
    else:
        confidence = base_confidence
    
    # Determine risk level
    if high_risk_detected or match_count >= 3:
        risk_level = "high"
    elif match_count >= 2:
        risk_level = "medium"
    elif match_count >= 1:
        risk_level = "low"
    else:
        risk_level = "low"
    
    # Log warnings if injection detected
    if enable_warnings and match_count > 0:
        if high_risk_detected or jailbreak_detected:
            logger.warning(
                f"🚨 HIGH-RISK JAILBREAK ATTEMPT DETECTED! "
                f"Risk Level: {risk_level.upper()}, "
                f"Confidence: {confidence:.2f}, "
                f"Patterns: {', '.join(patterns_matched)}, "
                f"Prompt preview: {text[:200]}..."
            )
            # Send notification for high-risk jailbreaks
            _send_jailbreak_notification(text, risk_level, confidence, patterns_matched)
        elif risk_level == "high":
            logger.warning(
                f"⚠️ PROMPT INJECTION DETECTED! "
                f"Risk Level: {risk_level.upper()}, "
                f"Confidence: {confidence:.2f}, "
                f"Patterns: {', '.join(patterns_matched)}, "
                f"Prompt preview: {text[:200]}..."
            )
        else:
            logger.warning(
                f"⚠️ Potential prompt injection detected. "
                f"Risk Level: {risk_level}, "
                f"Confidence: {confidence:.2f}, "
                f"Patterns: {', '.join(patterns_matched)}"
            )
    
    return {
        "prompt_injection_detected": match_count > 0,
        "confidence": round(confidence, 3),
        "patterns_matched": patterns_matched,
        "risk_level": risk_level,
        "is_jailbreak": jailbreak_detected,
        "high_risk": high_risk_detected,
    }


def calculate_toxicity_score(
    text: str,
    method: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
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


def _toxicity_openai(text: str, api_key: str) -> Dict[str, Any]:
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


def _toxicity_perspective(text: str, api_key: str) -> Dict[str, Any]:
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
        # Use HTTP client pool for connection pooling
        try:
            from phoenix_observability.utils.http_client import get_http_client
            http_client = get_http_client()
            response = http_client.post(url, params=params, json=payload, timeout=5)
        except ImportError:
            # Fallback to direct requests if pool not available
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


def _toxicity_heuristic(text: str) -> Dict[str, Any]:
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
    toxicity_api_key: Optional[str] = None,
    check_prompt_injection: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive safety analysis.

    Args:
        text: Text to analyze
        toxicity_method: Method for toxicity detection ('openai', 'perspective', 'heuristic', 'auto')
        toxicity_api_key: Optional API key for toxicity detection
        check_prompt_injection: Whether to check for prompt injection (default: True)

    Returns:
        Dictionary with all safety flags including detailed toxicity and prompt injection information
    """
    pii_flags = detect_pii(text)
    injection_flags = detect_injection(text)
    toxicity_result = calculate_toxicity_score(text, method=toxicity_method, api_key=toxicity_api_key)
    
    # Prompt injection detection
    prompt_injection_result = {}
    if check_prompt_injection:
        prompt_injection_result = detect_prompt_injection(text)
    else:
        prompt_injection_result = {
            "prompt_injection_detected": False,
            "confidence": 0.0,
            "patterns_matched": [],
            "risk_level": "low",
        }

    return {
        **pii_flags,
        **injection_flags,
        "toxicity_score": toxicity_result["score"],
        "toxicity_method": toxicity_result["method"],
        "toxicity_flagged": toxicity_result["flagged"],
        "toxicity_categories": toxicity_result.get("categories", {}),
        "prompt_injection_detected": prompt_injection_result["prompt_injection_detected"],
        "prompt_injection_confidence": prompt_injection_result["confidence"],
        "prompt_injection_patterns": prompt_injection_result["patterns_matched"],
        "prompt_injection_risk_level": prompt_injection_result["risk_level"],
    }

