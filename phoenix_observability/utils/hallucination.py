"""
Hallucination detection utility.

Uses LLM-as-a-judge method with fallback to keyword-overlap heuristic.
Supports multiple LLM providers: OpenAI, Google Gemini, Anthropic Claude.
Auto-detects available API keys and uses the first available provider.
Based on instructions.md requirements.
"""

import json
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Try loading OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try loading Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.debug("google.generativeai not available. Install with: pip install google-generativeai")

# Try loading Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Cache for API keys and judge configuration
import threading
_api_key_cache: dict = {}
_judge_config_cache: dict = {}
_api_cache_lock = threading.Lock()

def _get_cached_api_key(key_name: str, alt_key_name: Optional[str] = None) -> Optional[str]:
    """
    Get API key from environment with caching and validation.
    
    Args:
        key_name: Primary environment variable name.
        alt_key_name: Alternative environment variable name (optional).
        
    Returns:
        API key value or None.
    """
    global _api_key_cache
    
    cache_key = f"{key_name}|{alt_key_name or ''}"
    
    # Check cache first
    if cache_key in _api_key_cache:
        return _api_key_cache[cache_key]
    
    # Read from environment
    key = os.getenv(key_name)
    if not key and alt_key_name:
        key = os.getenv(alt_key_name)
    
    result = key.strip() if key and key.strip() else None
    
    # Validate API key format if present
    if result:
        from phoenix_observability.utils.security import validate_api_key_format
        
        # Determine key type from key name
        key_type = "openai" if "OPENAI" in key_name.upper() else \
                   "anthropic" if "ANTHROPIC" in key_name.upper() else \
                   "gemini" if "GEMINI" in key_name.upper() or "GOOGLE" in key_name.upper() else "openai"
        
        if not validate_api_key_format(result, key_type):
            logger.warning(
                f"API key format validation failed for {key_name}. "
                f"Key may be invalid. Expected format for {key_type} provider."
            )
    
    # Cache the result
    with _api_cache_lock:
        if cache_key not in _api_key_cache:
            _api_key_cache[cache_key] = result
        else:
            result = _api_key_cache[cache_key]
    
    return result


def _get_cached_judge_config(key_name: str, default: str = "") -> str:
    """
    Get judge configuration from environment with caching.
    
    Args:
        key_name: Environment variable name.
        default: Default value if not found.
        
    Returns:
        Configuration value.
    """
    global _judge_config_cache
    
    # Check cache first
    if key_name in _judge_config_cache:
        return _judge_config_cache[key_name]
    
    # Read from environment
    value = os.getenv(key_name, default)
    if key_name == "JUDGE_PROVIDER":
        value = value.lower()
    
    # Cache the result
    with _api_cache_lock:
        if key_name not in _judge_config_cache:
            _judge_config_cache[key_name] = value
        else:
            value = _judge_config_cache[key_name]
    
    return value


def clear_api_key_cache():
    """
    Clear the API key and judge config cache.
    
    Useful for testing or when API keys change at runtime.
    """
    global _api_key_cache, _judge_config_cache
    with _api_cache_lock:
        _api_key_cache.clear()
        _judge_config_cache.clear()


# Get API keys from environment (read at module load, but will re-read at runtime)
def _get_openai_key():
    return _get_cached_api_key("OPENAI_API_KEY")

def _get_gemini_key():
    return _get_cached_api_key("GEMINI_API_KEY", "GOOGLE_API_KEY")

def _get_anthropic_key():
    return _get_cached_api_key("ANTHROPIC_API_KEY")

# For backward compatibility, keep these but they'll be re-read at runtime
OPENAI_API_KEY = _get_openai_key()
GEMINI_API_KEY = _get_gemini_key()
ANTHROPIC_API_KEY = _get_anthropic_key()

# Judge model configuration (provider-specific defaults) - cached
JUDGE_MODEL = _get_cached_judge_config("JUDGE_MODEL", "")  # Empty = auto-detect
JUDGE_PROVIDER = _get_cached_judge_config("JUDGE_PROVIDER", "").lower()  # "openai", "gemini", "anthropic", or "" for auto

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


def _simple_keyword_score(context: str, answer: str) -> Dict[str, Any]:
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


def _detect_available_provider() -> Tuple[Optional[str], Optional[str]]:
    """
    Detect which LLM provider is available based on API keys.
    
    Returns:
        Tuple of (provider_name, model_name) or (None, None) if none available
        Priority: OpenAI > Gemini > Anthropic
    """
    # Get API keys (cached)
    openai_key = _get_openai_key()
    gemini_key = _get_gemini_key()
    anthropic_key = _get_anthropic_key()
    # Get judge config (cached)
    judge_provider = _get_cached_judge_config("JUDGE_PROVIDER", "").lower()
    judge_model = _get_cached_judge_config("JUDGE_MODEL", "")
    
    # If provider is explicitly set, use it
    if judge_provider:
        if judge_provider == "openai" and OPENAI_AVAILABLE and openai_key:
            model = judge_model or "gpt-4o-mini"
            logger.info(f"Using explicitly set provider: OpenAI with model {model}")
            return ("openai", model)
        elif judge_provider == "gemini" and GEMINI_AVAILABLE and gemini_key:
            model = judge_model or "gemini-2.0-flash-exp"
            logger.info(f"Using explicitly set provider: Gemini with model {model}")
            return ("gemini", model)
        elif judge_provider == "anthropic" and ANTHROPIC_AVAILABLE and anthropic_key:
            model = judge_model or "claude-3-5-sonnet-20241022"
            logger.info(f"Using explicitly set provider: Anthropic with model {model}")
            return ("anthropic", model)
        else:
            logger.warning(f"JUDGE_PROVIDER={judge_provider} specified but not available or missing API key. "
                          f"OpenAI: available={OPENAI_AVAILABLE}, key={'set' if openai_key else 'not set'}, "
                          f"Gemini: available={GEMINI_AVAILABLE}, key={'set' if gemini_key else 'not set'}, "
                          f"Anthropic: available={ANTHROPIC_AVAILABLE}, key={'set' if anthropic_key else 'not set'}")
    
    # Auto-detect: Priority order OpenAI > Gemini > Anthropic
    if OPENAI_AVAILABLE and openai_key:
        model = judge_model or "gpt-4o-mini"
        logger.info(f"Auto-detected provider: OpenAI with model {model}")
        return ("openai", model)
    elif GEMINI_AVAILABLE and gemini_key:
        model = judge_model or "gemini-2.0-flash-exp"
        logger.info(f"Auto-detected provider: Gemini with model {model}")
        return ("gemini", model)
    elif ANTHROPIC_AVAILABLE and anthropic_key:
        model = judge_model or "claude-3-5-sonnet-20241022"
        logger.info(f"Auto-detected provider: Anthropic with model {model}")
        return ("anthropic", model)
    
    # Log why no provider was found with detailed diagnostics
    # SECURITY: Never log API key values, only indicate if set
    logger.error(f"❌ No LLM judge provider available!")
    logger.error(f"   OpenAI: package_installed={OPENAI_AVAILABLE}, api_key={'✅ SET' if openai_key else '❌ NOT SET'}")
    logger.error(f"   Gemini: package_installed={GEMINI_AVAILABLE}, api_key={'✅ SET' if gemini_key else '❌ NOT SET'}")
    logger.error(f"   Anthropic: package_installed={ANTHROPIC_AVAILABLE}, api_key={'✅ SET' if anthropic_key else '❌ NOT SET'}")
    
    # Provide specific fix instructions
    if not OPENAI_AVAILABLE:
        logger.error(f"   → Install OpenAI: pip install openai")
    if not GEMINI_AVAILABLE:
        logger.error(f"   → Install Gemini: pip install google-generativeai")
    if not ANTHROPIC_AVAILABLE:
        logger.error(f"   → Install Anthropic: pip install anthropic")
    
    if not openai_key and not gemini_key and not anthropic_key:
        logger.error(f"   → Set API keys in .env file: OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY")
    
    return (None, None)


def _call_openai_judge(prompt_text: str, system_prompt: str, model: str) -> Optional[str]:
    """Call OpenAI API for hallucination detection."""
    try:
        # Re-read API key at runtime
        openai_key = _get_openai_key()
        if not openai_key:
            logger.error("OPENAI_API_KEY not set but OpenAI judge was called")
            return None
            
        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"OpenAI judge call failed: {e}")
        return None


def _call_gemini_judge(prompt_text: str, system_prompt: str, model: str) -> Optional[str]:
    """Call Google Gemini API for hallucination detection."""
    try:
        # Re-read API key at runtime
        gemini_key = _get_gemini_key()
        
        # Configure Gemini
        if not gemini_key:
            logger.error("GEMINI_API_KEY not set but Gemini judge was called")
            return None
            
        genai.configure(api_key=gemini_key)
        logger.debug(f"Calling Gemini judge with model: {model}")
        
        # Combine system prompt and user prompt for Gemini
        full_prompt = f"{system_prompt}\n\n{prompt_text}"
        
        # Create model instance
        gemini_model = genai.GenerativeModel(model)
        
        # Generate content with JSON response format
        generation_config = genai.types.GenerationConfig(
            temperature=0,
            response_mime_type="application/json"
        )
        
        response = gemini_model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        if not response or not hasattr(response, 'text') or not response.text:
            logger.warning("Gemini judge returned empty response")
            return None
            
        result = response.text.strip()
        logger.debug(f"Gemini judge response received: {result[:100]}...")
        return result
    except Exception as e:
        logger.error(f"Gemini judge call failed: {e}", exc_info=True)
        return None


def _call_anthropic_judge(prompt_text: str, system_prompt: str, model: str) -> Optional[str]:
    """Call Anthropic Claude API for hallucination detection."""
    try:
        # Re-read API key at runtime
        anthropic_key = _get_anthropic_key()
        if not anthropic_key:
            logger.error("ANTHROPIC_API_KEY not set but Anthropic judge was called")
            return None
            
        client = anthropic.Anthropic(api_key=anthropic_key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt_text}
            ]
        )
        # Extract text from response
        if response.content and len(response.content) > 0:
            return response.content[0].text.strip()
        return None
    except Exception as e:
        logger.warning(f"Anthropic judge call failed: {e}")
        return None


def judge_hallucination(
    context: Optional[str] = None,
    answer: str = "",
    user_query: Optional[str] = None,
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    Run LLM judge if available, else fallback heuristic.
    
    Supports multiple LLM providers: OpenAI, Google Gemini, Anthropic Claude.
    Auto-detects available API keys and uses the first available provider.
    
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
    logger.info(f"judge_hallucination called: context={'provided' if context else 'None'}, "
                f"answer_length={len(answer) if answer else 0}, "
                f"user_query={'provided' if user_query else 'None'}, use_llm={use_llm}")
    
    if not answer:
        logger.warning("judge_hallucination: Empty answer provided")
        return {
            "score": 1.0,
            "hallucinates": True,
            "reason": "Empty answer provided"
        }
    
    # Use LLM judge if available
    if use_llm:
        logger.debug("Attempting to detect LLM judge provider...")
        provider, model = _detect_available_provider()
        
        if provider and model:
            logger.info(f"✅ Using {provider} judge (model: {model}) for hallucination detection")
            logger.debug(f"Inputs - Context length: {len(context) if context else 0}, "
                        f"Answer length: {len(answer)}, "
                        f"User query length: {len(user_query) if user_query else 0}")
            try:
                # Prepare prompts based on context availability
                if context:
                    system_prompt = JUDGE_PROMPT_WITH_CONTEXT
                    prompt_text = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
                    logger.debug(f"Judge prompt (WITH context): System prompt length={len(system_prompt)}, "
                               f"User prompt length={len(prompt_text)}, "
                               f"Context preview={context[:200]}..., "
                               f"Answer preview={answer[:200]}...")
                else:
                    system_prompt = JUDGE_PROMPT_NO_CONTEXT
                    if user_query:
                        prompt_text = f"QUERY:\n{user_query}\n\nANSWER:\n{answer}"
                        logger.debug(f"Judge prompt (NO context, WITH query): System prompt length={len(system_prompt)}, "
                                   f"User prompt length={len(prompt_text)}, "
                                   f"Query preview={user_query[:200]}..., "
                                   f"Answer preview={answer[:200]}...")
                    else:
                        prompt_text = f"ANSWER:\n{answer}"
                        logger.debug(f"Judge prompt (NO context, NO query): System prompt length={len(system_prompt)}, "
                                   f"User prompt length={len(prompt_text)}, "
                                   f"Answer preview={answer[:200]}...")
                
                # Call appropriate provider
                content = None
                if provider == "openai":
                    content = _call_openai_judge(prompt_text, system_prompt, model)
                elif provider == "gemini":
                    content = _call_gemini_judge(prompt_text, system_prompt, model)
                elif provider == "anthropic":
                    content = _call_anthropic_judge(prompt_text, system_prompt, model)
                
                if content:
                    logger.debug(f"Judge LLM ({provider}) returned response: {content[:200]}...")
                    try:
                        data = json.loads(content)
                        result = {
                            "score": float(data.get("score", 0.5)),
                            "hallucinates": bool(data.get("hallucinates", False)),
                            "reason": data.get("reason", "No reason provided")
                        }
                        logger.info(f"Hallucination evaluation result: score={result['score']}, "
                                  f"hallucinates={result['hallucinates']}, reason={result['reason'][:100]}")
                        return result
                    except json.JSONDecodeError:
                        # Fallback if model didn't return JSON
                        logger.warning(f"Judge LLM ({provider}) returned non-JSON: {content[:200]}")
                        # Try to extract JSON from response if it's wrapped in markdown
                        import re
                        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                        if json_match:
                            try:
                                data = json.loads(json_match.group())
                                return {
                                    "score": float(data.get("score", 0.5)),
                                    "hallucinates": bool(data.get("hallucinates", False)),
                                    "reason": data.get("reason", "No reason provided")
                                }
                            except json.JSONDecodeError as e:
                                logger.debug(f"Failed to parse JSON from judge response: {e}")
                                pass
                        return {
                            "score": 0.5,
                            "hallucinates": None,
                            "reason": f"Non-JSON response from {provider}: {content[:200]}"
                        }
                else:
                    logger.warning(f"Judge LLM ({provider}) returned empty response")
            except Exception as e:
                logger.error(f"LLM judge ({provider}) failed: {e}", exc_info=True)
                # Fall through to heuristic
        else:
            logger.warning(f"LLM judge requested but no provider available. "
                          f"Provider={provider}, Model={model}")

    # Fallback heuristic (only works with context)
    if context:
        logger.info("Using keyword overlap heuristic (context available)")
        result = _simple_keyword_score(context, answer)
        logger.info(f"Heuristic result: score={result['score']}, hallucinates={result['hallucinates']}")
        return result
    else:
        # Without context, heuristic can't work - return neutral score
        logger.warning("No context available and LLM judge not available - cannot evaluate. "
                      f"use_llm={use_llm}, provider detection returned: provider={provider if 'provider' in locals() else 'not checked'}")
        return {
            "score": 0.5,
            "hallucinates": False,
            "reason": "No context available and LLM judge not available - cannot evaluate"
        }
