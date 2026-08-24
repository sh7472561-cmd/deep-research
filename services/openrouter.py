import os
import time
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"

# "openrouter/free" auto-routes to whichever free model is currently healthy, so it
# doesn't go stale the way a pinned model ID does when a provider retires it.
SYNTHESIS_MODELS = ["openrouter/free"]
FAST_MODELS = ["openrouter/free"]
GROQ_FALLBACK = "llama-3.3-70b-versatile"  # confirmed current/active on Groq, not deprecated

# Backoff settings
RATE_LIMIT_WAIT  = 15   # seconds to wait after a 429 before trying next model
GROQ_RETRY_WAIT  = 30   # seconds to wait before retrying Groq after a 429
INTER_MODEL_WAIT = 2    # seconds between model attempts (avoid burst rate limits)


def _validate_response_content(data: dict, provider: str) -> str:
    """
    Validate that the API response contains non-null content.
    Raises ValueError if content is missing, null, or empty.
    """
    if not data.get("choices"):
        raise ValueError(f"{provider} returned no choices: {data}")
    
    choice = data["choices"][0]
    if not isinstance(choice, dict):
        raise ValueError(f"{provider} returned invalid choice format: {choice}")
    
    message = choice.get("message", {})
    if not isinstance(message, dict):
        raise ValueError(f"{provider} returned invalid message format: {message}")
    
    content = message.get("content")
    if content is None:
        raise ValueError(f"{provider} returned null content. Full response: {data}")
    
    if not isinstance(content, str):
        raise ValueError(f"{provider} returned non-string content: {type(content)}")
    
    if not content.strip():
        raise ValueError(f"{provider} returned empty content")
    
    return content


def _call_openrouter(model: str, messages: list, max_tokens: int, temperature: float) -> str:
    key = os.environ.get("OPEN_ROUTER_API_KEY", "")
    if not key:
        raise ValueError("OPEN_ROUTER_API_KEY not set")
    
    res = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/sh7472561-cmd/deep-research",
        },
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        },
        timeout=120,
    )
    res.raise_for_status()
    
    data = res.json()
    return _validate_response_content(data, "OpenRouter")


def _call_groq(messages: list, max_tokens: int, temperature: float) -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    
    res = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        json={
            "model": GROQ_FALLBACK,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        },
        timeout=90,
    )
    res.raise_for_status()
    
    data = res.json()
    return _validate_response_content(data, "Groq")


def _retry_with_backoff(
    call_func,
    *args,
    max_retries: int = 2,
    retry_wait: int = 15,
    provider_name: str = "Provider",
    **kwargs
) -> str:
    """
    Retry a call function with exponential backoff on rate limits.
    Retries on 429 errors and transient failures.
    """
    last_error: Exception | None = None
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = retry_wait * attempt
                print(f"  {provider_name} retry {attempt}/{max_retries - 1} — waiting {wait_time}s...")
                time.sleep(wait_time)
            
            return call_func(*args, **kwargs)
            
        except requests.HTTPError as e:
            last_error = e
            if e.response is not None and e.response.status_code == 429:
                print(f"  {provider_name} rate limited (429)")
                continue
            else:
                # Non-429 HTTP error — don't retry
                print(f"  {provider_name} HTTP error: {e}")
                break
                
        except (ValueError, KeyError, IndexError) as e:
            last_error = e
            # These are content validation errors — retry might help if it's transient
            print(f"  {provider_name} returned invalid content: {e}")
            continue
            
        except requests.RequestException as e:
            last_error = e
            # Network errors — retry might help
            print(f"  {provider_name} network error: {e}")
            continue
            
        except Exception as e:
            last_error = e
            print(f"  {provider_name} unexpected error: {e}")
            break
    
    raise RuntimeError(f"{provider_name} failed after {max_retries} attempts. Last error: {last_error}")


def complete(messages: list, tier: str = "fast", max_tokens: int = 4000, temperature: float = 0.3) -> str:
    """
    Call the best available model for the given tier.
    Both tiers -> openrouter/free (auto-routed) -> Groq fallback
    """
    models = SYNTHESIS_MODELS if tier == "synthesis" else FAST_MODELS

    # Try OpenRouter models first
    for i, model in enumerate(models):
        if i > 0:
            time.sleep(INTER_MODEL_WAIT)
        
        try:
            print(f"  Trying {model}...")
            return _call_openrouter(model, messages, max_tokens, temperature)
            
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print(f"  {model} rate limited (429) — moving to fallback...")
                break  # Don't waste time on other OpenRouter models if rate limited
            else:
                print(f"  {model} failed: {e}")
                
        except (ValueError, KeyError, IndexError) as e:
            print(f"  {model} returned invalid content: {e}")
            # Continue to next model or fallback
            
        except Exception as e:
            print(f"  {model} failed: {e}")

    # Groq fallback with retry logic
    print(f"  Falling back to Groq ({GROQ_FALLBACK})...")
    
    try:
        return _retry_with_backoff(
            _call_groq,
            messages,
            max_tokens,
            temperature,
            max_retries=2,
            retry_wait=GROQ_RETRY_WAIT,
            provider_name="Groq"
        )
    except RuntimeError as e:
        pass  # Groq also failed, raise comprehensive error below

    # Every model failed — raise rather than return an error string as if it were real
    # content. Returning a string here previously caused the failure message itself to
    # get used downstream as a search query / report body, silently corrupting output.
    raise RuntimeError(
        f"All models failed for tier={tier}. "
        f"OpenRouter models attempted: {models}. "
        f"Groq fallback attempted with model: {GROQ_FALLBACK}. "
        f"Please check API keys, rate limits, and model availability."
    )
