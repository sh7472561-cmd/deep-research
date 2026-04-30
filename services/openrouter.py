import os
import time
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"

# Model tiers
SYNTHESIS_MODELS = [
    "google/gemini-2.5-pro-exp-03-25:free",
    "google/gemini-2.0-flash-exp:free",
]
FAST_MODELS = [
    "google/gemma-3-27b-it:free",
    "google/gemini-2.0-flash-exp:free",
]
GROQ_FALLBACK = "llama-3.3-70b-versatile"

# Backoff settings
RATE_LIMIT_WAIT  = 15   # seconds to wait after a 429 before trying next model
GROQ_RETRY_WAIT  = 30   # seconds to wait before retrying Groq after a 429
INTER_MODEL_WAIT = 2    # seconds between model attempts (avoid burst rate limits)


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
        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        timeout=120,
    )
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]


def _call_groq(messages: list, max_tokens: int, temperature: float) -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    res = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": GROQ_FALLBACK, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        timeout=90,
    )
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]


def complete(messages: list, tier: str = "fast", max_tokens: int = 4000, temperature: float = 0.3) -> str:
    """
    Call the best available model for the given tier.
    tier="synthesis" -> Gemini 2.5 Pro -> 2.0 Flash -> Groq
    tier="fast"      -> Gemma-3 27b -> 2.0 Flash -> Groq
    """
    models = SYNTHESIS_MODELS if tier == "synthesis" else FAST_MODELS

    for i, model in enumerate(models):
        if i > 0:
            time.sleep(INTER_MODEL_WAIT)
        try:
            print(f"  Trying {model}...")
            return _call_openrouter(model, messages, max_tokens, temperature)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print(f"  {model} rate limited (429) — waiting {RATE_LIMIT_WAIT}s...")
                time.sleep(RATE_LIMIT_WAIT)
            else:
                print(f"  {model} failed: {e}")
        except Exception as e:
            print(f"  {model} failed: {e}")

    # Groq fallback — retry once on 429
    for attempt in range(2):
        try:
            print(f"  Falling back to Groq ({GROQ_FALLBACK}), attempt {attempt + 1}...")
            return _call_groq(messages, max_tokens, temperature)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429 and attempt == 0:
                print(f"  Groq rate limited (429) — waiting {GROQ_RETRY_WAIT}s before retry...")
                time.sleep(GROQ_RETRY_WAIT)
            else:
                return f"All models failed. Last error: {e}"
        except Exception as e:
            return f"All models failed. Last error: {e}"

    return "All models failed."
