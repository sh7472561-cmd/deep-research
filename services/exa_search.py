import os
import requests

EXA_URL = "https://api.exa.ai/search"


def search_exa(query: str, num_results: int = 5) -> list[dict]:
    """
    Neural search via Exa. Returns list of {title, url, text, highlights}.
    Falls back to empty list on any error (so the pipeline degrades gracefully).
    """
    key = os.environ.get("EXA_API_KEY", "")
    if not key:
        print("  EXA_API_KEY not set — skipping web search")
        return []

    try:
        res = requests.post(
            EXA_URL,
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json={
                "query": query,
                "numResults": num_results,
                "contents": {
                    "text": {"maxCharacters": 3000},
                    "highlights": {"numSentences": 4, "highlightsPerUrl": 3},
                },
                "useAutoprompt": True,
            },
            timeout=30,
        )
        res.raise_for_status()
        results = []
        for item in res.json().get("results", []):
            results.append({
                "title": item.get("title", "Untitled"),
                "url":   item.get("url", ""),
                "text":  (item.get("text") or "").strip(),
                "highlights": item.get("highlights") or [],
            })
        return results
    except Exception as e:
        print(f"  Exa search error for '{query[:60]}': {e}")
        return []


def multi_search(queries: list[str], results_per_query: int = 5) -> list[dict]:
    """Run multiple Exa queries, deduplicating by URL."""
    seen_urls = set()
    all_results = []
    for q in queries:
        for r in search_exa(q, num_results=results_per_query):
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                all_results.append(r)
    return all_results
