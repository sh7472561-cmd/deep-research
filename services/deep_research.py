"""
Deep Research Agent
-------------------
Triggered by main.py at the end of each briefing run.

Pipeline per queued item:
  1. Load research instructions from research_instructions.md
  2. Stage 1 (fast model): expand raw topic into 4-5 targeted Exa search queries
  3. Stage 2 (Exa):        run those queries, deduplicate results
  4. Stage 3 (synthesis):  write full structured report + pick Tags
  5. Notion:               publish as Article, archive Inbox item
  6. Return summary list for inclusion in daily_briefing.md
"""

import os
import re
import time
import datetime

from services.exa_search import multi_search
from services.openrouter import complete
from services.notion import (
    get_pending_research_items,
    archive_inbox_item,
    create_article_page,
)

# Path to the user profile / standing instructions
INSTRUCTIONS_PATH = "research_instructions.md"

# Valid Tags for the Articles DB (keep in sync with Notion schema)
VALID_TAGS = [
    "Geopolitics", "Economics", "Tech", "Defence", "AI", "Climate",
    "Biotech", "Crypto", "SA Politics", "UK Politics", "US Politics",
    "Germany", "China", "VC/Startups", "Science", "Technology",
    "Quantum", "Government Funding", "hypersonics", "CUAS",
]


def _load_instructions() -> str:
    try:
        with open(INSTRUCTIONS_PATH) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "(No research_instructions.md found — using defaults.)"


def _expand_queries(topic: str, instructions: str) -> list[str]:
    """Stage 1: Ask the fast model for 8 targeted Exa search queries."""
    prompt = f"""You are preparing a deep research dossier for a very senior analyst with a PhD and finance background.

USER PROFILE:
{instructions}

RESEARCH TOPIC: "{topic}"

Generate exactly 8 search queries for a neural web search engine (Exa).
These must go well beyond surface-level coverage. Each query targets a distinct, specific angle:

  1. Latest concrete developments, targets, milestones, and timelines (specific numbers/dates)
  2. Specific data, statistics, capacity figures, investment volumes, growth rates
  3. The key institutions, agencies, or individuals driving this (names, roles, policies)
  4. A specific technical or operational challenge — something that is genuinely hard
  5. International competitive response — who is catching up and how close are they?
  6. A non-obvious economic or financial angle (incentives, market structure, capital flows)
  7. Historical precedent or structural parallel — what does this resemble from history?
  8. A serious critical or sceptical analysis from credible experts (not obvious objections)

Queries should be precise enough to surface specialist content — think think-tank reports,
academic papers, government filings, long-form journalism — not Wikipedia or general explainers.

Output ONLY the 8 queries, one per line, no numbering, no preamble."""

    response = complete(
        [{"role": "user", "content": prompt}],
        tier="fast",
        max_tokens=500,
        temperature=0.4,
    )
    queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
    return queries[:8] if queries else [topic]


def _format_sources(results: list[dict]) -> str:
    """Format Exa results into a readable block for the synthesis prompt."""
    if not results:
        return "(No web sources retrieved.)"
    parts = []
    for i, r in enumerate(results, 1):
        highlight = " | ".join(r["highlights"][:3]) if r["highlights"] else ""
        snippet = r["text"][:2000] if r["text"] else highlight
        parts.append(
            f"[{i}] {r['title']}\nURL: {r['url']}\n{snippet}"
        )
    return "\n\n".join(parts)


def _synthesise(topic: str, sources_text: str, instructions: str) -> str:
    """Stage 3: Write the full research report using the synthesis model."""
    today = datetime.date.today().strftime("%d %B %Y")
    valid_tags_str = ", ".join(VALID_TAGS)

    prompt = f"""You are a world-class research analyst — equivalent to a senior fellow at a top think-tank
(Brookings, IISS, PIIE) — writing a classified-style briefing for a senior client.

USER PROFILE:
{instructions}

RESEARCH TOPIC: "{topic}"

WEB SOURCES (cite inline as [N] — never fabricate a citation):
{sources_text}

WRITE THE REPORT in this exact structure. Aim for depth over breadth. At least 700 words.

---

## {topic}
*Research report — {today}*

### Executive Summary
3 bullets. Most surprising or actionable findings — not a table of contents.

### Key Findings
12-16 bullets, most important first. Group by sub-theme with bold sub-headers.
Every bullet must have a specific fact + citation.

### Contrarian View — What Do Smart Sceptics Argue?
4-6 bullets. Smart, non-obvious critique from credible experts.

### Cross-Disciplinary Angle
2-3 paragraphs. Historical parallel, economic concept, or mechanism from another field.

### Timelines & Milestones
Key stated targets, deadlines, milestones from the sources.

### Open Questions
4-5 bullets. Questions experts actually disagree about, where the answer changes the conclusion.

### Further Reading
Full URLs only for sources cited above.

---

After the report body, on a NEW line, output EXACTLY (pick 1-3 tags):
TAGS: [comma-separated tags from: {valid_tags_str}]"""

    return complete(
        [{"role": "user", "content": prompt}],
        tier="synthesis",
        max_tokens=6000,
        temperature=0.3,
    )


def _parse_tags(report: str) -> tuple[str, list[str]]:
    """Strip the TAGS line from the report and return (clean_report, tags)."""
    match = re.search(r"\nTAGS:\s*(.+)$", report, re.MULTILINE)
    if not match:
        return report, []
    tags_raw = match.group(1)
    tags = [t.strip() for t in tags_raw.split(",") if t.strip() in VALID_TAGS]
    clean = report[: match.start()].strip()
    return clean, tags


def _extract_tldr(report: str) -> str:
    """Pull the Executive Summary bullets for the Notes field."""
    match = re.search(
        r"### Executive Summary\n(.+?)(?=\n###|\Z)", report, re.DOTALL
    )
    if not match:
        return ""
    lines = [l.strip() for l in match.group(1).strip().split("\n") if l.strip()]
    return " | ".join(lines[:3])[:2000]


def run_deep_research(delay_seconds: int = 0) -> list[dict]:
    """
    Process all pending 'New Deep Research' inbox items.
    Returns list of dicts: {title, url, tldr}
    """
    items = get_pending_research_items()
    if not items:
        print("  No pending deep research items.")
        return []

    if delay_seconds:
        print(f"  Found {len(items)} research item(s). Starting in {delay_seconds}s...")
        time.sleep(delay_seconds)
    else:
        print(f"  Found {len(items)} research item(s).")

    instructions = _load_instructions()
    completed = []

    for item in items:
        topic = item["name"]
        page_id = item["id"]
        print(f"  Researching: {topic}")

        try:
            print("    -> Expanding queries...")
            queries = _expand_queries(topic, instructions)

            print(f"    -> Searching Exa ({len(queries)} queries)...")
            results = multi_search(queries, results_per_query=6)
            print(f"    -> {len(results)} unique sources found")

            sources_text = _format_sources(results)

            print("    -> Synthesising report...")
            raw_report = _synthesise(topic, sources_text, instructions)

            report, tags = _parse_tags(raw_report)
            tldr = _extract_tldr(report)

            print("    -> Publishing to Notion...")
            article_url = create_article_page(
                title=topic,
                report_markdown=report,
                tags=tags,
                tldr=tldr,
            )

            archive_inbox_item(page_id)
            print(f"    Done: {article_url}")

            completed.append({
                "title": topic,
                "url": article_url,
                "tldr": tldr,
            })

            if len(items) > 1:
                time.sleep(5)

        except Exception as e:
            print(f"    Failed for '{topic}': {e}")

    return completed
