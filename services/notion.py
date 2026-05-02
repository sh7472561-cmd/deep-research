import os
import datetime
import requests

NOTION_VERSION = "2022-06-28"

INBOX_ITEMS_DB = "32f08bae-3021-8030-9c58-cf3f3a91a6d5"
ARTICLES_DB    = "64a6b2e9-6c27-47f4-91f2-a42420e75c14"


def _headers():
    return {
        "Authorization": f"Bearer {os.environ['NOTION_API_KEY']}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }


def _query_db(db_id, filter_=None, sorts=None, limit=None):
    """Query a Notion database with full pagination."""
    url = f"https://api.notion.com/v1/databases/{db_id}/query"
    body = {"page_size": 100}
    if filter_:
        body["filter"] = filter_
    if sorts:
        body["sorts"] = sorts

    results = []
    cursor = None
    while True:
        if cursor:
            body["start_cursor"] = cursor
        res = requests.post(url, headers=_headers(), json=body, timeout=20).json()
        results.extend(res.get("results", []))
        if limit and len(results) >= limit:
            return results[:limit]
        if res.get("has_more"):
            cursor = res.get("next_cursor")
        else:
            break
    return results


def _prop(row, key):
    """Extract a plain-text value from any Notion property type."""
    prop = row.get("properties", {}).get(key, {})
    ptype = prop.get("type", "")

    if ptype == "title":
        return "".join(i.get("plain_text", "") for i in prop.get("title", []))
    elif ptype == "rich_text":
        return "".join(i.get("plain_text", "") for i in prop.get("rich_text", []))
    elif ptype in ("select", "status"):
        s = prop.get(ptype)
        return s.get("name", "") if s else ""
    elif ptype == "multi_select":
        return ", ".join(s.get("name", "") for s in prop.get("multi_select", []))
    elif ptype == "url":
        return prop.get("url") or ""
    elif ptype == "email":
        return prop.get("email") or ""
    elif ptype == "date":
        d = prop.get("date")
        return d.get("start", "") if d else ""
    elif ptype == "checkbox":
        return "Yes" if prop.get("checkbox") else "No"
    elif ptype == "number":
        n = prop.get("number")
        return str(n) if n is not None else ""
    elif ptype == "relation":
        count = len(prop.get("relation", []))
        return f"({count} linked)" if count else ""
    elif ptype == "formula":
        f = prop.get("formula", {})
        return str(f.get(f.get("type", ""), ""))
    return ""


def _markdown_to_blocks(text: str) -> list:
    """Convert simple markdown to Notion block objects (best-effort)."""
    blocks = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            blocks.append({"object": "block", "type": "heading_2",
                            "heading_2": {"rich_text": [{"type": "text", "text": {"content": stripped[3:][:2000]}}]}})
        elif stripped.startswith("### "):
            blocks.append({"object": "block", "type": "heading_3",
                            "heading_3": {"rich_text": [{"type": "text", "text": {"content": stripped[4:][:2000]}}]}})
        elif stripped.startswith("- ") or stripped.startswith("* "):
            blocks.append({"object": "block", "type": "bulleted_list_item",
                            "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": stripped[2:][:2000]}}]}})
        else:
            for i in range(0, len(stripped), 2000):
                blocks.append({"object": "block", "type": "paragraph",
                                "paragraph": {"rich_text": [{"type": "text", "text": {"content": stripped[i:i+2000]}}]}})
    return blocks


def get_pending_research_items() -> list[dict]:
    """Return Inbox Items with Priority Level = 'In Progress', not archived."""
    rows = _query_db(
        INBOX_ITEMS_DB,
        filter_={
            "and": [
                {"property": "Priority Level", "select": {"equals": "In Progress"}},
                {"property": "Archived", "checkbox": {"equals": False}},
            ]
        },
    )
    return [{"id": row["id"], "name": _prop(row, "Name")} for row in rows if _prop(row, "Name")]


def archive_inbox_item(page_id: str) -> None:
    """Tick the Archived checkbox on an Inbox Item."""
    requests.patch(
        f"https://api.notion.com/v1/pages/{page_id}",
        headers=_headers(),
        json={"properties": {"Archived": {"checkbox": True}}},
        timeout=15,
    ).raise_for_status()


def create_article_page(title: str, report_markdown: str,
                        tags: list | None = None, tldr: str = "") -> str:
    """Create a new page in the Articles database. Returns the Notion page URL."""
    today = datetime.date.today().isoformat()
    properties = {
        "Title": {"title": [{"text": {"content": title[:2000]}}]},
        "Type": {"select": {"name": "AI"}},
        "Date": {"date": {"start": today}},
        "Source": {"rich_text": [{"text": {"content": "Deep Research Agent"}}]},
    }
    if tags:
        properties["Tags"] = {"multi_select": [{"name": t} for t in tags]}
    if tldr:
        properties["Notes"] = {"rich_text": [{"text": {"content": tldr[:2000]}}]}

    blocks = _markdown_to_blocks(report_markdown)

    body = {
        "parent": {"database_id": ARTICLES_DB},
        "properties": properties,
        "children": blocks[:100],
    }

    res = requests.post(
        "https://api.notion.com/v1/pages",
        headers=_headers(),
        json=body,
        timeout=30,
    )
    res.raise_for_status()
    data = res.json()

    remaining = blocks[100:]
    if remaining:
        page_id = data["id"]
        for chunk_start in range(0, len(remaining), 100):
            chunk = remaining[chunk_start:chunk_start + 100]
            requests.patch(
                f"https://api.notion.com/v1/blocks/{page_id}/children",
                headers=_headers(),
                json={"children": chunk},
                timeout=30,
            )

    return data.get("url", "")
