"""
Standalone deep research runner — executed by run.yml workflow.
Triggered by the Cloudflare Worker hourly cron when new research items are found.
"""
from services.deep_research import run_deep_research

if __name__ == "__main__":
    print("Deep research agent starting...")
    completed = run_deep_research(delay_seconds=0)
    if completed:
        print(f"\nCompleted {len(completed)} report(s):")
        for r in completed:
            print(f"  {r['title']}")
            print(f"    {r['url']}")
    else:
        print("No pending research items.")
