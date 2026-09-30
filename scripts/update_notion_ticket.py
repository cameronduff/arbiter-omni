"""
Helper utility to update ticket status on Notion board.
Usage:
  python scripts/update_notion_ticket.py <page_id> <status>
"""
import os
import sys
import json
import urllib.request

def update_status(page_id: str, status: str):
    token = os.environ.get("NOTION_API_TOKEN")
    if not token:
        print("NOTION_API_TOKEN not found.")
        return False
    
    url = f"https://api.notion.com/v1/pages/{page_id}"
    payload = {
        "properties": {
            "Status": {"select": {"name": status}}
        }
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        },
        method="PATCH"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"✅ Ticket {page_id} status updated to: {status}")
            return True
    except Exception as e:
        print(f"❌ Failed to update ticket {page_id}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python update_notion_ticket.py <page_id> <status>")
        sys.exit(1)
    update_status(sys.argv[1], sys.argv[2])
