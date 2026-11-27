import os
import sys

import requests
from dotenv import load_dotenv
from db import init_db

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
OWNER_REPO = "psf/requests"
HEADERS = {"Accept": "application/vnd.github+json"}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

MAX_PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 5   # 100 issues per page

conn = init_db()
saved = 0

for page in range(1, MAX_PAGES + 1):
    resp = requests.get(
        f"https://api.github.com/repos/{OWNER_REPO}/issues",
        headers=HEADERS,
        params={"state": "open", "per_page": 100, "page": page},
    )
    if resp.status_code != 200:
        print("Error", resp.status_code, resp.text[:200])
        break
    items = resp.json()
    if not items:
        break
    for d in items:
        if "pull_request" in d:      # this endpoint also returns PRs; skip them
            continue
        conn.execute(
            "INSERT OR REPLACE INTO issues VALUES (?,?,?,?,?,?,?,?)",
            (d["number"], d["title"], d.get("body") or "", d["state"],
             ",".join(l["name"] for l in d["labels"]),
             d["created_at"], d["updated_at"], 0),
        )
        saved += 1
    conn.commit()

print(f"Saved {saved} open issues.")