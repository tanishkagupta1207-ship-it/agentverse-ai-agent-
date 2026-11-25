import os
import requests
from dotenv import load_dotenv
from db import init_db

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
OWNER_REPO = "psf/requests"
MAX_PAGES = 3  # 3 pages x 100 = at most 300 open issues (avoids rate limits)
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
}

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
        if "pull_request" in d:
            continue  # keep real issues only, skip pull requests
        conn.execute(
            "INSERT OR REPLACE INTO issues VALUES (?,?,?,?,?,?,?,?)",
            (
                d["number"],
                d["title"],
                d.get("body") or "",
                d["state"],
                ",".join(l["name"] for l in d["labels"]),
                d["created_at"],
                d["updated_at"],
                0,
            ),
        )
        saved += 1
    conn.commit()

print(f"Saved {saved} open issues.")