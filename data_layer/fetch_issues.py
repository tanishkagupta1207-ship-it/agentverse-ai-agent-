import os
import requests
from dotenv import load_dotenv
from db import init_db

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
OWNER_REPO = "psf/requests"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
}

conn = init_db()

# sirf wahi issues/PRs layenge jo commits mein mention hue hain
numbers = [r[0] for r in conn.execute("SELECT DISTINCT issue_number FROM commit_refs")]
already = {r[0] for r in conn.execute("SELECT number FROM issues")}

saved = skipped = missing = 0

for n in numbers:
    if n in already:
        skipped += 1  # cache: dobara API call nahi
        continue
    resp = requests.get(
        f"https://api.github.com/repos/{OWNER_REPO}/issues/{n}", headers=HEADERS
    )
    if resp.status_code == 404:
        missing += 1
        continue
    if resp.status_code != 200:
        print("Error on", n, resp.status_code, resp.text[:200])
        break
    d = resp.json()
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
            1 if "pull_request" in d else 0,
        ),
    )
    conn.commit()
    saved += 1

print(f"Saved {saved}, skipped (already cached) {skipped}, not found {missing}.")
