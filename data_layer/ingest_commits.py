import re
from git import Repo
from db import init_db

REPO_PATH = "repos/requests"
MAX_COMMITS = 500  # keep this small for now so it runs fast
REF_PATTERN = re.compile(r"#(\d+)")

conn = init_db()
repo = Repo(REPO_PATH)

commit_count = 0
ref_count = 0

for commit in repo.iter_commits("HEAD", max_count=MAX_COMMITS):
    sha = commit.hexsha
    conn.execute(
        "INSERT OR REPLACE INTO commits VALUES (?, ?, ?, ?)",
        (sha, commit.author.name, commit.committed_datetime.isoformat(), commit.message),
    )
    conn.execute("DELETE FROM commit_refs WHERE sha = ?", (sha,))
    for number in set(REF_PATTERN.findall(commit.message)):
        conn.execute(
            "INSERT INTO commit_refs VALUES (?, ?)", (sha, int(number))
        )
        ref_count += 1
    commit_count += 1

conn.commit()
print(f"Saved {commit_count} commits and {ref_count} issue/PR references.")