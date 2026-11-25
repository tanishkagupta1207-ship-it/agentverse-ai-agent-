import re
from git import Repo
from db import init_db

REPO_PATH = "repos/requests"
REF_PATTERN = re.compile(r"#(\d+)")


def trace(path, start, end):
    conn = init_db()
    repo = Repo(REPO_PATH)

    # All commits that touched these lines (newest first)
    out = repo.git.log(f"-L{start},{end}:{path}", "-s", "--format=%H")
    shas = list(dict.fromkeys(l.strip() for l in out.splitlines() if len(l.strip()) == 40))

    timeline = []
    for sha in shas:
        c = repo.commit(sha)
        refs = sorted({int(n) for n in REF_PATTERN.findall(c.message)})
        linked = []
        for n in refs:
            row = conn.execute(
                "SELECT number, title, is_pr FROM issues WHERE number = ?", (n,)
            ).fetchone()
            if row:
                linked.append({"number": row[0], "title": row[1], "is_pr": bool(row[2])})
        timeline.append({
            "sha": sha[:7],
            "date": c.committed_datetime.date().isoformat(),
            "author": c.author.name,
            "message": c.message.strip().splitlines()[0],
            "linked": linked,
        })

    # If no commit links to an issue or PR, there is no explanation available
    has_reason = any(t["linked"] for t in timeline)
    return {
        "path": path,
        "lines": [start, end],
        "timeline": timeline,
        "has_explanatory_history": has_reason,
    }


def git_history(path, start_line, end_line):
    """Tool for the agent: explains the history behind a range of lines."""
    r = trace(path, start_line, end_line)
    evidence = []
    for t in r["timeline"]:
        evidence.append({"type": "commit", "sha": t["sha"]})
        for i in t["linked"]:
            evidence.append({"type": "pr" if i["is_pr"] else "issue", "number": i["number"]})
    found = r["has_explanatory_history"]
    return {
        "timeline": r["timeline"],
        "evidence": evidence,
        "found": found,
        "note": None if found else "No explanatory history found for these lines.",
    }


if __name__ == "__main__":
    result = git_history("src/requests/sessions.py", 100, 130)
    print(f"Found {len(result['timeline'])} commits")
    for t in result["timeline"]:
        print(f"- {t['date']} {t['sha']} {t['message']}")
        for i in t["linked"]:
            kind = "PR" if i["is_pr"] else "Issue"
            print(f"    {kind} #{i['number']}: {i['title']}")
    print("found:", result["found"])
    if result["note"]:
        print(result["note"])