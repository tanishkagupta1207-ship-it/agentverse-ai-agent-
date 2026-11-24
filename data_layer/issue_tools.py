import json
from db import init_db


def get_issue_or_pr(number):
    """Ek issue ya PR ki details. A ka agent isko tool ki tarah call karega."""
    conn = init_db()
    row = conn.execute(
        "SELECT number, title, body, state, labels, created_at, updated_at, is_pr "
        "FROM issues WHERE number = ?",
        (number,),
    ).fetchone()

    if row is None:
        return {
            "found": False,
            "note": f"#{number} not found in the local index.",
            "evidence": [],
        }

    kind = "pr" if row[7] else "issue"
    return {
        "found": True,
        "number": row[0],
        "kind": kind,
        "title": row[1],
        "body": (row[2] or "")[:1500],  # bahut lamba body agent ko mat bhejo
        "state": row[3],
        "labels": [l for l in (row[4] or "").split(",") if l],
        "created_at": row[5],
        "updated_at": row[6],
        "evidence": [{"type": kind, "number": row[0]}],
        "note": None,
    }


def list_issues(state="open", labels=None, include_prs=False, limit=20):
    """Issues ki list. state: open / closed / all. labels: ['bug', ...]"""
    conn = init_db()
    query = (
        "SELECT number, title, state, labels, created_at, updated_at, is_pr, length(body) "
        "FROM issues WHERE 1=1"
    )
    params = []

    if state in ("open", "closed"):
        query += " AND state = ?"
        params.append(state)
    if not include_prs:
        query += " AND is_pr = 0"

    query += " ORDER BY updated_at DESC"
    rows = conn.execute(query, params).fetchall()

    wanted = {l.lower() for l in (labels or [])}
    results = []
    for r in rows:
        issue_labels = [l for l in (r[3] or "").split(",") if l]
        if wanted and not wanted.issubset({l.lower() for l in issue_labels}):
            continue
        results.append({
            "number": r[0],
            "title": r[1],
            "state": r[2],
            "labels": issue_labels,
            "created_at": r[4],
            "updated_at": r[5],
            "is_pr": bool(r[6]),
            "body_length": r[7] or 0,
        })
        if len(results) >= limit:
            break

    return {"count": len(results), "issues": results}


if __name__ == "__main__":
    print("--- get_issue_or_pr(7272) ---")
    print(json.dumps(get_issue_or_pr(7272), indent=2)[:600])

    print("\n--- get_issue_or_pr(99999999) (should be not found) ---")
    print(get_issue_or_pr(99999999))

    print("\n--- list_issues(state='all') ---")
    r = list_issues(state="all", include_prs=True, limit=5)
    print("count:", r["count"])
    for i in r["issues"]:
        kind = "PR" if i["is_pr"] else "Issue"
        print(f"- {kind} #{i['number']} [{i['state']}] {i['title']}")
        