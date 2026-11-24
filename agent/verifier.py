import sys
from pathlib import Path

try:
    from agent.claims import AnswerResult, Claim
except ImportError:
    from claims import AnswerResult, Claim

REPO_PATH = "repos/requests"


# ---- default lookups (replaced by fakes in tests) -------------------------

def _default_index_ranges(path):
    """(start, end) of every indexed chunk for this file."""
    try:
        from agent.indexer import get_collection
    except ImportError:
        from indexer import get_collection
    data = get_collection().get(where={"path": path}, include=["metadatas"])
    return [(m["start_line"], m["end_line"]) for m in data["metadatas"]]


def _default_commit_exists(sha):
    from git import Repo
    try:
        Repo(REPO_PATH).commit(sha)
        return True
    except Exception:
        return False


def _default_issue_kind(number):
    """'issue', 'pr' or None, from B's SQLite index."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_layer"))
    from db import init_db
    row = init_db().execute(
        "SELECT is_pr FROM issues WHERE number = ?", (number,)
    ).fetchone()
    if row is None:
        return None
    return "pr" if row[0] else "issue"


# ---- stage 1: existence ----------------------------------------------------

def _check_evidence(e, repo_root, index_ranges, commit_exists, issue_kind):
    """Return an error string, or None if the evidence exists."""
    if e.type == "code":
        f = Path(repo_root) / e.path
        if not f.is_file():
            return f"file not found: {e.path}"
        if e.start < 1 or e.end < e.start:
            return f"invalid line range {e.start}-{e.end} in {e.path}"
        n_lines = len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
        if e.end > n_lines:
            return f"{e.path} has only {n_lines} lines (cited {e.start}-{e.end})"
        if not any(s <= e.end and e.start <= en for s, en in index_ranges(e.path)):
            return f"{e.path}:{e.start}-{e.end} is not in the index"
    elif e.type == "commit":
        if not commit_exists(e.sha):
            return f"commit {e.sha} not found"
    else:  # issue / pr
        kind = issue_kind(e.number)
        if kind is None:
            return f"#{e.number} not found in the index"
        if kind != e.type:
            return f"#{e.number} is listed as {kind}, expected {e.type}"
    return None


def verify_existence(answer, repo_root=REPO_PATH, index_ranges=None,
                     commit_exists=None, issue_kind=None):
    """Stage 1. Claims whose evidence doesn't exist become 'assumption'.
    Returns (new_answer, log). The log feeds the evaluation metrics."""
    index_ranges = index_ranges or _default_index_ranges
    commit_exists = commit_exists or _default_commit_exists
    issue_kind = issue_kind or _default_issue_kind

    new_claims, log = [], []
    for c in answer.claims:
        errors = [err for e in c.evidence
                  if (err := _check_evidence(e, repo_root, index_ranges,
                                             commit_exists, issue_kind))]
        if errors:
            reason = "; ".join(errors)
            log.append({"claim_id": c.id, "stage": "existence",
                        "from": c.label, "to": "assumption", "reason": reason})
            new_claims.append(Claim(id=c.id, text=c.text, label="assumption",
                                    evidence=[],
                                    note=f"Cited evidence not found: {reason}"))
        else:
            new_claims.append(c)
    return AnswerResult(claims=new_claims, not_found=answer.not_found,
                        note=answer.note), log


if __name__ == "__main__":
    import json
    demo = AnswerResult.model_validate({"claims": [
        {"id": "c1", "text": "Redirects are resolved in sessions.py.", "label": "verified",
         "evidence": [{"type": "code", "path": "src/requests/sessions.py",
                       "start": 134, "end": 152}]},
        {"id": "c2", "text": "Made-up file.", "label": "verified",
         "evidence": [{"type": "code", "path": "src/requests/nope.py",
                       "start": 1, "end": 10}]},
        {"id": "c3", "text": "Made-up issue.", "label": "history_backed",
         "evidence": [{"type": "issue", "number": 999999999}]},
    ]})
    out, log = verify_existence(demo)
    for c in out.claims:
        print(c.id, c.label, "|", c.note)
    print(json.dumps(log, indent=2))