import json
import os
import re
import sys
import time
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
        if e.path is None or e.start is None or e.end is None:
            return "code evidence is missing path/start/end"
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


# ---- stage 2: semantic support ---------------------------------------------

def _default_commit_message(sha):
    from git import Repo
    return Repo(REPO_PATH).commit(sha).message


def _default_issue_text(number):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_layer"))
    from db import init_db
    row = init_db().execute(
        "SELECT title, body FROM issues WHERE number = ?", (number,)
    ).fetchone()
    return f"{row[0]}\n{(row[1] or '')[:800]}" if row else ""


def _evidence_text(e, repo_root, commit_message, issue_text):
    """The actual content behind one piece of evidence."""
    if e.type == "code":
        f = Path(repo_root) / e.path
        lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        body = "\n".join(lines[e.start - 1:e.end])[:3000]
        return f"[code {e.path}:{e.start}-{e.end}]\n{body}"
    if e.type == "commit":
        return f"[commit {e.sha}]\n{commit_message(e.sha)[:1500]}"
    return f"[{e.type} #{e.number}]\n{issue_text(e.number)[:1200]}"


JUDGE_PROMPT = """You are a strict fact checker. Decide whether the EVIDENCE supports the CLAIM.
Use only the evidence below, nothing else.
- supported: the evidence clearly shows the whole claim
- partial: the evidence shows only part of the claim, or the claim goes beyond it
- unsupported: the evidence does not show the claim
Reply with JSON only, no other text: {{"verdict": "supported" | "partial" | "unsupported", "reason": "<one short line>"}}

CLAIM: {claim}

EVIDENCE:
{evidence}"""


def _default_llm(prompt):
    from openai import OpenAI
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL") or None)
    model = os.getenv("LLM_VERIFIER_MODEL") or os.getenv("LLM_MODEL", "gpt-4o-mini")
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content or ""
        except Exception as e:
            if any(k in str(e).lower() for k in ("rate limit", "429", "413")) and attempt < 2:
                time.sleep(30)
                continue
            raise


def _parse_verdict(text):
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        d = json.loads(m.group(0))
        if d.get("verdict") in ("supported", "partial", "unsupported"):
            return d["verdict"], str(d.get("reason", ""))[:200]
    except Exception:
        pass
    return None, "judge reply could not be parsed"


def verify_semantic(answer, repo_root=REPO_PATH, llm_call=None, commit_message=None,
                    issue_text=None):
    """Stage 2. A separate LLM call sees only the claim and its cited evidence.
    partial -> assumption (note kept, evidence kept); unsupported -> assumption
    with evidence cleared; supported -> unchanged. Returns (new_answer, log)."""
    llm_call = llm_call or _default_llm
    commit_message = commit_message or _default_commit_message
    issue_text = issue_text or _default_issue_text

    new_claims, log = [], []
    for c in answer.claims:
        if c.label == "assumption" or not c.evidence:
            new_claims.append(c)
            continue
        try:
            evidence = "\n\n".join(_evidence_text(e, repo_root, commit_message, issue_text)
                                   for e in c.evidence)
            verdict, reason = _parse_verdict(
                llm_call(JUDGE_PROMPT.format(claim=c.text, evidence=evidence)))
        except Exception as e:  # judge failure must not crash or wrongly downgrade
            verdict, reason = None, f"judge error: {type(e).__name__}"

        if verdict is None:
            log.append({"claim_id": c.id, "stage": "semantic", "verdict": "error",
                        "from": c.label, "to": c.label, "reason": reason})
            new_claims.append(c)
        elif verdict == "supported":
            log.append({"claim_id": c.id, "stage": "semantic", "verdict": verdict,
                        "from": c.label, "to": c.label, "reason": reason})
            new_claims.append(c)
        else:
            log.append({"claim_id": c.id, "stage": "semantic", "verdict": verdict,
                        "from": c.label, "to": "assumption", "reason": reason})
            keep = c.evidence if verdict == "partial" else []
            tag = "Partially supported" if verdict == "partial" else "Not supported by cited evidence"
            new_claims.append(Claim(id=c.id, text=c.text, label="assumption",
                                    evidence=keep, note=f"{tag}: {reason}"))
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