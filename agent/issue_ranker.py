import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from agent.chunker import SKIP_DIRS
except ImportError:
    from chunker import SKIP_DIRS

REPO_PATH = "repos/requests"
DB_PATH = "oracle.db"
LEVELS = ("beginner", "intermediate", "experienced")

# Heuristic weights per level (ours, to be tuned on the test repos). Each sums to 1.
# Beginners are steered by easiness; experienced users mostly by "fit" (harder issues).
WEIGHTS_BY_LEVEL = {
    "beginner":     {"clarity": 0.25, "scope": 0.20, "files": 0.15, "tests": 0.10, "activity": 0.10, "fit": 0.20},
    "intermediate": {"clarity": 0.25, "scope": 0.10, "files": 0.10, "tests": 0.10, "activity": 0.10, "fit": 0.35},
    "experienced":  {"clarity": 0.20, "scope": 0.05, "files": 0.05, "tests": 0.05, "activity": 0.10, "fit": 0.55},
}
EXCLUDE_LABELS = ("question", "needs info", "discussion")


def _repo_index(repo_root):
    """basename -> path for source files, plus all test code as one lowercase string."""
    root, files, tests_text = Path(repo_root), {}, []
    for f in root.rglob("*.py"):
        rel = f.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        p = rel.as_posix()
        if p.startswith("tests/") or "/tests/" in p or rel.name.startswith("test_"):
            tests_text.append(f.read_text(encoding="utf-8", errors="ignore").lower())
            files.setdefault("__test__" + rel.name, p)
        else:
            files.setdefault(rel.name, p)
    return files, "\n".join(tests_text)


def _mentioned_files(text, files):
    names = {Path(n).name for n in re.findall(r"[\w./-]+\.py", text)}
    return sorted({files[n] for n in names if n in files})


def _has_tests(path, files, tests_text):
    stem = Path(path).stem
    if ("__test__test_" + stem + ".py") in files:
        return True
    return bool(re.search(rf"(from|import)\s+[\w.]*\b{re.escape(stem)}\b", tests_text))


def _clarity(title, body):
    n = len(body)
    if n < 50:
        length = 0.0
    elif n < 300:
        length = 0.3 + 0.7 * (n - 50) / 250
    elif n <= 2000:
        length = 1.0
    else:
        length = max(0.4, 1 - (n - 2000) / 8000)
    detail = 1.0 if ("```" in body or "Traceback" in body
                     or re.search(r"^\s{4}\S", body, re.M)) else 0.0
    keys = ("expected", "actual", "reproduce", "version", "python")
    repro = min(1.0, sum(k in body.lower() for k in keys) / 3)
    score = 0.5 * length + 0.25 * detail + 0.25 * repro
    return score, f"{n} chars, {'has' if detail else 'no'} code/trace, {int(repro * 3)}/3 repro hints"


def _scope(body, labels):
    words = len(body.split())
    s = 1 - min(words / 600, 1) * 0.7
    s -= 0.1 * min(body.count("- [ ]") + body.count("* [ ]"), 3)
    low = [l.lower() for l in labels]
    if any("feature" in l for l in low):
        s -= 0.25
    if any("doc" in l for l in low):
        s += 0.2
    if any("good first" in l for l in low):
        s += 0.2
    return max(0.0, min(1.0, s)), f"{words} words" + (", feature request" if any("feature" in l for l in low) else "")


def _files_score(k):
    return {0: 0.5, 1: 1.0, 2: 0.7, 3: 0.45, 4: 0.45}.get(k, 0.1)


def _activity(updated_at, now):
    try:
        days = (now - datetime.fromisoformat(updated_at.replace("Z", "+00:00"))).days
    except Exception:
        return 0.3, "unknown date"
    s = 1.0 if days <= 30 else 0.7 if days <= 90 else 0.4 if days <= 365 else 0.1
    return s, f"updated {days} days ago"


def _fit(level, ease, labels):
    if level == "beginner":
        s = ease
        if any("good first" in l.lower() for l in labels):
            s = min(1.0, s + 0.3)
    elif level == "intermediate":
        s = max(0.0, 1 - abs(ease - 0.6) * 2)
    else:
        s = 1 - ease
    return s, f"estimated ease {int(ease * 100)}/100 for a {level}"


def rank_issues(level="beginner", n=5, db_path=DB_PATH, repo_root=REPO_PATH, now=None):
    """Top n open issues for an experience level, each with a visible score breakdown."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    now = now or datetime.now(timezone.utc)
    weights = WEIGHTS_BY_LEVEL[level]
    files, tests_text = _repo_index(repo_root)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT number, title, body, labels, updated_at FROM issues "
        "WHERE state = 'open' AND is_pr = 0").fetchall()
    conn.close()

    ranked = []
    for number, title, body, labels_s, updated in rows:
        body, labels = body or "", [l for l in (labels_s or "").split(",") if l]
        if any(x in l.lower() for l in labels for x in EXCLUDE_LABELS):
            continue
        mentioned = _mentioned_files(f"{title}\n{body}", files)
        clarity, c_note = _clarity(title, body)
        scope, s_note = _scope(body, labels)
        f_score = _files_score(len(mentioned))
        if mentioned:
            tests = sum(_has_tests(p, files, tests_text) for p in mentioned) / len(mentioned)
            t_note = f"{sum(_has_tests(p, files, tests_text) for p in mentioned)}/{len(mentioned)} files have tests"
        else:
            tests, t_note = 0.5, "no files mentioned"
        act, a_note = _activity(updated, now)
        fit, fit_note = _fit(level, (scope + f_score) / 2, labels)

        parts = {"clarity": (clarity, c_note), "scope": (scope, s_note),
                 "files": (f_score, f"{len(mentioned)} files mentioned"
                           + (f": {', '.join(mentioned[:3])}" if mentioned else "")),
                 "tests": (tests, t_note), "activity": (act, a_note), "fit": (fit, fit_note)}
        total = sum(weights[k] * v[0] for k, v in parts.items()) * 100
        ranked.append({
            "number": number, "title": title, "labels": labels,
            "total": round(total, 1), "updated_at": updated,
            "breakdown": {k: {"score": round(v[0] * 100), "weight": weights[k], "note": v[1]}
                          for k, v in parts.items()},
            "evidence": [{"type": "issue", "number": number}],
        })
    ranked.sort(key=lambda r: (r["total"], r["updated_at"] or ""), reverse=True)
    return ranked[:n]


if __name__ == "__main__":
    level = sys.argv[1] if len(sys.argv) > 1 else "beginner"
    for r in rank_issues(level):
        print(f"\n#{r['number']}  score {r['total']}  {r['title'][:70]}")
        print("   " + " | ".join(f"{k} {v['score']}" for k, v in r["breakdown"].items()))
        print("   labels:", ", ".join(r["labels"]) or "-")