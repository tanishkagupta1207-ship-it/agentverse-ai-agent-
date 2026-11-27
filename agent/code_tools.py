import ast
from pathlib import Path

try:
    from agent.chunker import SKIP_DIRS, chunk_repo
except ImportError:
    from chunker import SKIP_DIRS, chunk_repo

REPO_PATH = "repos/requests"
_cache = {}


def _is_test(path):
    p = path.lower()
    return p.startswith("tests/") or "/tests/" in p or p.split("/")[-1].startswith("test_")


def _chunks(repo_root):
    if repo_root not in _cache:
        _cache[repo_root] = chunk_repo(repo_root)
    return _cache[repo_root]


def get_symbol(name, repo_root=REPO_PATH, limit=5):
    """Find where a function, method or class is defined (e.g. 'Session.send')."""
    n = name.strip().lower()
    chunks = _chunks(repo_root)
    hits = [c for c in chunks
            if c["symbol"].lower() == n or c["symbol"].lower().endswith("." + n)]
    if not hits:
        hits = [c for c in chunks if n in c["symbol"].lower()]
    hits.sort(key=lambda c: _is_test(c["path"]))
    hits = hits[:limit]
    results = [{"path": c["path"], "symbol": c["symbol"], "start_line": c["start_line"],
                "end_line": c["end_line"], "text": c["text"][:1500]} for c in hits]
    return {
        "found": bool(results),
        "results": results,
        "evidence": [{"type": "code", "path": r["path"], "start": r["start_line"],
                      "end": r["end_line"]} for r in results],
        "note": None if results else f"No symbol matching '{name}' found.",
    }


def get_callers(name, repo_root=REPO_PATH, limit=15):
    """Find call sites of a function or method, with the function they sit in."""
    short = name.strip().split(".")[-1]
    root = Path(repo_root)
    by_path = {}
    for c in _chunks(repo_root):
        by_path.setdefault(c["path"], []).append(c)

    callers = []
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        rel_path = rel.as_posix()
        try:
            source = f.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except SyntaxError:
            continue
        lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            called = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if called != short:
                continue
            owner = next((c for c in by_path.get(rel_path, [])
                          if c["start_line"] <= node.lineno <= c["end_line"]), None)
            if owner is None:   # call sits outside any indexed chunk
                continue
            callers.append({"path": rel_path, "line": node.lineno, "caller": owner["symbol"],
                            "code": lines[node.lineno - 1].strip()[:200]})
    callers.sort(key=lambda c: _is_test(c["path"]))
    callers = callers[:limit]
    return {
        "found": bool(callers),
        "callers": callers,
        "evidence": [{"type": "code", "path": c["path"], "start": c["line"], "end": c["line"]}
                     for c in callers],
        "note": None if callers else f"No callers of '{name}' found.",
    }


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else REPO_PATH
    r = get_symbol("Session.send", root)
    print("get_symbol:", [(x["path"], x["start_line"], x["end_line"]) for x in r["results"]])
    r = get_callers("resolve_redirects", root)
    print("get_callers:", [(x["path"], x["line"], x["caller"]) for x in r["callers"]])