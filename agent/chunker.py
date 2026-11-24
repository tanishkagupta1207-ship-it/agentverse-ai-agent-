import ast
import hashlib
from pathlib import Path

SKIP_DIRS = {".git", "venv", ".venv", "env", "__pycache__", "node_modules",
             "build", "dist", ".tox", ".eggs", "site-packages"}


def _hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _make_chunk(rel_path, symbol, start, end, lines):
    text = "\n".join(lines[start - 1:end])
    h = _hash(text)
    return {
        "id": f"{rel_path}::{symbol}::{h}",   # plan: path::symbol::content_hash
        "path": rel_path,
        "symbol": symbol,
        "start_line": start,
        "end_line": end,
        "content_hash": h,
        "text": text,
    }


def _start_line(node):
    decorators = [d.lineno for d in getattr(node, "decorator_list", [])]
    return min([node.lineno] + decorators)


def chunk_file(repo_root, file_path):
    """One chunk per top-level function, per method, and per class header."""
    repo_root, file_path = Path(repo_root), Path(file_path)
    rel_path = file_path.relative_to(repo_root).as_posix()
    source = file_path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    chunks = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(_make_chunk(rel_path, node.name,
                                      _start_line(node), node.end_lineno, lines))
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            # class header = from 'class' line up to the first method
            header_end = (_start_line(methods[0]) - 1) if methods else node.end_lineno
            chunks.append(_make_chunk(rel_path, node.name,
                                      _start_line(node), header_end, lines))
            for m in methods:
                chunks.append(_make_chunk(rel_path, f"{node.name}.{m.name}",
                                          _start_line(m), m.end_lineno, lines))
    return chunks


def chunk_repo(repo_root):
    repo_root = Path(repo_root)
    chunks = []
    for p in sorted(repo_root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in p.relative_to(repo_root).parts):
            continue
        chunks.extend(chunk_file(repo_root, p))
    return chunks


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "repos/requests"
    chunks = chunk_repo(root)
    print(f"{len(chunks)} chunks from {root}")
    for c in chunks[:5]:
        print(f"- {c['path']}:{c['start_line']}-{c['end_line']}  {c['symbol']}  ({c['id']})")