import math
import re
from pathlib import Path

import chromadb

try:
    from agent.chunker import chunk_file, chunk_repo
except ImportError:
    from chunker import chunk_file, chunk_repo

DB_PATH = "chroma_db"
COLLECTION = "code_chunks"

STOPWORDS = {
    "how", "are", "is", "was", "were", "where", "what", "when", "why", "which",
    "who", "does", "do", "did", "the", "a", "an", "in", "of", "to", "and", "or",
    "for", "it", "this", "that", "with", "from", "by", "on", "at", "be", "as",
    "can", "i", "me", "my", "used", "use", "work", "works",
}

_cache = {}  # all chunks, loaded once per collection size


def get_collection(embedding_function=None):
    client = chromadb.PersistentClient(path=DB_PATH)
    kwargs = {"name": COLLECTION}
    if embedding_function is not None:
        kwargs["embedding_function"] = embedding_function
    return client.get_or_create_collection(**kwargs)


def _upsert(col, chunks):
    for i in range(0, len(chunks), 200):
        batch = chunks[i:i + 200]
        col.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{
                "path": c["path"],          # repo-relative, forward slashes
                "symbol": c["symbol"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
            } for c in batch],
        )


def index_repo(repo_root, reset=True, embedding_function=None):
    """Chunk the repo and store every chunk in ChromaDB."""
    client = chromadb.PersistentClient(path=DB_PATH)
    if reset and COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    _cache.clear()
    col = get_collection(embedding_function)
    chunks = chunk_repo(repo_root)
    _upsert(col, chunks)
    return len(chunks)


def delete_file_chunks(path, embedding_function=None):
    """Delete every chunk whose metadata 'path' equals path. Returns how many."""
    path = path.replace("\\", "/")
    col = get_collection(embedding_function)
    ids = col.get(where={"path": path}, include=[])["ids"]
    if ids:
        col.delete(ids=ids)
    _cache.clear()
    return len(ids)


def index_file(repo_root, path, embedding_function=None):
    """Re-index one file without a full reset: drop its old chunks, add fresh ones.
    path is relative to repo_root. If the file no longer exists, its chunks are
    just removed. Returns the number of chunks added."""
    path = path.replace("\\", "/")
    delete_file_chunks(path, embedding_function)
    full = Path(repo_root) / path
    if not full.is_file():
        return 0
    chunks = chunk_file(repo_root, full)
    if chunks:
        _upsert(get_collection(embedding_function), chunks)
    _cache.clear()
    return len(chunks)


def _tokens(query):
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query.lower())
    out = []
    for w in words:
        if w in STOPWORDS or len(w) < 3:
            continue
        for suffix in ("ing", "ed", "es", "s"):   # crude stemming
            if w.endswith(suffix) and len(w) - len(suffix) >= 4:
                w = w[: -len(suffix)]
                break
        out.append(w)
    return out


def _is_test(path):
    p = path.lower()
    return p.startswith("tests/") or "/tests/" in p or p.split("/")[-1].startswith("test_")


def _all_chunks(col):
    n = col.count()
    if _cache.get("n") != n:
        data = col.get(include=["documents", "metadatas"])
        _cache["n"] = n
        _cache["rows"] = list(zip(data["ids"], data["documents"], data["metadatas"]))
    return _cache["rows"]


def _keyword_ranking(col, query):
    toks = _tokens(query)
    rows = _all_chunks(col)
    lows = [doc.lower() for _, doc, _ in rows]
    n = len(rows)
    # rare words matter more: 'request' appears everywhere in the requests repo
    idf = {t: math.log((n + 1) / (1 + sum(t in low for low in lows))) for t in toks}
    scored = []
    for (cid, _, meta), low in zip(rows, lows):
        sym = meta["symbol"].lower()
        s = sum(idf[t] * (3 * (t in sym) + min(low.count(t), 5)) for t in toks)
        if s > 0:
            scored.append((s, cid))
    scored.sort(reverse=True)
    return [cid for _, cid in scored[:50]]


def search_code(query, k=5, embedding_function=None, test_penalty=0.5):
    """Agent tool: hybrid search (embeddings + keywords), tests ranked lower."""
    col = get_collection(embedding_function)
    if col.count() == 0:
        return {"found": False, "results": [], "evidence": [],
                "note": "Index is empty. Run index_repo first."}

    sem = col.query(query_texts=[query], n_results=min(30, col.count()))
    sem_ids = sem["ids"][0]
    sem_dist = dict(zip(sem_ids, sem["distances"][0]))
    kw_ids = _keyword_ranking(col, query)

    # Reciprocal rank fusion of the two rankings
    score = {}
    for rank, cid in enumerate(sem_ids):
        score[cid] = score.get(cid, 0) + 1 / (60 + rank)
    for rank, cid in enumerate(kw_ids):
        score[cid] = score.get(cid, 0) + 1 / (60 + rank)

    rows = {cid: (doc, meta) for cid, doc, meta in _all_chunks(col)}
    for cid in score:
        if _is_test(rows[cid][1]["path"]):
            score[cid] *= test_penalty

    top = sorted(score, key=score.get, reverse=True)[:k]
    results = []
    for cid in top:
        doc, meta = rows[cid]
        results.append({
            "path": meta["path"],
            "symbol": meta["symbol"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "distance": round(sem_dist[cid], 4) if cid in sem_dist else None,
            "score": round(score[cid], 5),
            "text": doc[:1500],
        })
    evidence = [{"type": "code", "path": r["path"],
                 "start": r["start_line"], "end": r["end_line"]} for r in results]
    return {
        "found": bool(results),
        "results": results,
        "evidence": evidence,
        "note": None if results else "No matching code found.",
    }


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "repos/requests"
    print(f"Indexed {index_repo(root)} chunks")
    for q in ["how are cookies handled in a session",
              "where are redirects followed",
              "how does authentication work"]:
        print(f"\nQ: {q}")
        for r in search_code(q, k=5)["results"]:
            print(f"- {r['path']}:{r['start_line']}-{r['end_line']}  {r['symbol']}")