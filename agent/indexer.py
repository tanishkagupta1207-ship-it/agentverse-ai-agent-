import chromadb

try:
    from agent.chunker import chunk_repo
except ImportError:
    from chunker import chunk_repo

DB_PATH = "chroma_db"
COLLECTION = "code_chunks"


def get_collection(embedding_function=None):
    client = chromadb.PersistentClient(path=DB_PATH)
    kwargs = {"name": COLLECTION}
    if embedding_function is not None:
        kwargs["embedding_function"] = embedding_function
    return client.get_or_create_collection(**kwargs)


def index_repo(repo_root, reset=True, embedding_function=None):
    """Chunk the repo and store every chunk in ChromaDB."""
    if reset:
        chromadb.PersistentClient(path=DB_PATH).delete_collection(COLLECTION) \
            if COLLECTION in [c.name for c in chromadb.PersistentClient(path=DB_PATH).list_collections()] \
            else None
    col = get_collection(embedding_function)
    chunks = chunk_repo(repo_root)

    for i in range(0, len(chunks), 200):
        batch = chunks[i:i + 200]
        col.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{
                "path": c["path"],
                "symbol": c["symbol"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
            } for c in batch],
        )
    return len(chunks)


def search_code(query, k=5, embedding_function=None):
    """Tool for the agent. Same return style as B's tools."""
    col = get_collection(embedding_function)
    res = col.query(query_texts=[query], n_results=k)
    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        results.append({
            "path": meta["path"],
            "symbol": meta["symbol"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "distance": round(dist, 4),
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
              "where are redirects followed"]:
        print(f"\nQ: {q}")
        for r in search_code(q, k=3)["results"]:
            print(f"- {r['path']}:{r['start_line']}-{r['end_line']}  {r['symbol']}  (d={r['distance']})")