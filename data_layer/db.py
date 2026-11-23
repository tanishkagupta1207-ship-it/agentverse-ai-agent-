import sqlite3

def init_db(path="oracle.db"):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS commits(
        sha TEXT PRIMARY KEY,
        author TEXT,
        date TEXT,
        message TEXT
    );
    CREATE TABLE IF NOT EXISTS issues(
        number INTEGER PRIMARY KEY,
        title TEXT,
        body TEXT,
        state TEXT,
        labels TEXT,
        created_at TEXT,
        updated_at TEXT,
        is_pr INTEGER
    );
    CREATE TABLE IF NOT EXISTS commit_refs(
        sha TEXT,
        issue_number INTEGER
    );
    CREATE TABLE IF NOT EXISTS meta(
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """)
    conn.commit()
    return conn

if __name__ == "__main__":
    init_db()
    print("Database ready: oracle.db")