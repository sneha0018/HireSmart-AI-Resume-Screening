import sqlite3

conn = sqlite3.connect("hiresmart.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_name TEXT,
    match_score REAL,
    status TEXT
)
""")

conn.commit()

def save_result(resume_name, match_score, status):
    cursor.execute(
        "INSERT INTO results (resume_name, match_score, status) VALUES (?, ?, ?)",
        (resume_name, match_score, status)
    )
    conn.commit()

def get_all_results():
    cursor.execute("SELECT * FROM results")
    return cursor.fetchall()
