"""Print a quick summary of the analytics dataset (the SQLite warehouse)."""
import sqlite3

conn = sqlite3.connect("data/assbi.db")
conn.row_factory = sqlite3.Row

print("=== ASSBI dataset (data/assbi.db) ===\n")
for table in ("sessions", "frame_analytics", "crossings", "session_summary"):
    n = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
    print(f"{table:18}: {n:>6} rows")

print("\n--- sample crossing events (fact_crossings) ---")
rows = conn.execute(
    "SELECT session_id, object_class, direction, line_name, frame_index "
    "FROM crossings ORDER BY frame_index LIMIT 8"
).fetchall()
for r in rows:
    print(dict(r))

print("\n--- per-session totals (session_summary) ---")
for r in conn.execute(
    "SELECT session_id, frames_processed, vehicles_in, vehicles_out, "
    "people_in, people_out, peak_crowd FROM session_summary "
    "ORDER BY frames_processed DESC LIMIT 5"
).fetchall():
    print(dict(r))
conn.close()
