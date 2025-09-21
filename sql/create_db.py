# File: create_checkpoint_db.py

import sqlite3
import os

# Ensure directory exists
db_dir = "sql/DB/"
os.makedirs(db_dir, exist_ok=True)

# Path to SQLite database
db_path = os.path.join(db_dir, "AI_EVER_DB.db")

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Table: checkpoint_track_master
cursor.execute("""
CREATE TABLE IF NOT EXISTS checkpoint_track_master (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    checkpoint_dir TEXT NOT NULL,
    epoch INTEGER,
    train_loss REAL,
    val_loss REAL,
    accuracy REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    status int DEFAULT 1,  -- 0: pending, 1: active
    deleted int DEFAULT 0  -- 0: not deleted, 1: deleted
)
""")

# Table: ai_ever_log
cursor.execute("""
CREATE TABLE IF NOT EXISTS ai_ever_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,                -- E.g., 'training_started', 'checkpoint_saved', 'error'
    message TEXT NOT NULL,
    related_checkpoint_id INTEGER,
    status int DEFAULT 1,  -- 0: pending, 1: active
    deleted int DEFAULT 0,  -- 0: not deleted, 1: deleted
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (related_checkpoint_id) REFERENCES checkpoint_track_master(id)
)
""")

conn.commit()
conn.close()

print(f"✅ SQLite DB created and tables initialized at: {db_path}")
