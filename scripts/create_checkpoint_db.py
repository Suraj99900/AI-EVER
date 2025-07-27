# File: create_checkpoint_db.py

import sqlite3

# Set path where your AI EVER project is located
db_path = "../model/checkpoints/checkpoints.db"  # Change path as needed

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create checkpoints table
cursor.execute("""
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    checkpoint_dir TEXT NOT NULL,
    epoch INTEGER,
    train_loss REAL,
    val_loss REAL,
    accuracy REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print(f"SQLite DB created at {db_path}")
