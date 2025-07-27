import sqlite3
from sql.ConnDB import BaseDB


class AIEverLog(BaseDB):
    def __init__(self):
        super().__init__("sql/DB/AI_EVER_DB.db")
    
    def add_log(self, event_type, message, related_checkpoint_id=None):
        try:
            self.cursor.execute("""
                INSERT INTO ai_ever_log (event_type, message, related_checkpoint_id)
                VALUES (?, ?, ?)
            """, (event_type, message, related_checkpoint_id))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            print(f"[❌ ERROR] Failed to insert log: {e}")
            return None

    def get_all_logs(self):
        try:
            self.cursor.execute("SELECT * FROM ai_ever_log WHERE deleted = 0")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"[❌ ERROR] Failed to fetch logs: {e}")
            return []

    def delete_log(self, log_id):
        try:
            self.cursor.execute("""
                UPDATE ai_ever_log SET deleted = 1 WHERE id = ?
            """, (log_id,))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"[❌ ERROR] Failed to delete log: {e}")
            return False
        
    def log_event(self, event_type, message, related_checkpoint_id=None):
        try:
            self.cursor.execute("""
                INSERT INTO ai_ever_log (event_type, message, related_checkpoint_id)
                VALUES (?, ?, ?)
            """, (event_type, message, related_checkpoint_id))
            self.conn.commit()
        except Exception as e:
            print("Failed to log event:", e)

    def log_error(self, source, message):
        self.log_event(f"{source}_error", message)