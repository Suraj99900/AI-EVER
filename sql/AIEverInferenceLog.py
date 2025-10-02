from sql.ConnDB import BaseDB
import sqlite3

class AIEverInferenceLog(BaseDB):
    """
    Handles CRUD operations for the ai_ever_inference_log_req_res table.
    """
    def __init__(self):
        # Adjust path to your DB file if needed
        super().__init__("sql/DB/AI_EVER_DB.db")

    # ➕ Insert a new inference request/response log
    def add_log(self, check_point_id, req_msg, res_msg, related_checkpoint_id=None,
                status=1, deleted=0):
        try:
            self.cursor.execute("""
                INSERT INTO ai_ever_inference_log_req_res
                (check_point_id, req_msg, res_msg, related_checkpoint_id, status, deleted)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (check_point_id, req_msg, res_msg, related_checkpoint_id, status, deleted))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            print(f"[❌ ERROR] Failed to insert inference log: {e}")
            return None

    # 📥 Fetch all active logs
    def get_all_logs(self, include_deleted=False):
        try:
            if include_deleted:
                self.cursor.execute("SELECT * FROM ai_ever_inference_log_req_res ORDER BY id DESC")
            else:
                self.cursor.execute(
                    "SELECT * FROM ai_ever_inference_log_req_res WHERE deleted = 0 ORDER BY id DESC"
                )
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"[❌ ERROR] Failed to fetch inference logs: {e}")
            return []

    # 📥 Fetch a single log by its ID
    def get_log_by_id(self, log_id):
        try:
            self.cursor.execute(
                "SELECT * FROM ai_ever_inference_log_req_res WHERE id = ? AND deleted = 0",
                (log_id,)
            )
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"[❌ ERROR] Failed to fetch inference log {log_id}: {e}")
            return None

    # ✏️ Update fields of a specific log
    def update_log(self, log_id, **kwargs):
        if not kwargs:
            return False
        try:
            updates = ", ".join(f"{k} = ?" for k in kwargs)
            values = list(kwargs.values()) + [log_id]
            self.cursor.execute(
                f"UPDATE ai_ever_inference_log_req_res SET {updates} WHERE id = ?",
                values
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"[❌ ERROR] Failed to update inference log {log_id}: {e}")
            return False

    # 🗑️ Soft-delete a log (mark as deleted)
    def delete_log(self, log_id):
        try:
            self.cursor.execute("""
                UPDATE ai_ever_inference_log_req_res
                SET deleted = 1
                WHERE id = ?
            """, (log_id,))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"[❌ ERROR] Failed to delete inference log {log_id}: {e}")
            return False
        
    def get_logs_by_checkpoint(self, checkpoint_id):
        """
        Returns all past logs (req_msg + res_msg) for a given checkpoint/model ID.
        """
        query = "SELECT req_msg, res_msg FROM ai_ever_inference_log_req_res WHERE check_point_id = ? ORDER BY timestamp ASC"
        self.cursor.execute(query, (checkpoint_id,))
        rows = self.cursor.fetchall()
        return [{"prompt": r[0], "response": r[1]} for r in rows]

