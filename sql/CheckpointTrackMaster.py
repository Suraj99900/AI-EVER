from sql.ConnDB import BaseDB
import sqlite3

class CheckpointTrackMaster(BaseDB):
    def __init__(self):
        super().__init__("sql/DB/AI_EVER_DB.db")
    
    def add_checkpoint(self, model_name, checkpoint_dir, epoch, train_loss, val_loss, accuracy):
        try:
            self.cursor.execute("""
                INSERT INTO checkpoint_track_master 
                (model_name, checkpoint_dir, epoch, train_loss, val_loss, accuracy)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (model_name, checkpoint_dir, epoch, train_loss, val_loss, accuracy))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            print(f"[❌ ERROR] Failed to insert checkpoint: {e}")
            return None

    def get_all_checkpoints(self):
        try:
            self.cursor.execute("SELECT * FROM checkpoint_track_master WHERE deleted = 0 order by id desc")
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"[❌ ERROR] Failed to fetch checkpoints: {e}")
            return []
    
    def get_checkpoint_by_id(self, checkpoint_id):
        try:
            self.cursor.execute("SELECT * FROM checkpoint_track_master WHERE id = ? AND deleted = 0", (checkpoint_id,))
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"[❌ ERROR] Failed to fetch checkpoint {checkpoint_id}: {e}")
            return None

    def update_checkpoint(self, checkpoint_id, **kwargs):
        try:
            updates = ", ".join(f"{k} = ?" for k in kwargs)
            values = list(kwargs.values())
            values.append(checkpoint_id)
            self.cursor.execute(f"""
                UPDATE checkpoint_track_master SET {updates} WHERE id = ?
            """, values)
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"[❌ ERROR] Failed to update checkpoint {checkpoint_id}: {e}")
            return False

    def delete_checkpoint(self, checkpoint_id):
        try:
            self.cursor.execute("""
                UPDATE checkpoint_track_master SET deleted = 1 WHERE id = ?
            """, (checkpoint_id,))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"[❌ ERROR] Failed to soft delete checkpoint {checkpoint_id}: {e}")
            return False
