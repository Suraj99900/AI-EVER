# making connection to the database
import sqlite3
from datetime import datetime
import os

class BaseDB:
    def __init__(self, db_path = "sql/DB/AI_EVER_DB.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()


    def close(self):
        self.conn.commit()
        self.conn.close()