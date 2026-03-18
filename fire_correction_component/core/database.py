import sqlite3
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="corrections.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Table for corrections
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    box_id INTEGER NOT NULL,
                    x1 REAL,
                    y1 REAL,
                    x2 REAL,
                    y2 REAL,
                    confidence REAL,
                    user_label TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    reviewer_name TEXT
                )
            ''')
            # Table for keeping track of scanned files to avoid duplicates
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scanned_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE,
                    scanned_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def add_correction(self, file_path, box_id, x1, y1, x2, y2, confidence, user_label, reviewer_name):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO corrections (file_path, box_id, x1, y1, x2, y2, confidence, user_label, reviewer_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (file_path, box_id, x1, y1, x2, y2, confidence, user_label, reviewer_name))
            conn.commit()

    def update_correction(self, file_path, box_id, user_label, reviewer_name):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE corrections 
                SET user_label = ?, reviewer_name = ?, timestamp = CURRENT_TIMESTAMP
                WHERE file_path = ? AND box_id = ?
            ''', (user_label, reviewer_name, file_path, box_id))
            conn.commit()

    def get_corrections(self, file_path=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if file_path:
                cursor.execute('SELECT * FROM corrections WHERE file_path = ?', (file_path,))
            else:
                cursor.execute('SELECT * FROM corrections')
            return [dict(row) for row in cursor.fetchall()]

    def is_file_scanned(self, file_path):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM scanned_files WHERE file_path = ?', (file_path,))
            return cursor.fetchone() is not None

    def mark_file_scanned(self, file_path):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('INSERT INTO scanned_files (file_path) VALUES (?)', (file_path,))
                conn.commit()
            except sqlite3.IntegrityError:
                pass

    def delete_correction(self, file_path, box_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM corrections WHERE file_path = ? AND box_id = ?', (file_path, box_id))
            conn.commit()
