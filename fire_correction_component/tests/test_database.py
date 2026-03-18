import pytest
import os
import sqlite3
from core.database import DatabaseManager

@pytest.fixture
def db_manager(tmp_path):
    db_path = tmp_path / "test_corrections.sqlite"
    return DatabaseManager(str(db_path))

def test_db_init(db_manager):
    assert os.path.exists(db_manager.db_path)
    with sqlite3.connect(db_manager.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='corrections'")
        assert cursor.fetchone() is not None
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scanned_files'")
        assert cursor.fetchone() is not None

def test_add_and_get_correction(db_manager):
    db_manager.add_correction("img1.jpg", 0, 10, 20, 100, 200, 0.95, "pending", "Admin")
    corrections = db_manager.get_corrections("img1.jpg")
    assert len(corrections) == 1
    assert corrections[0]['file_path'] == "img1.jpg"
    assert corrections[0]['box_id'] == 0
    assert corrections[0]['user_label'] == "pending"

def test_update_correction(db_manager):
    db_manager.add_correction("img1.jpg", 0, 10, 20, 100, 200, 0.95, "pending", "Admin")
    db_manager.update_correction("img1.jpg", 0, "fire", "Operator_01")
    corrections = db_manager.get_corrections("img1.jpg")
    assert corrections[0]['user_label'] == "fire"
    assert corrections[0]['reviewer_name'] == "Operator_01"

def test_scanned_files(db_manager):
    assert not db_manager.is_file_scanned("img1.jpg")
    db_manager.mark_file_scanned("img1.jpg")
    assert db_manager.is_file_scanned("img1.jpg")
