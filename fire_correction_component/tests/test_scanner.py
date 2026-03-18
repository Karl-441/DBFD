import pytest
import os
import json
from unittest.mock import MagicMock
from core.scanner import Scanner

@pytest.fixture
def mock_db():
    db = MagicMock()
    db.is_file_scanned.return_value = False
    return db

@pytest.fixture
def test_dir(tmp_path):
    # Create mock runs folder
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    run_folder = runs_dir / "2024-03-17_120000"
    run_folder.mkdir()
    
    # Create mock image and label
    img_path = run_folder / "fire_01.jpg"
    img_path.write_text("dummy_img")
    
    label_path = run_folder / "fire_01.json"
    label_content = {
        "filename": "fire_01.jpg",
        "detections": [
            {"box": [10, 20, 100, 200], "confidence": 0.95}
        ]
    }
    label_path.write_text(json.dumps(label_content))
    
    return str(runs_dir)

def test_scan_now(test_dir, mock_db):
    scanner = Scanner(test_dir, mock_db)
    new_files = scanner.scan_now()
    
    assert len(new_files) == 1
    assert "fire_01.jpg" in new_files[0]['file_path']
    assert len(new_files[0]['detections']) == 1
    assert new_files[0]['detections'][0]['confidence'] == 0.95
    
    # Ensure DB mark_file_scanned was called
    mock_db.mark_file_scanned.assert_called_once()

def test_parse_label_invalid(test_dir, mock_db):
    scanner = Scanner(test_dir, mock_db)
    # Test with non-existent label
    result = scanner._parse_label("non_existent.json", "some_img.jpg")
    assert result is None
