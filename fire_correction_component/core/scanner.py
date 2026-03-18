import os
import json
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class Scanner:
    def __init__(self, base_dir, db_manager, file_patterns=["*.jpg", "*.png"], label_ext=".json"):
        self.base_dir = base_dir
        self.db_manager = db_manager
        self.file_patterns = file_patterns
        self.label_ext = label_ext
        self.logger = logging.getLogger(__name__)

    def update_config(self, base_dir, file_patterns=None, label_ext=None):
        self.base_dir = base_dir
        if file_patterns:
            self.file_patterns = file_patterns
        if label_ext:
            self.label_ext = label_ext

    def scan_now(self):
        """Perform a full scan of the base directory for new output folders."""
        if not os.path.exists(self.base_dir):
            self.logger.warning(f"Base directory {self.base_dir} does not exist.")
            return []

        new_files = []
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if any(file.endswith(ext.replace("*", "")) for ext in self.file_patterns):
                    img_path = os.path.join(root, file)
                    if not self.db_manager.is_file_scanned(img_path):
                        # Look for corresponding label file
                        label_path = os.path.splitext(img_path)[0] + self.label_ext
                        if os.path.exists(label_path):
                            data = self._parse_label(label_path, img_path)
                            if data:
                                new_files.append(data)
                                self.db_manager.mark_file_scanned(img_path)
        return new_files

    def _parse_label(self, label_path, img_path):
        """Parse the JSON/XML label file and add initial detections to the DB."""
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Expecting data format:
            # {
            #   "filename": "...",
            #   "detections": [{"box": [x1, y1, x2, y2], "confidence": 0.95}, ...]
            # }
            detections = data.get("detections", [])
            parsed_detections = []
            for idx, det in enumerate(detections):
                box = det.get("box", [0, 0, 0, 0])
                conf = det.get("confidence", 0.0)
                
                # Add to database with initial user_label as None (pending)
                self.db_manager.add_correction(
                    file_path=img_path,
                    box_id=idx,
                    x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                    confidence=conf,
                    user_label="pending",
                    reviewer_name="System"
                )
                parsed_detections.append({
                    "file_path": img_path,
                    "box_id": idx,
                    "box": box,
                    "confidence": conf
                })
            return {"file_path": img_path, "detections": parsed_detections}
        except Exception as e:
            self.logger.error(f"Error parsing label file {label_path}: {e}")
            return None

class DirectoryWatcher(FileSystemEventHandler):
    def __init__(self, scanner, callback):
        self.scanner = scanner
        self.callback = callback

    def on_created(self, event):
        if not event.is_directory:
            # If a new image or label is created, we might want to wait a bit 
            # for both to be present, then scan.
            time.sleep(1) 
            new_data = self.scanner.scan_now()
            if new_data and self.callback:
                self.callback(new_data)
