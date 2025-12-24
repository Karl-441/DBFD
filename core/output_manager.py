import os
import datetime
import shutil
import json
import cv2

class OutputManager:
    def __init__(self, base_dir=None):
        if base_dir is None:
            # Default to d:\Github\DBFD\output
            self.base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        else:
            self.base_dir = base_dir
            
        self.ensure_structure()
        
    def ensure_structure(self):
        """Creates the standardized directory structure."""
        subdirs = ["models", "predictions", "logs", "visualizations"]
        for sd in subdirs:
            os.makedirs(os.path.join(self.base_dir, sd), exist_ok=True)
            
    def get_run_dir(self):
        """Creates a timestamped directory for a specific experiment run."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.base_dir, "predictions", f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def save_model(self, model_path, model_name=None):
        """Archives a model file."""
        if model_name is None:
            model_name = os.path.basename(model_path)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        dest_dir = os.path.join(self.base_dir, "models", timestamp)
        os.makedirs(dest_dir, exist_ok=True)
        
        dest_path = os.path.join(dest_dir, model_name)
        shutil.copy2(model_path, dest_path)
        return dest_path

    def save_prediction(self, image, detections, metadata=None, filename=None):
        """Saves a prediction image and its metadata."""
        if filename is None:
            filename = f"pred_{datetime.datetime.now().strftime('%H%M%S_%f')}.jpg"
            
        # Determine where to save (daily folder in predictions)
        today = datetime.datetime.now().strftime("%Y%m%d")
        save_dir = os.path.join(self.base_dir, "predictions", today)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save Image
        img_path = os.path.join(save_dir, filename)
        cv2.imwrite(img_path, image)
        
        # Save Metadata (JSON)
        if metadata or detections:
            json_path = img_path.replace(os.path.splitext(filename)[1], ".json")
            data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "detections": detections,
                "metadata": metadata or {}
            }
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
                
        return img_path

    def log_metric(self, metric_name, value):
        """Logs a performance metric."""
        today = datetime.datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(self.base_dir, "logs", f"metrics_{today}.csv")
        
        is_new = not os.path.exists(log_file)
        with open(log_file, 'a') as f:
            if is_new:
                f.write("timestamp,metric,value\n")
            f.write(f"{datetime.datetime.now().isoformat()},{metric_name},{value}\n")

    def clean_old_files(self, days_to_keep=30):
        """Removes files older than X days."""
        cutoff = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
        
        for root, dirs, files in os.walk(self.base_dir):
            for name in files:
                path = os.path.join(root, name)
                mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
                if mtime < cutoff:
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"Error removing {path}: {e}")

    def validate_output(self):
        """Checks if output directories are writable and valid."""
        try:
            test_file = os.path.join(self.base_dir, "logs", ".test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return True
        except Exception as e:
            print(f"Output validation failed: {e}")
            return False
