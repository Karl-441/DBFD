import os
import shutil
import glob
import json
import datetime

class DatasetManager:
    def __init__(self, root_dir=None):
        if root_dir is None:
            self.root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
        else:
            self.root_dir = root_dir
            
    def get_images(self, split='train'):
        """Returns list of images in a split."""
        path = os.path.join(self.root_dir, "images", split, "*.*")
        return glob.glob(path)

    def save_label(self, img_path, boxes, classes=None):
        """
        Saves labels in YOLO format.
        boxes: list of [x_center, y_center, w, h] normalized
        """
        # Assume corresponding label path
        # images/train/img.jpg -> labels/train/img.txt
        
        dir_name = os.path.dirname(img_path) # .../images/train
        base_name = os.path.basename(img_path)
        name_no_ext = os.path.splitext(base_name)[0]
        
        # Replace 'images' with 'labels' in path
        label_dir = dir_name.replace("images", "labels")
        if not os.path.exists(label_dir):
            os.makedirs(label_dir, exist_ok=True)
            
        label_path = os.path.join(label_dir, name_no_ext + ".txt")
        
        with open(label_path, 'w') as f:
            for i, box in enumerate(boxes):
                cls = 0 
                if classes and i < len(classes):
                    cls = classes[i]
                # box is xc, yc, w, h
                line = f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"
                f.write(line)
        
        self.log_change("label_update", f"Updated labels for {base_name}")

    def merge_datasets(self, source_dir, dest_split='train'):
        """
        Merges another dataset (YOLO format) into this one.
        source_dir must contain 'images' and 'labels' subdirs.
        """
        # Simple copy with rename to avoid collision
        src_images = glob.glob(os.path.join(source_dir, "images", "*.*"))
        
        count = 0
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        dest_img_dir = os.path.join(self.root_dir, "images", dest_split)
        dest_lbl_dir = os.path.join(self.root_dir, "labels", dest_split)
        
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lbl_dir, exist_ok=True)
        
        for img_p in src_images:
            base = os.path.basename(img_p)
            name, ext = os.path.splitext(base)
            
            # Find label
            lbl_p = os.path.join(source_dir, "labels", name + ".txt")
            
            new_name = f"{name}_merged_{timestamp}{ext}"
            new_lbl_name = f"{name}_merged_{timestamp}.txt"
            
            shutil.copy2(img_p, os.path.join(dest_img_dir, new_name))
            if os.path.exists(lbl_p):
                shutil.copy2(lbl_p, os.path.join(dest_lbl_dir, new_lbl_name))
                
            count += 1
            
        self.log_change("merge", f"Merged {count} images from {source_dir}")
        return count

    def log_change(self, action, details):
        log_path = os.path.join(self.root_dir, "dataset_version.json")
        history = []
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    history = json.load(f)
            except: pass
            
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "details": details
        }
        history.append(entry)
        
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=4)
