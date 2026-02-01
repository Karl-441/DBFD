import sys
import os
import cv2
import numpy as np
import glob
import pickle
import shutil
import datetime
import yaml
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

# Add parent dir to path to import algorithm modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.features import extract_features
from algorithm.pnn import PNN
from algorithm.preprocess import preprocess_image

def get_roi_from_yolo(img, line):
    try:
        parts = line.strip().split()
        cls = int(parts[0])
        x_c, y_c, w, h = map(float, parts[1:])
        
        H, W = img.shape[:2]
        x1 = int((x_c - w/2) * W)
        y1 = int((y_c - h/2) * H)
        w_px = int(w * W)
        h_px = int(h * H)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        w_px = min(W - x1, w_px)
        h_px = min(H - y1, h_px)
        
        if w_px <= 0 or h_px <= 0:
            return None
            
        return img[y1:y1+h_px, x1:x1+w_px]
    except:
        return None

def train_pnn_logic(dataset_path, output_dir):
    # Determine images and labels path
    # We assume standard YOLO structure: root/images/train and root/labels/train
    # If data.yaml exists, we could parse it, but standard YOLO structure is implied by the request.
    
    # Check if data.yaml exists (just to confirm it's a YOLO dataset)
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    # Paths
    train_images_dir = os.path.join(dataset_path, "images", "train")
    train_labels_dir = os.path.join(dataset_path, "labels", "train")
    
    if not os.path.exists(train_images_dir):
        # Try looking into data.yaml if it exists
        if os.path.exists(yaml_path):
             with open(yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
                if 'train' in data_config:
                    # data.yaml paths can be relative or absolute
                    # if relative, it's relative to the yaml file location typically, or the dataset root
                    # ultralytics handles this complexly, but here we try simple join
                    candidate = os.path.join(dataset_path, data_config['train'])
                    if os.path.exists(candidate):
                        train_images_dir = candidate
                        # Assume labels are in parallel folder .../images/train -> .../labels/train
                        # This is a bit tricky if custom paths are used. 
                        # Let's try to deduce labels dir by replacing 'images' with 'labels'
                        train_labels_dir = candidate.replace("images", "labels")

    if not os.path.exists(train_images_dir):
        raise Exception(f"Could not find training images directory at {train_images_dir}")
        
    print(f"Scanning images in {train_images_dir}...")
    train_imgs = glob.glob(os.path.join(train_images_dir, "*.jpg")) + \
                 glob.glob(os.path.join(train_images_dir, "*.png")) + \
                 glob.glob(os.path.join(train_images_dir, "*.jpeg"))
    
    if not train_imgs:
         raise Exception("No images found in training directory.")

    X = []
    y = []
    
    count_fire = 0
    count_nofire = 0
    target_per_class = 500 # Increased limit for better accuracy while keeping PNN fast enough
    
    print(f"Found {len(train_imgs)} training images. Extracting features...")
    
    for i, img_path in enumerate(train_imgs):
        if count_fire >= target_per_class and count_nofire >= target_per_class:
            break
            
        # Find corresponding label
        # Logic: .../images/train/img1.jpg -> .../labels/train/img1.txt
        basename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(basename)[0]
        label_path = os.path.join(train_labels_dir, name_no_ext + ".txt")
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Fire Samples
        has_fire = False
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # 1. Extract Fire (Positive)
            if count_fire < target_per_class:
                for line in lines:
                    if count_fire >= target_per_class: break
                    roi = get_roi_from_yolo(img, line)
                    if roi is not None and roi.size > 0:
                        mask = preprocess_image(roi)
                        if np.count_nonzero(mask) > 10: 
                            feats = extract_features(roi, mask)
                            X.append(feats)
                            y.append(1)
                            count_fire += 1
                            has_fire = True

            # 2. Extract Non-Fire (Negative) from the SAME image
            if count_nofire < target_per_class:
                # Try to find a crop that does not overlap with fire boxes
                H, W = img.shape[:2]
                if H > 60 and W > 60:
                    # Parse all boxes to avoid
                    boxes = []
                    for line in lines:
                        parts = line.strip().split()
                        boxes.append(list(map(float, parts[1:]))) # xc, yc, w, h
                    
                    # Try 5 times to find a non-overlapping crop
                    for _ in range(5):
                        cw, ch = 50, 50
                        rx = np.random.randint(0, W-cw)
                        ry = np.random.randint(0, H-ch)
                        
                        # Check overlap (simple center check)
                        rcx = (rx + cw/2) / W
                        rcy = (ry + ch/2) / H
                        
                        overlap = False
                        for b in boxes:
                            # b: xc, yc, w, h
                            if (b[0] - b[2]/2 < rcx < b[0] + b[2]/2) and \
                               (b[1] - b[3]/2 < rcy < b[1] + b[3]/2):
                                overlap = True
                                break
                        
                        if not overlap:
                            # Found a negative crop
                            roi_neg = img[ry:ry+ch, rx:rx+cw]
                            mask_neg = np.ones((ch, cw), dtype=np.uint8) * 255
                            feats = extract_features(roi_neg, mask_neg)
                            X.append(feats)
                            y.append(-1)
                            count_nofire += 1
                            break

        # Non-Fire Samples from images without labels (Background images)
        elif count_nofire < target_per_class:
            # Whole image might be background, but we need small crops usually?
            # Or use the whole image processing. 
            # In train_pnn.py logic:
            mask = preprocess_image(img)
            if np.count_nonzero(mask) > 10:
                # False positive candidate (something bright/red but no label)
                feats = extract_features(img, mask)
                X.append(feats)
                y.append(-1)
                count_nofire += 1
            else:
                # Random crop as background
                h, w = img.shape[:2]
                if h > 50 and w > 50:
                    rx = np.random.randint(0, w-50)
                    ry = np.random.randint(0, h-50)
                    roi = img[ry:ry+50, rx:rx+50]
                    mask = np.ones((roi.shape[0], roi.shape[1]), dtype=np.uint8)*255
                    feats = extract_features(roi, mask)
                    X.append(feats)
                    y.append(-1)
                    count_nofire += 1
                        
        if i % 50 == 0:
            print(f"Processed {i} images. Fire: {count_fire}, Non-fire: {count_nofire}")

    print(f"Collected {len(X)} samples. Fire: {count_fire}, Non-fire: {count_nofire}")
    
    if len(X) == 0:
        raise Exception("No samples collected. Cannot train.")

    # Train PNN
    print("Training PNN (optimizing sigmas)...")
    pnn = PNN()
    pnn.fit(X, y)
    pnn.optimize_ecm()
    
    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pnn_trained_{timestamp}.pkl"
    save_path = os.path.join(output_dir, filename)
    
    with open(save_path, 'wb') as f:
        pickle.dump(pnn, f)
    print(f"Model saved to {save_path}")
    
    return save_path

def import_and_train():
    app = QApplication(sys.argv)
    
    # 1. Select Dataset Folder
    dataset_path = QFileDialog.getExistingDirectory(None, "Select Dataset Root (Standard YOLO Format)")
    if not dataset_path:
        return

    # 2. Configure Output
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 3. Train
    print(f"Starting PNN training on {dataset_path}...")
    
    try:
        saved_path = train_pnn_logic(dataset_path, models_dir)
        
        # 4. Create/Update Latest Link
        latest_path = os.path.join(models_dir, "pnn_latest.pkl")
        shutil.copy2(saved_path, latest_path)
        
        QMessageBox.information(None, "Success", f"Training completed.\nModel saved to {saved_path}\n(Copied to pnn_latest.pkl)")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        QMessageBox.critical(None, "Error", f"Training failed: {e}")

if __name__ == "__main__":
    import_and_train()
