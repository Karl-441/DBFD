import os
import cv2
import numpy as np
import glob
import pickle
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

def main():
    dataset_path = r"d:\Github\DBFD\dataset"
    train_imgs = glob.glob(os.path.join(dataset_path, "images", "train", "*.jpg"))
    
    X = []
    y = []
    
    count_fire = 0
    count_nofire = 0
    target_per_class = 100 # Reduced to ensure execution speed, can be increased
    
    print(f"Found {len(train_imgs)} training images.")
    
    for i, img_path in enumerate(train_imgs):
        if count_fire >= target_per_class and count_nofire >= target_per_class:
            break
            
        basename = os.path.basename(img_path)
        label_name = os.path.splitext(basename)[0] + ".txt"
        label_path = os.path.join(dataset_path, "labels", "train", label_name)
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Fire Samples
        has_fire = False
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            has_fire = True
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
                            # Check if crop center is inside box (approx)
                            if (b[0] - b[2]/2 < rcx < b[0] + b[2]/2) and \
                               (b[1] - b[3]/2 < rcy < b[1] + b[3]/2):
                                overlap = True
                                break
                        
                        if not overlap:
                            # Found a negative crop
                            roi_neg = img[ry:ry+ch, rx:rx+cw]
                            # Use full mask for negative sample to characterize the texture
                            mask_neg = np.ones((ch, cw), dtype=np.uint8) * 255
                            feats = extract_features(roi_neg, mask_neg)
                            X.append(feats)
                            y.append(-1)
                            count_nofire += 1
                            break

        # Non-Fire Samples from images without labels
        elif count_nofire < target_per_class:
                mask = preprocess_image(img)
                if np.count_nonzero(mask) > 10:
                    # False positive candidate
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
                        # Use full mask for random background crop
                        mask = np.ones((roi.shape[0], roi.shape[1]), dtype=np.uint8)*255
                        feats = extract_features(roi, mask)
                        X.append(feats)
                        y.append(-1)
                        count_nofire += 1
                        
        if i % 20 == 0:
            print(f"Processed {i} images. Fire: {count_fire}, Non-fire: {count_nofire}")

    print(f"Collected {len(X)} samples. Fire: {count_fire}, Non-fire: {count_nofire}")
    
    if len(X) == 0:
        print("No samples collected. Cannot train.")
        return

    # Train PNN
    pnn = PNN()
    pnn.fit(X, y)
    pnn.optimize_ecm()
    
    # Save
    with open(r'd:\Github\DBFD\model_pnn.pkl', 'wb') as f:
        pickle.dump(pnn, f)
    print("Model saved to d:\\Github\\DBFD\\model_pnn.pkl")

if __name__ == "__main__":
    main()
