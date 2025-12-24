import cv2
import numpy as np
import pickle
import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithm.preprocess import preprocess_image
from algorithm.features import extract_features
from algorithm.pnn import PNN
from core.output_manager import OutputManager

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def detect_fire(img, pnn_model):
    """
    Runs the full detection pipeline:
    1. Preprocess (Color Fusion + Otsu) -> Candidate Regions
    2. Feature Extraction (Color + Texture)
    3. PNN Classification
    """
    # 1. Preprocess (Color + Otsu) -> Mask
    # The mask gives us the "suspected fire area".
    mask = preprocess_image(img)
    
    # Find connected components to classify each region separately
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    detections = []
    
    for i in range(1, num_labels): # Skip background 0
        x, y, w, h, area = stats[i]
        
        # Filter small noise (e.g. < 20 pixels)
        if area < 20:
            continue
            
        # Create a mask for this component
        component_mask = np.zeros_like(mask)
        component_mask[labels == i] = 255
        
        # Extract ROI from image
        roi = img[y:y+h, x:x+w]
        roi_mask = component_mask[y:y+h, x:x+w]
        
        # Extract features
        try:
            feats = extract_features(roi, roi_mask)
            
            # Classify
            pred = pnn_model.predict(feats)[0]
            
            if pred == 1:
                detections.append((x, y, w, h))
        except Exception as e:
            # Skip if feature extraction fails (e.g. too small)
            continue
            
    return detections, mask

def main():
    parser = argparse.ArgumentParser(description="DBFD Fire Detection System")
    parser.add_argument('--image', type=str, help="Path to input image")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model_pnn.pkl")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run train_pnn.py first.")
        return
        
    pnn = load_model(model_path)
    
    # Pick a random test image if not provided
    if not args.image:
        test_dir = os.path.join(base_dir, "dataset", "images", "test")
        if os.path.exists(test_dir):
            import random
            files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
            if files:
                args.image = os.path.join(test_dir, random.choice(files))
                print(f"Using random test image: {args.image}")
    
    if not args.image or not os.path.exists(args.image):
        print("No image provided or found.")
        return
        
    print(f"Processing {args.image}...")
    img = cv2.imread(args.image)
    if img is None:
        print("Failed to read image.")
        return
        
    detections, mask = detect_fire(img, pnn)
    
    print(f"Detected {len(detections)} fire regions.")
    
    # Visualize
    vis = img.copy()
    for (x, y, w, h) in detections:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(vis, "FIRE", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    output_manager = OutputManager()
    output_path = output_manager.save_prediction(vis, detections, filename=f"pnn_result_{os.path.basename(args.image)}")
    print(f"Result saved to {output_path}")
    
    # Also save mask for debugging
    # cv2.imwrite("mask.jpg", mask)

if __name__ == "__main__":
    main()
