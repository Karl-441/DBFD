from ultralytics import YOLO
import cv2
import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.output_manager import OutputManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="Path to image")
    
    # Default to the best model if it exists, else base model
    # Note: Ultralytics saves runs in 'runs/detect/train/weights/best.pt' by default relative to CWD
    default_model = 'yolov8n.pt'
    
    # Check common training paths
    # First check relative to CWD, then absolute path
    possible_paths = [
        r'runs/detect/train/weights/best.pt',
        r'd:/Github/DBFD/runs/detect/train/weights/best.pt',
        r'd:/Github/DBFD/output/models/yolo_auto_train/exp/weights/best.pt'
    ]
    
    for p in possible_paths:
        if os.path.exists(p):
            default_model = p
            break
        
    parser.add_argument('--model', type=str, default=default_model, help="Path to model .pt file")
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return

    print(f"Predicting on {args.image}...")
    results = model(args.image)
    
    output_manager = OutputManager()
    
    # Visualize
    for i, r in enumerate(results):
        im_array = r.plot()  # plot a BGR numpy array of predictions
        
        # Save using OutputManager
        output_path = output_manager.save_prediction(im_array, [], filename=f"yolo_result_{os.path.basename(args.image)}")
        print(f"Result saved to {output_path}")

if __name__ == "__main__":
    main()
