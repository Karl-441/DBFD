import sys
import os
import shutil
from ultralytics import YOLO
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

def import_and_train():
    app = QApplication(sys.argv)
    
    # 1. Select Dataset Folder (Must contain data.yaml or standard structure)
    dataset_path = QFileDialog.getExistingDirectory(None, "Select Dataset Root (Standard YOLO Format)")
    if not dataset_path:
        return

    # Check for data.yaml
    yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(yaml_path):
        # Try to auto-generate if structure exists
        if os.path.exists(os.path.join(dataset_path, "images", "train")):
            yaml_content = f"""
path: {dataset_path.replace(os.sep, '/')}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['fire']
"""
            yaml_path = os.path.join(dataset_path, "data.yaml")
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)
            print(f"Generated data.yaml at {yaml_path}")
        else:
            QMessageBox.critical(None, "Error", "Invalid dataset structure. Missing 'images/train' or 'data.yaml'.")
            return

    # 2. Configure Output
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "output", "models", "yolo_auto_train")
    
    # 3. Load Model
    print("Loading YOLOv8n model...")
    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to load model: {e}")
        return

    # 4. Train
    print(f"Starting training on {dataset_path}...")
    
    # Check for GPU
    import torch
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        results = model.train(
            data=yaml_path,
            epochs=50,
            imgsz=640,
            project=output_dir,
            name="exp",
            exist_ok=True,
            device=device
        )
        QMessageBox.information(None, "Success", f"Training completed.\nResults saved to {output_dir}")
        
        # Copy best model to main models directory
        best_model = os.path.join(output_dir, "exp", "weights", "best.pt")
        if os.path.exists(best_model):
            models_dir = os.path.join(project_root, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"yolo_trained_{timestamp}.pt"
            dest_path = os.path.join(models_dir, new_name)
            
            shutil.copy2(best_model, dest_path)
            print(f"Copied model to {dest_path}")
            
            # Also update a 'latest.pt' link/copy for convenience
            latest_path = os.path.join(models_dir, "yolo_latest.pt")
            shutil.copy2(best_model, latest_path)
            
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Training failed: {e}")

if __name__ == "__main__":
    import_and_train()
