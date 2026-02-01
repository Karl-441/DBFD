import platform
import os
import sys

def check_compatibility():
    report = []
    report.append(f"System: {platform.system()} {platform.release()}")
    report.append(f"Machine: {platform.machine()}")
    report.append(f"Python: {sys.version}")
    
    # 1. Check Libraries
    try:
        import cv2
        report.append("OpenCV: OK")
    except ImportError:
        report.append("OpenCV: Missing (Run 'pip install opencv-python' or 'sudo apt install python3-opencv')")
        
    try:
        import torch
        report.append(f"PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
        if platform.machine() in ['aarch64', 'armv7l']:
            report.append("WARN: On Raspberry Pi, ensure you use PyTorch builds for ARM.")
    except ImportError:
        report.append("PyTorch: Missing (YOLO requires PyTorch)")
        
    try:
        import PyQt6
        report.append("PyQt6: OK")
    except ImportError:
        report.append("PyQt6: Missing (Run 'sudo apt install python3-pyqt6' on Pi if pip fails)")
        
    # 2. Performance Check
    if platform.machine() in ['aarch64', 'armv7l']:
        report.append("\n--- Raspberry Pi Optimization Tips ---")
        report.append("1. Use PNN algorithm (CPU efficient) instead of YOLO.")
        report.append("2. If using YOLO, use 'yolov8n.pt' and export to ONNX/NCNN.")
        report.append("3. Reduce resolution to 640x480 or 320x240.")
        report.append("4. Use 'picamera' module if standard cv2.VideoCapture fails.")
    
    return "\n".join(report)

if __name__ == "__main__":
    print(check_compatibility())
