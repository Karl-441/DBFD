import os
import sys
from ultralytics import YOLO
import argparse

def export_to_ncnn(model_path, callback=None):
    """
    将 YOLO .pt 模型导出为 NCNN 格式。
    callback: 接收进度消息的回调函数
    """
    def log(msg):
        print(msg)
        if callback:
            callback(msg)

    if not os.path.exists(model_path):
        log(f"Error: Model file '{model_path}' not found.")
        return False
        
    log(f"Loading model: {model_path}...")
    try:
        model = YOLO(model_path)
        log("Exporting to NCNN format (this may take a few minutes)...")
        # NCNN 导出
        export_path = model.export(format="ncnn", imgsz=640)
        log(f"Export successful! NCNN model saved at: {export_path}")
        return True
    except Exception as e:
        log(f"Export failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO .pt to NCNN for Raspberry Pi")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to .pt model")
    args = parser.parse_args()
    
    # 确保 models 目录存在
    if not os.path.exists("models"):
        os.makedirs("models")
        
    export_to_ncnn(args.model)
