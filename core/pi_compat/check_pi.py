import platform
import os
import sys
"""
    检查当前运行环境的软硬件配置，包括操作系统信息、Python 版本、
    关键依赖库 (OpenCV, PyTorch, PyQt6) 的安装状态。
    针对树莓派设备提供特定的优化建议。
"""

def check_compatibility():
    """
    返回值:
        str: 格式化的检查报告字符串
    """
    report = []
    # 系统信息
    report.append(f"System: {platform.system()} {platform.release()}")
    report.append(f"Machine: {platform.machine()}")
    report.append(f"Python: {sys.version}")
    
    # 1. 检查依赖库 (Check Libraries)
    # OpenCV
    try:
        import cv2
        report.append("OpenCV: OK")
    except ImportError:
        report.append("OpenCV: Missing (Run 'pip install opencv-python' or 'sudo apt install python3-opencv')")
        
    # PyTorch (用于 YOLO)
    try:
        import torch
        report.append(f"PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
        # 检查是否为 ARM 架构 (树莓派)
        if platform.machine() in ['aarch64', 'armv7l']:
            report.append("WARN: On Raspberry Pi, ensure you use PyTorch builds for ARM. (警告: 请确保使用 ARM 版 PyTorch)")
    except ImportError:
        report.append("PyTorch: Missing (YOLO requires PyTorch) / 缺失 (YOLO 模式需要)")
        
    # PyQt6 (用于 GUI)
    try:
        import PyQt6
        report.append("PyQt6: OK")
    except ImportError:
        report.append("PyQt6: Missing (Run 'sudo apt install python3-pyqt6' on Pi if pip fails)")
        
    # 2. 性能与优化检查 (Performance Check)
    if platform.machine() in ['aarch64', 'armv7l']:
        report.append("\n--- Raspberry Pi Optimization Tips (树莓派优化建议) ---")
        report.append("1. Use PNN algorithm (CPU efficient) instead of YOLO. (优先使用 PNN 算法)")
        report.append("2. If using YOLO, use 'yolov8n.pt' and export to ONNX/NCNN. (若用 YOLO，请使用 nano 模型并量化)")
        report.append("3. Reduce resolution to 640x480 or 320x240. (降低分辨率)")
        report.append("4. Use 'picamera' module if standard cv2.VideoCapture fails. (使用 picamera 模块)")
    
    return "\n".join(report)

if __name__ == "__main__":
    print(check_compatibility())
