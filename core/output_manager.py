import os
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

"""
输出管理器
    统一管理系统的所有文件输出，包括：
    1. 检测到的火焰图片 (Predictions)
    2. 模型文件归档 (Models)
    3. 运行日志 (Logs)
    4. 结果元数据 (Metadata JSON)
    提供自动化的目录结构创建、文件清理和格式化存储功能。
"""

class OutputManager:
    def __init__(self, base_dir=None):
        """
        初始化输出管理器
        参数:
            base_dir: 基础输出目录，默认为项目根目录下的 'output'
        """
        if base_dir is None:
            # 默认为 d:\Github\DBFD\output
            self.base_dir = Path(__file__).resolve().parent.parent / "output"
        else:
            self.base_dir = Path(base_dir)
            
        self.ensure_structure()
        
    def ensure_structure(self):
        """
        创建以下子目录以确保标准目录结构存在
        - models: 存放训练好的模型
        - predictions: 存放检测结果图片和元数据
        - logs: 存放运行日志
        - visualizations: 存放可视化分析图表
        """
        subdirs = ["models", "predictions", "logs", "visualizations"]
        for sd in subdirs:
            (self.base_dir / sd).mkdir(parents=True, exist_ok=True)
            
    def get_run_dir(self):
        """
            为每次实验或运行创建一个带有时间戳的独立目录。
        返回值:
            Path: 新创建的运行目录路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.base_dir / "predictions" / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_model(self, model_path, model_name=None):
        """
        归档模型文件
        参数:
            model_path: 源模型文件路径
            model_name: 目标文件名 (可选)
        返回值:
            Path: 归档后的文件路径
        """
        model_path = Path(model_path)
        if model_name is None:
            model_name = model_path.name
            
        timestamp = datetime.now().strftime("%Y%m%d")
        dest_dir = self.base_dir / "models" / timestamp
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = dest_dir / model_name
        shutil.copy2(model_path, dest_path)
        return dest_path

    def save_prediction(self, image, detections, metadata=None, filename=None):
        """
        保存检测到火焰的图片，并可选地保存包含检测框信息的 JSON 元数据文件。
        参数:
            image: 图像数据 (numpy array)
            detections: 检测框列表
            metadata: 额外的元数据字典 (可选)
            filename: 文件名 (可选)
        返回值:
            Path: 图片保存路径
        """
        if filename is None:
            filename = f"pred_{datetime.now().strftime('%H%M%S_%f')}.jpg"
            
        # 确定保存位置 (按日期分类)
        today = datetime.now().strftime("%Y%m%d")
        save_dir = self.base_dir / "predictions" / today
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存图片
        img_path = save_dir / filename
        # cv2.imwrite 在某些系统上不支持 Path 对象，需转为字符串
        cv2.imwrite(str(img_path), image)
        
        # 保存元数据 (JSON)
        if metadata or detections:
            json_path = img_path.with_suffix('.json')
            
            data = {
                "timestamp": datetime.now().isoformat(),
                "filename": filename,
                "detections": detections, # 假设 detections 已经是可序列化格式
                "metadata": metadata or {}
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                
        return img_path
