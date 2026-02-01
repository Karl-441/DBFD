import os
import datetime
import shutil
import json
import cv2
import numpy as np

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
            self.base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        else:
            self.base_dir = base_dir
            
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
            os.makedirs(os.path.join(self.base_dir, sd), exist_ok=True)
            
    def get_run_dir(self):
        """
            为每次实验或运行创建一个带有时间戳的独立目录。
        返回值:
            str: 新创建的运行目录路径
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.base_dir, "predictions", f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def save_model(self, model_path, model_name=None):
        """
        归档模型文件
        参数:
            model_path: 源模型文件路径
            model_name: 目标文件名 (可选)
        返回值:
            str: 归档后的文件路径
        """
        if model_name is None:
            model_name = os.path.basename(model_path)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        dest_dir = os.path.join(self.base_dir, "models", timestamp)
        os.makedirs(dest_dir, exist_ok=True)
        
        dest_path = os.path.join(dest_dir, model_name)
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
            str: 图片保存路径
        """
        if filename is None:
            filename = f"pred_{datetime.datetime.now().strftime('%H%M%S_%f')}.jpg"
            
        # 确定保存位置 (按日期分类)
        today = datetime.datetime.now().strftime("%Y%m%d")
        save_dir = os.path.join(self.base_dir, "predictions", today)
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存图片
        img_path = os.path.join(save_dir, filename)
        cv2.imwrite(img_path, image)
        
        # 保存元数据 (JSON)
        if metadata or detections:
            json_path = img_path.replace(os.path.splitext(filename)[1], ".json")
            
            # 辅助函数: 将 numpy 类型转换为 Python 原生类型以便 JSON 序列化
            def convert_numpy(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                return obj

            # 递归转换 detections
            serializable_detections = []
            for d in detections:
                if isinstance(d, (list, tuple)):
                    serializable_detections.append([convert_numpy(x) for x in d])
                else:
                    serializable_detections.append(convert_numpy(d))

            data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "detections": serializable_detections,
                "metadata": metadata or {}
            }
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
                
        return img_path

    def log_metric(self, metric_name, value):
        """
        记录性能指标
        参数:
            metric_name: 指标名称
            value: 指标值
        """
        today = datetime.datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(self.base_dir, "logs", f"metrics_{today}.csv")
        
        is_new = not os.path.exists(log_file)
        with open(log_file, 'a') as f:
            if is_new:
                f.write("timestamp,metric,value\n")
            f.write(f"{datetime.datetime.now().isoformat()},{metric_name},{value}\n")

    def clean_old_files(self, days_to_keep=30):
        """
        删除超过指定天数的旧文件，防止磁盘占满。
        参数:
            days_to_keep: 保留天数
        """
        cutoff = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
        
        for root, dirs, files in os.walk(self.base_dir):
            for name in files:
                path = os.path.join(root, name)
                mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
                if mtime < cutoff:
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"Error removing {path}: {e}")

    def validate_output(self):
        """验证输出目录是否可写"""
        try:
            test_file = os.path.join(self.base_dir, "logs", ".test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return True
        except Exception as e:
            print(f"Output validation failed: {e}")
            return False
