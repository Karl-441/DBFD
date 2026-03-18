import os
import cv2
import numpy as np
import config
import logging

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("Ultralytics not installed. YOLO detector will be unavailable.")

logger = logging.getLogger(__name__)

class YoloDetector:
    """
    YOLOv8 火灾检测算法模块
        封装了模型加载、推理、类别判定及坐标处理逻辑。
    """
    def __init__(self, model_path=None):
        """
        初始化 YOLO 检测器
        参数:
            model_path: 模型权重文件路径。如果为 None，则尝试从 config 加载。
        """
        self.model = None
        if YOLO is None:
            return
            
        if model_path:
            self.load_model(model_path)
        else:
            # 尝试使用配置中的默认路径
            default_path = getattr(config, 'YOLO_MODEL_PATH', None)
            if default_path and os.path.exists(default_path):
                self.load_model(default_path)

    def load_model(self, path):
        """
        加载模型文件 (支持 .pt 文件或 NCNN 模型目录)
        """
        if YOLO is None:
            return False
        try:
            # 1. 路径预处理
            if not path or not os.path.exists(path):
                logger.error(f"DeepLearning: model path not found: {path}")
                return False
            
            # 2. 检查加载类型
            is_ncnn = path.endswith('_ncnn_model') or os.path.isdir(path)
            
            # 3. 加载模型 (Ultralytics 会根据路径类型自动选择后端)
            # 明确指定 task='detect' 以消除警告
            self.model = YOLO(path, task='detect')
            
            if is_ncnn:
                logger.info(f"DeepLearning: NCNN hardware-accelerated model loaded: {os.path.basename(path)}")
            else:
                logger.info(f"DeepLearning: Standard PyTorch model loaded: {os.path.basename(path)}")
                
            return True
        except Exception as e:
            logger.error(f"DeepLearning: Error loading model: {e}")
            self.model = None
            return False

    def detect(self, img, conf_thresh=None):
        """
        执行推理。
        """
        if self.model is None:
            return []
        
        if img is None or img.size == 0:
            return []
            
        # 智能设备选择
        device = getattr(config, 'DEVICE', 'cpu')
        # 如果是 NCNN 模型，ultralytics 会忽略 device 参数，内部自动走 Vulkan
        
        conf = conf_thresh if conf_thresh is not None else getattr(config, 'YOLO_CONF_THRESH', 0.45)
        min_area = getattr(config, 'YOLO_MIN_AREA', 50)
        
        try:
            # 1. 执行推理 (NCNN 会在支持的平台上自动使用 Vulkan)
            results = self.model(img, verbose=False, device=device, conf=conf)
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                names = result.names
                
                # --- 核心推理日志 ---
                raw_count = len(result.boxes)
                # 每隔 30 帧打印一次，避免刷屏
                do_log = (getattr(self, 'frame_count', 0) % 30 == 0)
                if do_log and raw_count > 0:
                    logger.debug(f"[DL Raw Debug] Found {raw_count} potential objects.")
                
                for box in result.boxes:
                    cls_idx = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy()
                    cls_name = names.get(cls_idx, f"class_{cls_idx}").lower()
                    
                    # --- 判定逻辑 ---
                    is_fire = False
                    if "fire" in cls_name or "smoke" in cls_name or "flame" in cls_name:
                        is_fire = True
                    elif cls_idx == 0 and "person" not in cls_name:
                        is_fire = True
                    
                    if is_fire and score >= conf:
                        x1, y1, x2, y2 = xyxy
                        ix1, iy1 = int(round(x1)), int(round(y1))
                        iw, ih = int(round(x2 - x1)), int(round(y2 - y1))
                        
                        if (iw * ih) >= min_area:
                            detections.append((ix1, iy1, iw, ih))
                            if do_log:
                                logger.info(f"[DL Match] Fire Confirmed: {cls_name} ({score:.2f})")
                            
            return detections
            
        except Exception as e:
            logger.error(f"DeepLearning inference error: {e}")
            return []
