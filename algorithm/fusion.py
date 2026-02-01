import cv2
import numpy as np
import config
from algorithm.preprocess import preprocess_image
from algorithm.features import extract_features

"""
多模态融合检测模块 
    该模块负责整合YOLO和 PNN两种检测算法的结果。
    采用了决策级融合策略，结合了YOLO的形状识别能力和PNN的纹理/颜色特征识别能力。
    融合逻辑旨在降低误报率并提高召回率。
"""

class FusionDetector:
    def __init__(self, pnn_model, yolo_model):
        """
        初始化融合检测器
        参数:
            pnn_model: 已训练的 PNN 模型实例
            yolo_model: 已加载的 YOLO 模型实例 (或 None)
        """
        self.pnn = pnn_model
        self.yolo = yolo_model
        
        # 配置参数
        self.yolo_conf_thresh = 0.4 # YOLO 基础置信度阈值
        self.pnn_iou_thresh = 0.1   # PNN 重叠阈值 (只要有重叠就认为相关)
        
    def detect(self, img):
        """
        执行检测并融合结果
            分别运行 YOLO 和 PNN 流程，然后根据 IOU 和置信度进行结果融合。
        参数:
            img: 输入图像
        返回值:
            list: 检测结果列表，每项包含 (x, y, w, h, confidence, source)
            source 说明: 'YOLO', 'PNN', 'FUSED' 等
        """
        # 1. 运行 YOLO 检测 (Run YOLO)
        yolo_boxes = [] # 格式: (x, y, w, h, conf)
        if self.yolo:
            results = self.yolo(img, verbose=False)
            for r in results:
                for box in r.boxes:
                    # class 0 通常是火焰 (fire)
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        w = x2 - x1
                        h = y2 - y1
                        yolo_boxes.append({'box': [int(x1), int(y1), int(w), int(h)], 'conf': conf})

        # 2. 运行 PNN 检测 (Run PNN)
        pnn_boxes = [] # 格式: (x, y, w, h)
        if self.pnn:
            # PNN 流程: 缩放 -> 预处理 -> 连通域提取 -> 特征提取 -> 分类
            try:
                h_orig, w_orig = img.shape[:2]
                target_w, target_h = config.PNN_TARGET_WIDTH, config.PNN_TARGET_HEIGHT
                
                # 缩放图像以提高速度
                small_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                
                mask = preprocess_image(small_img)
                # 连通组件分析
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                
                sx = w_orig / float(target_w)
                sy = h_orig / float(target_h)
                
                for i in range(1, num_labels):
                    x, y, w, h, area = stats[i]
                    if area < 12: continue # 忽略过小的区域
                    
                    # 创建该组件的 ROI 掩膜
                    component_mask = np.zeros_like(mask)
                    component_mask[labels == i] = 255
                    roi_mask = component_mask[y:y+h, x:x+w]
                    roi = small_img[y:y+h, x:x+w]
                    
                    try:
                        feats = extract_features(roi, roi_mask)
                        pred = self.pnn.predict(feats)[0]
                        if pred == 1: # 1 表示火焰
                            # 还原坐标到原始图像
                            xr = int(x * sx)
                            yr = int(y * sy)
                            wr = int(w * sx)
                            hr = int(h * sy)
                            pnn_boxes.append({'box': [xr, yr, wr, hr], 'conf': 1.0})
                    except:
                        pass
            except Exception as e:
                print(f"PNN Error: {e}")

        # 3. 融合逻辑 (Fusion Logic)
        final_detections = []
        
        # 融合策略:
        # - High Conf YOLO (> 0.6) -> 保留 (视觉特征强)
        # - Mid Conf YOLO (0.2 - 0.6) -> 仅当有 PNN 重叠时保留 (纹理特征确认)
        # - PNN Only -> 作为低置信度结果保留 (可能有 YOLO 漏检的纹理匹配)
        
        # 标记已使用的 PNN 框
        pnn_used = [False] * len(pnn_boxes)
        
        for yb in yolo_boxes:
            box_y = yb['box']
            conf_y = yb['conf']
            
            # 检查与任何 PNN 框的重叠 (IOU)
            has_overlap = False
            for i, pb in enumerate(pnn_boxes):
                box_p = pb['box']
                iou = self.compute_iou(box_y, box_p)
                if iou > 0.05: # 轻微重叠即可
                    has_overlap = True
                    pnn_used[i] = True
            
            if conf_y > 0.6:
                # 强 YOLO 结果 - 直接保留
                src = "YOLO+PNN" if has_overlap else "YOLO"
                final_detections.append((*box_y, conf_y, src))
            elif conf_y > 0.2 and has_overlap:
                # 弱 YOLO 但有纹理验证 - 保留并提升置信度
                final_detections.append((*box_y, conf_y + 0.2, "FUSED_WEAK"))
            else:
                # 弱 YOLO 且无纹理支持 - 丢弃 (减少误报)
                pass
                
        # 添加剩余的 PNN 框 (仅纹理匹配)
        # 这些可能是 YOLO 漏检的小火焰
        for i, pb in enumerate(pnn_boxes):
            if not pnn_used[i]:
                final_detections.append((*pb['box'], 0.5, "PNN_ONLY"))
                
        return final_detections

    def compute_iou(self, boxA, boxB):
        """
        计算两个矩形框的交并比 (Intersection over Union)
        参数:
            boxA, boxB: [x, y, w, h]
            
        返回值:
            float: IOU 值 (0.0 - 1.0)
        """
        # box: x, y, w, h
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou
