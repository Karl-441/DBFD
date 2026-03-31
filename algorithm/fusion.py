"""
多模态融合检测模块
    该模块负责整合 YOLO 和 PNN 两种检测算法的结果。
    采用了决策级融合策略，结合了 YOLO 的形状识别能力和 PNN 的纹理/颜色特征识别能力。
    融合逻辑旨在降低误报率并提高召回率。

    公开函数:
        run_pnn_pipeline(img, pnn_model, target_w, target_h, min_area) -> list[dict]
            执行完整 PNN 检测流水线，可被外部模块复用（如 headless_runner）。
"""

import logging
import cv2
import numpy as np
import config
from algorithm.preprocess import preprocess_image
from algorithm.features import extract_features

logger = logging.getLogger(__name__)

# PNN 检测到目标时使用的标记置信度（非真实概率，仅表示"已匹配"）
_PNN_MATCH_CONF = 1.0


def run_pnn_pipeline(
    img: np.ndarray,
    pnn_model,
    target_w: int,
    target_h: int,
    min_area: int = 12,
) -> list:
    """
    执行完整 PNN 检测流水线：缩放 → 预处理 → 连通域分析 → 特征提取 → PNN 分类。

    参数:
        img:        原始输入图像（BGR，任意尺寸）
        pnn_model:  已加载的 PNN 模型实例
        target_w:   PNN 处理时的目标宽度（像素）
        target_h:   PNN 处理时的目标高度（像素）
        min_area:   忽略面积小于此值的连通域（像素²）

    返回:
        list[dict]  每项格式为 {'box': [x, y, w, h], 'conf': float}
                    坐标已还原到原图尺寸。
    """
    if pnn_model is None:
        return []

    try:
        h_orig, w_orig = img.shape[:2]
        small_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        mask = preprocess_image(small_img)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        sx = w_orig / float(target_w)
        sy = h_orig / float(target_h)

        results = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue

            component_mask = np.zeros_like(mask)
            component_mask[labels == i] = 255
            roi_mask = component_mask[y:y+h, x:x+w]
            roi = small_img[y:y+h, x:x+w]

            try:
                feats = extract_features(roi, roi_mask)
                pred = pnn_model.predict(feats)[0]
                if pred == 1:  # 1 表示火焰
                    results.append({
                        'box': [int(x * sx), int(y * sy), int(w * sx), int(h * sy)],
                        'conf': _PNN_MATCH_CONF,
                    })
            except Exception as e:
                logger.debug(f"PNN 特征提取/分类失败（连通域 {i}）: {e}")
                continue

        return results

    except Exception as e:
        logger.error(f"run_pnn_pipeline 发生错误: {e}")
        return []


def _compute_iou(boxA: list, boxB: list) -> float:
    """
    计算两个矩形框的交并比 (Intersection over Union)。

    参数:
        boxA, boxB: [x, y, w, h]

    返回:
        float: IOU 值 (0.0 - 1.0)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    area_a = boxA[2] * boxA[3]
    area_b = boxB[2] * boxB[3]
    return inter_area / float(area_a + area_b - inter_area + 1e-6)


class FusionDetector:
    """决策级融合检测器，整合 YOLO 和 PNN 两路结果。"""

    # IOU 阈值：轻微重叠即视为两路检测相关
    _IOU_OVERLAP_THRESH = 0.05

    def __init__(self, pnn_model, yolo_detector):
        """
        参数:
            pnn_model:      已训练的 PNN 模型实例
            yolo_detector:  YoloDetector 实例（或 None）
        """
        self.pnn = pnn_model
        self.yolo_detector = yolo_detector
        self.yolo_conf_thresh = config.YOLO_CONF_THRESH

    def detect(self, img: np.ndarray) -> list:
        """
        执行检测并融合两路结果。

        参数:
            img: 输入图像（BGR）

        返回:
            list[tuple]  每项格式为 (x, y, w, h, confidence, source)
                         source 取值：'YOLO' | 'YOLO+PNN' | 'FUSED_WEAK' | 'PNN_ONLY'
        """
        yolo_boxes = self._run_yolo(img)
        pnn_boxes = run_pnn_pipeline(
            img, self.pnn,
            config.PNN_TARGET_WIDTH,
            config.PNN_TARGET_HEIGHT,
        )
        return self._merge(yolo_boxes, pnn_boxes)

    def _run_yolo(self, img: np.ndarray) -> list:
        """
        执行 YOLO 推理，返回火焰相关的检测框。

        返回:
            list[dict]  每项格式为 {'box': [x, y, w, h], 'conf': float}
        """
        if not self.yolo_detector or self.yolo_detector.model is None:
            return []

        try:
            results = self.yolo_detector.model(img, verbose=False, conf=self.yolo_conf_thresh)
        except Exception as e:
            logger.error(f"YOLO 推理失败: {e}")
            return []

        boxes = []
        for r in results:
            names = r.names
            for box in r.boxes:
                cls_idx = int(box.cls[0].item())
                cls_name = names.get(cls_idx, "").lower()

                is_fire = (
                    "fire" in cls_name or "smoke" in cls_name or "flame" in cls_name
                    or (cls_idx == 0 and "person" not in cls_name)
                )
                if not is_fire:
                    continue

                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = xyxy
                boxes.append({
                    'box': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    'conf': conf,
                })
        return boxes

    def _merge(self, yolo_boxes: list, pnn_boxes: list) -> list:
        """
        决策级融合。

        融合策略:
        - YOLO conf > 0.6  → 直接保留（视觉特征强）
        - YOLO conf 0.2-0.6 且与 PNN 重叠 → 保留，置信度 +0.2（纹理确认）
        - YOLO conf < 0.2 或无 PNN 支持 → 丢弃（减少误报）
        - 剩余 PNN 框 → 作为低置信度结果保留（YOLO 漏检补充）
        """
        final_detections = []
        pnn_used = [False] * len(pnn_boxes)

        for yb in yolo_boxes:
            box_y = yb['box']
            conf_y = yb['conf']

            # 检查是否有 PNN 框与本框重叠
            has_overlap = False
            for i, pb in enumerate(pnn_boxes):
                if _compute_iou(box_y, pb['box']) > self._IOU_OVERLAP_THRESH:
                    has_overlap = True
                    pnn_used[i] = True

            if conf_y > 0.6:
                src = "YOLO+PNN" if has_overlap else "YOLO"
                final_detections.append((*box_y, conf_y, src))
            elif conf_y > 0.2 and has_overlap:
                final_detections.append((*box_y, conf_y + 0.2, "FUSED_WEAK"))
            # else: 弱 YOLO 且无纹理支持，丢弃

        # 添加未被 YOLO 覆盖的 PNN 框（可能是 YOLO 漏检的小火焰）
        for i, pb in enumerate(pnn_boxes):
            if not pnn_used[i]:
                final_detections.append((*pb['box'], 0.5, "PNN_ONLY"))

        return final_detections
