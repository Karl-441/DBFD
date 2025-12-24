import cv2
import numpy as np
from algorithm.preprocess import preprocess_image
from algorithm.features import extract_features

class FusionDetector:
    def __init__(self, pnn_model, yolo_model):
        self.pnn = pnn_model
        self.yolo = yolo_model
        
        # Configuration
        self.yolo_conf_thresh = 0.4
        self.pnn_iou_thresh = 0.1 # Any overlap is good enough to confirm
        
    def detect(self, img):
        """
        Runs both models and fuses results.
        Returns list of (x, y, w, h, confidence, source)
        source: 'YOLO', 'PNN', 'FUSED'
        """
        # 1. Run YOLO
        yolo_boxes = [] # (x, y, w, h, conf)
        if self.yolo:
            results = self.yolo(img, verbose=False)
            for r in results:
                for box in r.boxes:
                    # class 0 is fire
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        w = x2 - x1
                        h = y2 - y1
                        yolo_boxes.append({'box': [int(x1), int(y1), int(w), int(h)], 'conf': conf})

        # 2. Run PNN
        pnn_boxes = [] # (x, y, w, h)
        if self.pnn:
            # PNN Pipeline
            try:
                mask = preprocess_image(img)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                
                for i in range(1, num_labels):
                    x, y, w, h, area = stats[i]
                    if area < 20: continue
                    
                    # Create ROI mask
                    component_mask = np.zeros_like(mask)
                    component_mask[labels == i] = 255
                    roi_mask = component_mask[y:y+h, x:x+w]
                    roi = img[y:y+h, x:x+w]
                    
                    try:
                        feats = extract_features(roi, roi_mask)
                        pred = self.pnn.predict(feats)[0]
                        if pred == 1:
                            pnn_boxes.append({'box': [x, y, w, h], 'conf': 1.0}) # PNN doesn't give prob in current impl
                    except:
                        pass
            except Exception as e:
                print(f"PNN Error: {e}")

        # 3. Fusion Logic
        final_detections = []
        
        # Strategy:
        # - High Conf YOLO (> 0.6) -> Keep (Strong Visual)
        # - Mid Conf YOLO (0.2 - 0.6) -> Keep ONLY if overlapped by PNN (Texture Confirmation)
        # - PNN Only -> Keep as Low Confidence (Potential Texture Match but weak shape)
        
        # Mark used PNN boxes
        pnn_used = [False] * len(pnn_boxes)
        
        for yb in yolo_boxes:
            box_y = yb['box']
            conf_y = yb['conf']
            
            # Check overlap with any PNN box
            has_overlap = False
            for i, pb in enumerate(pnn_boxes):
                box_p = pb['box']
                iou = self.compute_iou(box_y, box_p)
                if iou > 0.05: # Slight overlap
                    has_overlap = True
                    pnn_used[i] = True
            
            if conf_y > 0.6:
                # Strong YOLO - Keep
                src = "YOLO+PNN" if has_overlap else "YOLO"
                final_detections.append((*box_y, conf_y, src))
            elif conf_y > 0.2 and has_overlap:
                # Weak YOLO but confirmed by Texture - Keep and Boost
                final_detections.append((*box_y, conf_y + 0.2, "FUSED_WEAK"))
            else:
                # Weak YOLO, no texture support - Discard (False Positive reduction)
                pass
                
        # Add remaining PNN boxes (Texture only)
        # These might be small fires YOLO missed
        for i, pb in enumerate(pnn_boxes):
            if not pnn_used[i]:
                final_detections.append((*pb['box'], 0.5, "PNN_ONLY"))
                
        return final_detections

    def compute_iou(self, boxA, boxB):
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
