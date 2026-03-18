import os
import json
import shutil
import logging
from datetime import datetime

class Exporter:
    def __init__(self, db_manager, output_base_dir="./exported_datasets", format="coco"):
        self.db_manager = db_manager
        self.output_base_dir = output_base_dir
        self.format = format
        self.logger = logging.getLogger(__name__)

    def update_config(self, output_base_dir, format):
        self.output_base_dir = output_base_dir
        self.format = format

    def export(self):
        """Export corrected data to the specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(self.output_base_dir, f"export_{timestamp}")
        img_dir = os.path.join(export_dir, "images")
        ann_dir = os.path.join(export_dir, "annotations")
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        
        # Get all non-pending corrections
        corrections = self.db_manager.get_corrections()
        valid_corrections = [c for c in corrections if c['user_label'] in ['fire', 'not_fire']]
        
        if not valid_corrections:
            raise ValueError("No corrected data available for export.")
            
        # Group by file path
        grouped = {}
        for c in valid_corrections:
            fpath = c['file_path']
            if fpath not in grouped:
                grouped[fpath] = []
            grouped[fpath].append(c)
            
        if self.format == "coco":
            self._export_coco(grouped, img_dir, ann_dir)
        elif self.format == "voc":
            self._export_voc(grouped, img_dir, ann_dir)
        else:
            raise ValueError(f"Unsupported export format: {self.format}")
            
        return export_dir

    def _export_coco(self, grouped, img_dir, ann_dir):
        """Export data in COCO format."""
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "fire", "supercategory": "none"},
                # "not_fire" is usually treated as background or a separate class
                {"id": 2, "name": "non_fire", "supercategory": "none"} 
            ]
        }
        
        ann_id = 1
        img_id = 1
        
        for fpath, boxes in grouped.items():
            # Copy image
            fname = os.path.basename(fpath)
            shutil.copy2(fpath, os.path.join(img_dir, fname))
            
            # Image info (dummy size for now, ideally read from image)
            coco_data["images"].append({
                "id": img_id,
                "file_name": fname,
                "width": 640, # Placeholder
                "height": 480 # Placeholder
            })
            
            for box in boxes:
                category_id = 1 if box['user_label'] == 'fire' else 2
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                w, h = x2 - x1, y2 - y1
                
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1
            img_id += 1
            
        with open(os.path.join(ann_dir, "instances.json"), 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=4)

    def _export_voc(self, grouped, img_dir, ann_dir):
        """Export data in Pascal VOC format (simple XMLs)."""
        # Pascal VOC uses one XML per image
        from xml.etree.ElementTree import Element, SubElement, ElementTree
        
        for fpath, boxes in grouped.items():
            fname = os.path.basename(fpath)
            shutil.copy2(fpath, os.path.join(img_dir, fname))
            
            root = Element("annotation")
            SubElement(root, "filename").text = fname
            size = SubElement(root, "size")
            SubElement(size, "width").text = "640"
            SubElement(size, "height").text = "480"
            SubElement(size, "depth").text = "3"
            
            for box in boxes:
                obj = SubElement(root, "object")
                SubElement(obj, "name").text = "fire" if box['user_label'] == 'fire' else "non_fire"
                bndbox = SubElement(obj, "bndbox")
                SubElement(bndbox, "xmin").text = str(int(box['x1']))
                SubElement(bndbox, "ymin").text = str(int(box['y1']))
                SubElement(bndbox, "xmax").text = str(int(box['x2']))
                SubElement(bndbox, "ymax").text = str(int(box['y2']))
            
            tree = ElementTree(root)
            tree.write(os.path.join(ann_dir, os.path.splitext(fname)[0] + ".xml"))
