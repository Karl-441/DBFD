import shutil
import json
import datetime
from pathlib import Path

"""
    管理用于训练和验证的数据集。
    主要功能包括：
    1. 获取指定划分 (train/val) 的图片列表。
    2. 将检测结果保存为 YOLO 格式的标签文件 (.txt)。
    3. 合并多个数据集。
    4. 记录数据集的变更历史。
"""

class DatasetManager:
    def __init__(self, root_dir=None):
        """
            root_dir: 数据集根目录
        """
        if root_dir is None:
            self.root_dir = Path(__file__).resolve().parent.parent / "dataset"
        else:
            self.root_dir = Path(root_dir)
            
    def get_images(self, split='train'):
        """
        获取图片列表
        参数:
            split: 数据集划分 ('train', 'val', 'test')
        返回值:
            list: 图片路径列表 (str)
        """
        target_dir = self.root_dir / "images" / split
        if not target_dir.exists():
            return []
            
        # 返回字符串列表以保持兼容性
        return [str(p) for p in target_dir.glob("*.*") if p.is_file()]

    def save_label(self, img_path, boxes, classes=None):
        """
            将检测框坐标归一化并保存为 .txt 文件，与图片同名。
            格式: <class> <x_center> <y_center> <width> <height>
            img_path: 对应的图片路径
            boxes: 归一化后的框列表 [x_center, y_center, w, h]
            classes: 类别列表 (可选)
        """
        img_path = Path(img_path)
        
        str_dir = str(img_path.parent)
        if "images" in str_dir:
            label_dir = Path(str_dir.replace("images", "labels"))
        else:
            # Fallback
            label_dir = img_path.parent.parent / "labels" / img_path.parent.name
            
        if not label_dir.exists():
            label_dir.mkdir(parents=True, exist_ok=True)
            
        label_path = label_dir / (img_path.stem + ".txt")
        
        with label_path.open('w', encoding='utf-8') as f:
            for i, box in enumerate(boxes):
                cls = 0 
                if classes and i < len(classes):
                    cls = classes[i]
                # box is xc, yc, w, h
                line = f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"
                f.write(line)
        
        self.log_change("label_update", f"Updated labels for {img_path.name}")

    def merge_datasets(self, source_dir, dest_split='train'):
        """
        合并数据集
            将另一个YOLO格式合并到当前数据集中。
            自动重命名文件以避免冲突。
            source_dir: 源数据集目录 (必须包含 images 和 labels 子目录)
            dest_split: 目标划分 (默认 'train')
        返回值:
            int: 合并的图片数量
        """
        source_dir = Path(source_dir)
        # glob images
        src_images = list((source_dir / "images").glob("*.*"))
        
        count = 0
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        dest_img_dir = self.root_dir / "images" / dest_split
        dest_lbl_dir = self.root_dir / "labels" / dest_split
        
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_p in src_images:
            if not img_p.is_file(): continue
            
            name = img_p.stem
            ext = img_p.suffix
            
            # 查找对应的标签文件
            lbl_p = source_dir / "labels" / (name + ".txt")
            
            new_name = f"{name}_merged_{timestamp}{ext}"
            new_lbl_name = f"{name}_merged_{timestamp}.txt"
            
            shutil.copy2(img_p, dest_img_dir / new_name)
            if lbl_p.exists():
                shutil.copy2(lbl_p, dest_lbl_dir / new_lbl_name)
                
            count += 1
            
        self.log_change("merge", f"Merged {count} images from {source_dir}")
        return count

    def log_change(self, action, details):
        """记录变更日志"""
        log_path = self.root_dir / "dataset_version.json"
        history = []
        if log_path.exists():
            try:
                with log_path.open('r', encoding='utf-8') as f:
                    history = json.load(f)
            except: pass
            
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "details": details
        }
        history.append(entry)
        
        with log_path.open('w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
