import os
import shutil
import glob
import json
import datetime

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
            self.root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
        else:
            self.root_dir = root_dir
            
    def get_images(self, split='train'):
        """
        获取图片列表 (Get Images)
        参数:
            split: 数据集划分 ('train', 'val', 'test')
        返回值:
            list: 图片路径列表
        """
        path = os.path.join(self.root_dir, "images", split, "*.*")
        return glob.glob(path)

    def save_label(self, img_path, boxes, classes=None):
        """
            将检测框坐标归一化并保存为 .txt 文件，与图片同名。
            格式: <class> <x_center> <y_center> <width> <height>
            img_path: 对应的图片路径
            boxes: 归一化后的框列表 [x_center, y_center, w, h]
            classes: 类别列表 (可选)
        """
        # 假设标签路径结构与图片路径对应
        # images/train/img.jpg -> labels/train/img.txt
        
        dir_name = os.path.dirname(img_path) # .../images/train
        base_name = os.path.basename(img_path)
        name_no_ext = os.path.splitext(base_name)[0]
        
        # 将路径中的 'images' 替换为 'labels'
        label_dir = dir_name.replace("images", "labels")
        if not os.path.exists(label_dir):
            os.makedirs(label_dir, exist_ok=True)
            
        label_path = os.path.join(label_dir, name_no_ext + ".txt")
        
        with open(label_path, 'w') as f:
            for i, box in enumerate(boxes):
                cls = 0 
                if classes and i < len(classes):
                    cls = classes[i]
                # box is xc, yc, w, h
                line = f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"
                f.write(line)
        
        self.log_change("label_update", f"Updated labels for {base_name}")

    def merge_datasets(self, source_dir, dest_split='train'):
        """
        合并数据集 (Merge Datasets)
            将另一个数据集 (YOLO 格式) 合并到当前数据集中。
            自动重命名文件以避免冲突。
            source_dir: 源数据集目录 (必须包含 images 和 labels 子目录)
            dest_split: 目标划分 (默认 'train')
        返回值:
            int: 合并的图片数量
        """
        # 简单的复制并重命名以避免冲突
        src_images = glob.glob(os.path.join(source_dir, "images", "*.*"))
        
        count = 0
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        dest_img_dir = os.path.join(self.root_dir, "images", dest_split)
        dest_lbl_dir = os.path.join(self.root_dir, "labels", dest_split)
        
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lbl_dir, exist_ok=True)
        
        for img_p in src_images:
            base = os.path.basename(img_p)
            name, ext = os.path.splitext(base)
            
            # 查找对应的标签文件
            lbl_p = os.path.join(source_dir, "labels", name + ".txt")
            
            new_name = f"{name}_merged_{timestamp}{ext}"
            new_lbl_name = f"{name}_merged_{timestamp}.txt"
            
            shutil.copy2(img_p, os.path.join(dest_img_dir, new_name))
            if os.path.exists(lbl_p):
                shutil.copy2(lbl_p, os.path.join(dest_lbl_dir, new_lbl_name))
                
            count += 1
            
        self.log_change("merge", f"Merged {count} images from {source_dir}")
        return count

    def log_change(self, action, details):
        """记录变更日志 (Log Change)"""
        log_path = os.path.join(self.root_dir, "dataset_version.json")
        history = []
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    history = json.load(f)
            except: pass
            
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "details": details
        }
        history.append(entry)
        
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=4)
