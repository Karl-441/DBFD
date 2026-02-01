from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QScrollArea, QFrame, QFileDialog, QListWidget, QMessageBox, QSplitter
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QAction
from PyQt6.QtCore import Qt, QPoint, QSize, QTimer
import cv2
import os
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.dataset_manager import DatasetManager

"""
数据管理界面
    提供一个可视化的界面来管理数据集，包括：
    1. 浏览数据集图片
    2. 绘制/编辑边界框
    3. 保存 YOLO 格式的标签
    4. 合并外部数据集
    该模块主要用于辅助生成或修正训练数据。
"""

class LabelingCanvas(QLabel):
    """
    标注画布 (Labeling Canvas)
        自定义的 QLabel，支持鼠标拖拽绘制矩形框，并将其转换为归一化坐标。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None
        self.pixmap_orig = None
        self.boxes = [] # 存储归一化坐标 [x_center, y_center, w, h]
        self.current_start = None
        self.current_end = None
        self.setMouseTracking(True)
        self.scale_factor = 1.0
        
    def load_image(self, path):
        """加载图片"""
        self.image_path = path
        self.pixmap_orig = QPixmap(path)
        self.boxes = []
        self.update_display()
        
        # 尝试加载已存在的标签
        self.load_existing_label()
        
    def load_existing_label(self):
        """加载同名 txt 标签文件"""
        # image/train/x.jpg -> labels/train/x.txt
        lbl_path = self.image_path.replace("images", "labels").rsplit('.', 1)[0] + ".txt"
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 5:
                        # cls, xc, yc, w, h
                        self.boxes.append(parts[1:])
            self.update_display()

    def update_display(self):
        """刷新显示"""
        if not self.pixmap_orig: return
        
        # 缩放以适应窗口
        w_avail = self.width()
        h_avail = self.height()
        
        scaled = self.pixmap_orig.scaled(QSize(w_avail, h_avail), Qt.AspectRatioMode.KeepAspectRatio)
        self.scale_factor = scaled.width() / self.pixmap_orig.width()
        
        # 计算偏移量以居中
        self.offset_x = (w_avail - scaled.width()) // 2
        self.offset_y = (h_avail - scaled.height()) // 2
        
        self.setPixmap(scaled)
        
    def paintEvent(self, event):
        """绘制事件：画框"""
        super().paintEvent(event)
        if not self.pixmap_orig: return
        
        painter = QPainter(self)
        painter.setPen(QPen(Qt.GlobalColor.green, 2))
        
        # 绘制已有的框
        img_w = self.pixmap_orig.width()
        img_h = self.pixmap_orig.height()
        
        for box in self.boxes:
            xc, yc, w, h = box
            # 归一化坐标 -> 像素坐标
            px_w = w * img_w * self.scale_factor
            px_h = h * img_h * self.scale_factor
            px_x = (xc * img_w * self.scale_factor) - (px_w / 2) + self.offset_x
            px_y = (yc * img_h * self.scale_factor) - (px_h / 2) + self.offset_y
            
            painter.drawRect(int(px_x), int(px_y), int(px_w), int(px_h))
            
        # 绘制当前正在拖拽的框
        if self.current_start and self.current_end:
            painter.setPen(QPen(Qt.GlobalColor.red, 2))
            x = min(self.current_start.x(), self.current_end.x())
            y = min(self.current_start.y(), self.current_end.y())
            w = abs(self.current_start.x() - self.current_end.x())
            h = abs(self.current_start.y() - self.current_end.y())
            painter.drawRect(x, y, w, h)

    def mousePressEvent(self, event):
        if not self.pixmap_orig: return
        if event.button() == Qt.MouseButton.LeftButton:
            self.current_start = event.pos()
            self.current_end = event.pos()

    def mouseMoveEvent(self, event):
        if self.current_start:
            self.current_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.current_start and event.button() == Qt.MouseButton.LeftButton:
            self.current_end = event.pos()
            
            # 转换为画布相对坐标
            x1 = min(self.current_start.x(), self.current_end.x()) - self.offset_x
            y1 = min(self.current_start.y(), self.current_end.y()) - self.offset_y
            w_px = abs(self.current_start.x() - self.current_end.x())
            h_px = abs(self.current_start.y() - self.current_end.y())
            
            img_w = self.pixmap_orig.width()
            img_h = self.pixmap_orig.height()
            
            # 还原缩放
            real_x1 = x1 / self.scale_factor
            real_y1 = y1 / self.scale_factor
            real_w = w_px / self.scale_factor
            real_h = h_px / self.scale_factor
            
            # 计算归一化坐标 (xc, yc, w, h)
            norm_w = real_w / img_w
            norm_h = real_h / img_h
            norm_xc = (real_x1 + real_w/2) / img_w
            norm_yc = (real_y1 + real_h/2) / img_h
            
            # 过滤太小的误操作
            if norm_w > 0.01 and norm_h > 0.01:
                self.boxes.append([norm_xc, norm_yc, norm_w, norm_h])
            
            self.current_start = None
            self.update()

    def save_labels(self):
        """保存当前框到文件"""
        if not self.image_path: return
        manager = DatasetManager()
        manager.save_label(self.image_path, self.boxes)
        return True

    def clear_labels(self):
        """清空所有框"""
        self.boxes = []
        self.update()

class DataManagerUI(QWidget):
    """
    数据管理主控件
    """
    def __init__(self):
        super().__init__()
        self.manager = DatasetManager()
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        
        # 左侧: 文件列表
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        
        self.btn_load_dir = QPushButton("Load Dataset Dir (加载数据集)")
        self.btn_load_dir.clicked.connect(self.load_dir)
        left_layout.addWidget(self.btn_load_dir)
        
        self.list_files = QListWidget()
        self.list_files.currentRowChanged.connect(self.change_image)
        left_layout.addWidget(self.list_files)
        
        # 工具栏
        self.btn_save = QPushButton("Save Labels (YOLO txt) (保存标签)")
        self.btn_save.clicked.connect(self.save_current)
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white;")
        left_layout.addWidget(self.btn_save)
        
        self.btn_clear = QPushButton("Clear Boxes (清空框)")
        self.btn_clear.clicked.connect(lambda: self.canvas.clear_labels())
        left_layout.addWidget(self.btn_clear)
        
        # 高级功能
        self.btn_merge = QPushButton("Merge External Dataset (合并外部数据)")
        self.btn_merge.clicked.connect(self.merge_dataset)
        left_layout.addWidget(self.btn_merge)
        
        # 状态提示
        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: #4CAF50; font-weight: bold;")
        left_layout.addStretch()
        left_layout.addWidget(self.lbl_info)
        
        left_panel.setFixedWidth(250)
        
        # 右侧: 画布
        self.canvas = LabelingCanvas()
        self.canvas.setStyleSheet("background-color: #333;")
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(left_panel)
        layout.addWidget(self.canvas)
        
    def load_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if path:
            self.manager = DatasetManager(path)
            # 查找训练集图片
            imgs = self.manager.get_images('train')
            self.list_files.clear()
            for img in imgs:
                self.list_files.addItem(img)
                
    def change_image(self, row):
        item = self.list_files.item(row)
        if item:
            path = item.text()
            self.canvas.load_image(path)
            
    def save_current(self):
        if self.canvas.save_labels():
            # 显示保存成功提示
            self.lbl_info.setText("Labels saved! (标签已保存)")
            # 2秒后自动清除提示
            QTimer.singleShot(2000, lambda: self.lbl_info.setText(""))
            
    def merge_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Select External Dataset (YOLO format)")
        if path:
            count = self.manager.merge_datasets(path)
            QMessageBox.information(self, "Merge", f"Merged {count} images successfully.")
            self.load_dir() # 刷新列表

