from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QScrollArea, QFrame, QFileDialog, QListWidget, QMessageBox, QSplitter
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QAction
from PyQt6.QtCore import Qt, QPoint, QSize
import cv2
import os
import sys

# Add parent path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.dataset_manager import DatasetManager

class LabelingCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None
        self.pixmap_orig = None
        self.boxes = [] # List of [x_center, y_center, w, h] (normalized)
        self.current_start = None
        self.current_end = None
        self.setMouseTracking(True)
        self.scale_factor = 1.0
        
    def load_image(self, path):
        self.image_path = path
        self.pixmap_orig = QPixmap(path)
        self.boxes = []
        self.update_display()
        
        # Try load existing label
        self.load_existing_label()
        
    def load_existing_label(self):
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
        if not self.pixmap_orig: return
        
        # Scale to fit window
        w_avail = self.width()
        h_avail = self.height()
        
        scaled = self.pixmap_orig.scaled(QSize(w_avail, h_avail), Qt.AspectRatioMode.KeepAspectRatio)
        self.scale_factor = scaled.width() / self.pixmap_orig.width()
        
        # Offset to center
        self.offset_x = (w_avail - scaled.width()) // 2
        self.offset_y = (h_avail - scaled.height()) // 2
        
        self.setPixmap(scaled)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap_orig: return
        
        painter = QPainter(self)
        painter.setPen(QPen(Qt.GlobalColor.green, 2))
        
        # Draw existing boxes
        img_w = self.pixmap_orig.width()
        img_h = self.pixmap_orig.height()
        
        for box in self.boxes:
            xc, yc, w, h = box
            # Convert norm to pixel
            px_w = w * img_w * self.scale_factor
            px_h = h * img_h * self.scale_factor
            px_x = (xc * img_w * self.scale_factor) - (px_w / 2) + self.offset_x
            px_y = (yc * img_h * self.scale_factor) - (px_h / 2) + self.offset_y
            
            painter.drawRect(int(px_x), int(px_y), int(px_w), int(px_h))
            
        # Draw current drag
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
            
            # Convert to normalized coords
            x1 = min(self.current_start.x(), self.current_end.x()) - self.offset_x
            y1 = min(self.current_start.y(), self.current_end.y()) - self.offset_y
            w_px = abs(self.current_start.x() - self.current_end.x())
            h_px = abs(self.current_start.y() - self.current_end.y())
            
            img_w = self.pixmap_orig.width()
            img_h = self.pixmap_orig.height()
            
            # De-scale
            real_x1 = x1 / self.scale_factor
            real_y1 = y1 / self.scale_factor
            real_w = w_px / self.scale_factor
            real_h = h_px / self.scale_factor
            
            # Normalize (xc, yc, w, h)
            norm_w = real_w / img_w
            norm_h = real_h / img_h
            norm_xc = (real_x1 + real_w/2) / img_w
            norm_yc = (real_y1 + real_h/2) / img_h
            
            if norm_w > 0.01 and norm_h > 0.01:
                self.boxes.append([norm_xc, norm_yc, norm_w, norm_h])
            
            self.current_start = None
            self.update()

    def save_labels(self):
        if not self.image_path: return
        manager = DatasetManager()
        manager.save_label(self.image_path, self.boxes)
        return True

    def clear_labels(self):
        self.boxes = []
        self.update()

class DataManagerUI(QWidget):
    def __init__(self):
        super().__init__()
        self.manager = DatasetManager()
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        
        # Left: File List
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        
        self.btn_load_dir = QPushButton("Load Dataset Dir")
        self.btn_load_dir.clicked.connect(self.load_dir)
        left_layout.addWidget(self.btn_load_dir)
        
        self.list_files = QListWidget()
        self.list_files.currentRowChanged.connect(self.change_image)
        left_layout.addWidget(self.list_files)
        
        # Tools
        self.btn_save = QPushButton("Save Labels (YOLO txt)")
        self.btn_save.clicked.connect(self.save_current)
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white;")
        left_layout.addWidget(self.btn_save)
        
        self.btn_clear = QPushButton("Clear Boxes")
        self.btn_clear.clicked.connect(lambda: self.canvas.clear_labels())
        left_layout.addWidget(self.btn_clear)
        
        # Advanced
        self.btn_merge = QPushButton("Merge External Dataset")
        self.btn_merge.clicked.connect(self.merge_dataset)
        left_layout.addWidget(self.btn_merge)
        
        left_panel.setFixedWidth(250)
        
        # Right: Canvas
        self.canvas = LabelingCanvas()
        self.canvas.setStyleSheet("background-color: #333;")
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(left_panel)
        layout.addWidget(self.canvas)
        
    def load_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if path:
            self.manager = DatasetManager(path)
            # Find train images
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
            # Flash success?
            pass
            
    def merge_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Select External Dataset (YOLO format)")
        if path:
            count = self.manager.merge_datasets(path)
            QMessageBox.information(self, "Merge", f"Merged {count} images successfully.")
            self.load_dir() # Refresh
