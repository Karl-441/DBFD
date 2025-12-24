import sys
import os
import cv2
import glob
import shutil
import json
import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QListWidget, QFileDialog, QMessageBox, 
    QFrame, QSplitter, QMenu, QInputDialog, QComboBox, QGroupBox,
    QScrollArea, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QAction, QImage, QIcon
from PyQt6.QtCore import Qt, QPoint, QSize, QRect

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DatasetCore:
    def __init__(self, root_dir=None):
        self.root_dir = root_dir
        self.images = []
        self.current_image_index = -1
        
    def load_images(self, directory):
        """Loads images from a directory (recursive or flat)."""
        self.root_dir = directory
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.images = []
        
        # Try standardized structure first: images/train, images/val
        std_train = os.path.join(directory, "images", "train")
        if os.path.exists(std_train):
            for ext in extensions:
                self.images.extend(glob.glob(os.path.join(std_train, ext)))
                self.images.extend(glob.glob(os.path.join(directory, "images", "val", ext)))
                self.images.extend(glob.glob(os.path.join(directory, "images", "test", ext)))
        else:
            # Flat directory
            for ext in extensions:
                self.images.extend(glob.glob(os.path.join(directory, ext)))
        
        self.images.sort()
        return len(self.images)

    def get_label_path(self, img_path):
        """Guess label path based on image path."""
        # Standard YOLO: images/x.jpg -> labels/x.txt
        # Or same dir: x.jpg -> x.txt
        
        base_dir = os.path.dirname(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Check 'labels' sibling folder
        if "images" in base_dir:
            label_dir = base_dir.replace("images", "labels")
            if os.path.exists(label_dir):
                return os.path.join(label_dir, base_name + ".txt")
        
        # Check same folder
        return os.path.join(base_dir, base_name + ".txt")

    def save_labels(self, img_path, boxes, classes):
        label_path = self.get_label_path(img_path)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        with open(label_path, 'w') as f:
            for box in boxes:
                # box: [cls_idx, xc, yc, w, h]
                line = f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
                f.write(line)

    def merge_dataset(self, source_dir):
        """Merges external dataset into current."""
        if not self.root_dir:
            return 0
            
        # Destination is always images/train in root
        dest_img_dir = os.path.join(self.root_dir, "images", "train")
        dest_lbl_dir = os.path.join(self.root_dir, "labels", "train")
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lbl_dir, exist_ok=True)
        
        # Scan source
        src_core = DatasetCore()
        src_core.load_images(source_dir)
        
        count = 0
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        for img_path in src_core.images:
            lbl_path = src_core.get_label_path(img_path)
            
            base_name = os.path.basename(img_path)
            name, ext = os.path.splitext(base_name)
            
            new_name = f"{name}_merged_{timestamp}{ext}"
            new_lbl_name = f"{name}_merged_{timestamp}.txt"
            
            shutil.copy2(img_path, os.path.join(dest_img_dir, new_name))
            if os.path.exists(lbl_path):
                shutil.copy2(lbl_path, os.path.join(dest_lbl_dir, new_lbl_name))
            count += 1
            
        return count

class Canvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap_orig = None
        self.boxes = [] # [cls, xc, yc, w, h]
        self.current_cls = 0
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet("background-color: #2b2b2b;")

    def load_image(self, pixmap, boxes):
        self.pixmap_orig = pixmap
        self.boxes = boxes
        self.update_display()

    def update_display(self):
        if not self.pixmap_orig: return
        
        # Calculate scaling to fit label while keeping aspect ratio
        w_avail = self.width()
        h_avail = self.height()
        
        if w_avail <= 0 or h_avail <= 0: return

        scaled_pixmap = self.pixmap_orig.scaled(w_avail, h_avail, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.scale_factor = scaled_pixmap.width() / self.pixmap_orig.width()
        
        self.offset_x = (w_avail - scaled_pixmap.width()) // 2
        self.offset_y = (h_avail - scaled_pixmap.height()) // 2
        
        self.setPixmap(scaled_pixmap)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap_orig: return
        
        painter = QPainter(self)
        
        # Draw existing boxes
        img_w = self.pixmap_orig.width()
        img_h = self.pixmap_orig.height()
        
        for box in self.boxes:
            cls, xc, yc, w, h = box
            
            # Color based on class
            color = Qt.GlobalColor.green if cls == 0 else Qt.GlobalColor.yellow
            painter.setPen(QPen(color, 2))
            
            px_w = w * img_w * self.scale_factor
            px_h = h * img_h * self.scale_factor
            px_x = (xc * img_w * self.scale_factor) - (px_w / 2) + self.offset_x
            px_y = (yc * img_h * self.scale_factor) - (px_h / 2) + self.offset_y
            
            painter.drawRect(int(px_x), int(px_y), int(px_w), int(px_h))
            
            # Label
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawText(int(px_x), int(px_y)-5, f"ID: {int(cls)}")

        # Draw current drawing
        if self.drawing and self.start_point and self.end_point:
            painter.setPen(QPen(Qt.GlobalColor.red, 2))
            
            x = min(self.start_point.x(), self.end_point.x())
            y = min(self.start_point.y(), self.end_point.y())
            w = abs(self.start_point.x() - self.end_point.x())
            h = abs(self.start_point.y() - self.end_point.y())
            
            painter.drawRect(x, y, w, h)

    def mousePressEvent(self, event):
        if not self.pixmap_orig: return
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.end_point = event.pos()
            
            # Convert to normalized coordinates
            x1 = min(self.start_point.x(), self.end_point.x()) - self.offset_x
            y1 = min(self.start_point.y(), self.end_point.y()) - self.offset_y
            w_px = abs(self.start_point.x() - self.end_point.x())
            h_px = abs(self.start_point.y() - self.end_point.y())
            
            # Check bounds
            if w_px < 5 or h_px < 5: return # Too small
            
            img_w = self.pixmap_orig.width()
            img_h = self.pixmap_orig.height()
            
            # Real image coords
            real_x = x1 / self.scale_factor
            real_y = y1 / self.scale_factor
            real_w = w_px / self.scale_factor
            real_h = h_px / self.scale_factor
            
            # Normalize YOLO (xc, yc, w, h)
            xc = (real_x + real_w/2) / img_w
            yc = (real_y + real_h/2) / img_h
            nw = real_w / img_w
            nh = real_h / img_h
            
            # Clamp
            xc = max(0, min(1, xc))
            yc = max(0, min(1, yc))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))
            
            self.boxes.append([self.current_cls, xc, yc, nw, nh])
            self.update()

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

class LabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DBFD Dataset Labeling Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        self.core = DatasetCore()
        self.current_img_path = None
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # --- Left Panel: File List & Controls ---
        left_panel = QFrame()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # 1. Dataset Controls
        gb_data = QGroupBox("Dataset Management")
        data_layout = QVBoxLayout()
        
        self.btn_open = QPushButton("Open Dataset Folder")
        self.btn_open.clicked.connect(self.open_folder)
        self.btn_open.setStyleSheet("background-color: #2196F3; color: white; padding: 5px;")
        
        self.btn_merge = QPushButton("Merge External Dataset")
        self.btn_merge.clicked.connect(self.merge_dataset)
        
        data_layout.addWidget(self.btn_open)
        data_layout.addWidget(self.btn_merge)
        gb_data.setLayout(data_layout)
        
        # 2. File List
        self.lbl_count = QLabel("Images: 0")
        self.list_files = QListWidget()
        self.list_files.currentRowChanged.connect(self.image_selected)
        
        # 3. Label Controls
        gb_label = QGroupBox("Labeling")
        label_layout = QVBoxLayout()
        
        self.combo_cls = QComboBox()
        self.combo_cls.addItems(["0: Fire", "1: Smoke/Other"])
        self.combo_cls.currentIndexChanged.connect(self.change_class)
        
        self.btn_save = QPushButton("Save Labels (Ctrl+S)")
        self.btn_save.clicked.connect(self.save_current)
        self.btn_save.setShortcut("Ctrl+S")
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        self.btn_clear = QPushButton("Clear All Boxes")
        self.btn_clear.clicked.connect(self.clear_boxes)
        self.btn_clear.setStyleSheet("background-color: #F44336; color: white;")
        
        label_layout.addWidget(QLabel("Current Class:"))
        label_layout.addWidget(self.combo_cls)
        label_layout.addWidget(self.btn_save)
        label_layout.addWidget(self.btn_clear)
        gb_label.setLayout(label_layout)
        
        # Add to left layout
        left_layout.addWidget(gb_data)
        left_layout.addWidget(self.lbl_count)
        left_layout.addWidget(self.list_files)
        left_layout.addWidget(gb_label)
        
        # --- Center Panel: Canvas ---
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        
        self.canvas = Canvas()
        right_layout.addWidget(self.canvas)
        
        # Nav Buttons
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("<< Previous")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next = QPushButton("Next >>")
        self.btn_next.clicked.connect(self.next_image)
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        right_layout.addLayout(nav_layout)
        
        # Main Layout
        layout.addWidget(left_panel)
        layout.addWidget(right_panel, stretch=1)
        
    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if path:
            count = self.core.load_images(path)
            self.lbl_count.setText(f"Images: {count}")
            self.list_files.clear()
            for img in self.core.images:
                self.list_files.addItem(os.path.basename(img))
                
            if count > 0:
                self.list_files.setCurrentRow(0)
            else:
                QMessageBox.warning(self, "No Images", "No images found in the selected folder.")

    def image_selected(self, row):
        if row < 0 or row >= len(self.core.images): return
        
        self.current_img_path = self.core.images[row]
        self.load_image_and_labels(self.current_img_path)

    def load_image_and_labels(self, img_path):
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            return
            
        boxes = []
        lbl_path = self.core.get_label_path(img_path)
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 5:
                        boxes.append(parts)
        
        self.canvas.load_image(pixmap, boxes)

    def change_class(self):
        self.canvas.current_cls = self.combo_cls.currentIndex()

    def save_current(self):
        if not self.current_img_path: return
        self.core.save_labels(self.current_img_path, self.canvas.boxes, [])
        
        # Visual feedback
        self.statusBar().showMessage(f"Saved {os.path.basename(self.current_img_path)}", 2000)

    def clear_boxes(self):
        self.canvas.boxes = []
        self.canvas.update()

    def next_image(self):
        curr = self.list_files.currentRow()
        if curr < self.list_files.count() - 1:
            self.list_files.setCurrentRow(curr + 1)

    def prev_image(self):
        curr = self.list_files.currentRow()
        if curr > 0:
            self.list_files.setCurrentRow(curr - 1)

    def merge_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Select External Dataset to Merge")
        if path:
            reply = QMessageBox.question(self, "Confirm Merge", 
                                       f"Are you sure you want to merge images from {os.path.basename(path)} into current dataset?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                count = self.core.merge_dataset(path)
                QMessageBox.information(self, "Success", f"Merged {count} images.")
                # Refresh
                if self.core.root_dir:
                    self.open_folder() # Re-load to see new files

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelingTool()
    window.show()
    sys.exit(app.exec())
