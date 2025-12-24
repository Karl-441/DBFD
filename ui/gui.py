import sys
import cv2
import numpy as np
import time
import os
import mss
import glob
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QAction, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QSlider, QFileDialog, QGroupBox,
    QProgressBar, QSplitter, QFrame, QMessageBox, QScrollArea, QSizePolicy,
    QTabWidget
)

# Import algorithms
# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.preprocess import preprocess_image
from algorithm.features import extract_features
from algorithm.pnn import PNN
from algorithm.fusion import FusionDetector
from core.output_manager import OutputManager
from ui.data_manager_ui import DataManagerUI
import pickle
from ultralytics import YOLO

class AlgorithmWorker(QThread):
    result_signal = pyqtSignal(object, object, bool) # image, fps, has_fire
    
    def __init__(self, source_type, source_path, algorithm_type, pnn_model, yolo_model):
        super().__init__()
        self.source_type = source_type # 'image', 'video', 'camera', 'screen'
        self.source_path = source_path
        self.algorithm_type = algorithm_type # 'PNN', 'YOLO', 'FUSION'
        self.pnn_model = pnn_model
        self.yolo_model = yolo_model
        self.fusion_detector = FusionDetector(pnn_model, yolo_model)
        self.output_manager = OutputManager()
        self.running = True
        self.paused = False
        self.frame_count = 0
        
    def run(self):
        cap = None
        sct = None
        
        if self.source_type == 'video' or self.source_type == 'camera':
            cap = cv2.VideoCapture(self.source_path)
        elif self.source_type == 'screen':
            sct = mss.mss()
            monitor = sct.monitors[1] # Primary monitor
            
        elif self.source_type == 'image':
            img = cv2.imread(self.source_path)
            if img is not None:
                self.process_frame(img)
            self.running = False
            return

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
                
            start_time = time.time()
            frame = None
            
            if self.source_type in ['video', 'camera']:
                ret, frame = cap.read()
                if not ret:
                    if self.source_type == 'video':
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop
                        continue
                    else:
                        break
            elif self.source_type == 'screen':
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                # Resize for performance if needed
                frame = cv2.resize(frame, (1280, 720))
                
            if frame is not None:
                res_frame, has_fire = self.process_frame(frame)
                fps = 1.0 / (time.time() - start_time)
                self.result_signal.emit(res_frame, fps, has_fire)
                
                # Log metrics every 100 frames
                self.frame_count += 1
                if self.frame_count % 100 == 0:
                    self.output_manager.log_metric("fps", fps)
            else:
                time.sleep(0.01)
                
        if cap:
            cap.release()

    def process_frame(self, frame):
        detections = []
        vis = frame.copy()
        has_fire = False
        
        if self.algorithm_type == 'PNN':
            dets, _ = self.detect_pnn(frame)
            if dets: has_fire = True
            for (x, y, w, h) in dets:
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(vis, "FIRE (PNN)", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
        elif self.algorithm_type == 'YOLO':
            dets = self.detect_yolo(frame)
            if dets: has_fire = True
            for (x, y, w, h) in dets:
                cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(vis, "FIRE (YOLO)", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        elif self.algorithm_type == 'FUSION':
            results = self.fusion_detector.detect(frame)
            if results: has_fire = True
            for (x, y, w, h, conf, src) in results:
                color = (0, 255, 0) # Green for fused
                if "PNN" in src and "YOLO" not in src: color = (0, 0, 255)
                if "YOLO" in src and "PNN" not in src: color = (255, 0, 0)
                
                cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
                label = f"{src} {conf:.2f}"
                cv2.putText(vis, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis, has_fire

    def detect_pnn(self, img):
        try:
            mask = preprocess_image(img)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            detections = []
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                if area < 20: continue
                component_mask = np.zeros_like(mask)
                component_mask[labels == i] = 255
                roi = img[y:y+h, x:x+w]
                roi_mask = component_mask[y:y+h, x:x+w]
                try:
                    feats = extract_features(roi, roi_mask)
                    pred = self.pnn_model.predict(feats)[0]
                    if pred == 1:
                        detections.append((x, y, w, h))
                except: continue
            return detections, mask
        except: return [], None

    def detect_yolo(self, img):
        if self.yolo_model is None: return []
        try:
            results = self.yolo_model(img, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = box.xyxy[0]
                        detections.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
            return detections
        except: return []

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DBFD System - Drone Based Fire Detector")
        self.setGeometry(100, 100, 1280, 800)
        
        self.output_manager = OutputManager()
        # self.load_models() # Now called when needed or on init with default
        self.pnn_model = None
        self.yolo_model = None
        self.load_pnn_model()
        self.load_yolo_model("yolov8n.pt") # Default
        
        self.init_ui()
        
        self.worker = None
        self.current_image = None
        self.recording = False
        self.video_writer = None
        
    def load_pnn_model(self):
        try:
            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_pnn.pkl"), 'rb') as f:
                self.pnn_model = pickle.load(f)
        except:
            self.pnn_model = None

    def load_yolo_model(self, path):
        try:
            # Check if path is just a name, look in models/
            if not os.path.exists(path):
                alt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", path)
                if os.path.exists(alt_path):
                    path = alt_path
            
            self.yolo_model = YOLO(path)
            print(f"Loaded YOLO model: {path}")
            return True
        except Exception as e:
            print(f"Error loading YOLO: {e}")
            self.yolo_model = None
            return False

    def init_ui(self):
        # Tabs
        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        
        # Tab 1: Detection
        detection_widget = QWidget()
        self.setup_detection_ui(detection_widget)
        tabs.addTab(detection_widget, "Real-time Detection")
        
        # Tab 2: Data Manager
        # data_widget = DataManagerUI()
        # tabs.addTab(data_widget, "Data Management")
        pass
        
    def setup_detection_ui(self, widget):
        main_layout = QHBoxLayout(widget)
        
        # --- Left Panel: Controls ---
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # 1. Media Input
        gb_input = QGroupBox("Media Input")
        input_layout = QVBoxLayout()
        self.btn_upload = QPushButton("Upload Image")
        self.btn_upload.clicked.connect(self.upload_image)
        self.btn_camera = QPushButton("Open Camera")
        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_screen = QPushButton("Screen Capture")
        self.btn_screen.clicked.connect(self.start_screen)
        self.btn_video_file = QPushButton("Open Video File")
        self.btn_video_file.clicked.connect(self.upload_video)
        input_layout.addWidget(self.btn_upload)
        input_layout.addWidget(self.btn_video_file)
        input_layout.addWidget(self.btn_camera)
        input_layout.addWidget(self.btn_screen)
        gb_input.setLayout(input_layout)
        
        # 2. Algorithm Control
        gb_algo = QGroupBox("Algorithm Control")
        algo_layout = QVBoxLayout()
        algo_layout.addWidget(QLabel("Select Algorithm:"))
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["PNN (Color+Texture)", "YOLO (Deep Learning)", "FUSION (Best Accuracy)"])
        self.combo_algo.currentIndexChanged.connect(self.change_algorithm)
        algo_layout.addWidget(self.combo_algo)
        
        # YOLO Model Selector
        algo_layout.addWidget(QLabel("YOLO Model:"))
        self.combo_model = QComboBox()
        self.refresh_model_list()
        self.combo_model.currentIndexChanged.connect(self.change_yolo_model)
        algo_layout.addWidget(self.combo_model)
        
        self.btn_start = QPushButton("Start Processing")
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_start.setEnabled(False)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setEnabled(False)
        algo_layout.addWidget(self.btn_start)
        algo_layout.addWidget(self.btn_stop)
        self.lbl_status = QLabel("Status: Idle")
        algo_layout.addWidget(self.lbl_status)
        gb_algo.setLayout(algo_layout)
        
        # 3. Export
        gb_export = QGroupBox("Output")
        export_layout = QVBoxLayout()
        self.btn_save_img = QPushButton("Save Current Frame")
        self.btn_save_img.clicked.connect(self.save_image)
        self.btn_save_img.setEnabled(False)
        self.btn_record = QPushButton("Start Recording")
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_record.setEnabled(False)
        export_layout.addWidget(self.btn_save_img)
        export_layout.addWidget(self.btn_record)
        gb_export.setLayout(export_layout)
        
        left_layout.addWidget(gb_input)
        left_layout.addWidget(gb_algo)
        left_layout.addWidget(gb_export)
        left_layout.addStretch()
        
        # --- Center Panel: Visualization ---
        center_panel = QFrame()
        center_layout = QVBoxLayout(center_panel)
        self.display_label = QLabel("No Media")
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_label.setStyleSheet("background-color: black; color: white;")
        self.display_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        center_layout.addWidget(self.display_label)
        self.lbl_fps = QLabel("FPS: 0.0")
        center_layout.addWidget(self.lbl_fps)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(center_panel)
        
        # State
        self.source_type = None
        self.source_path = None

    def refresh_model_list(self):
        self.combo_model.clear()
        # Add default
        self.combo_model.addItem("yolov8n.pt")
        
        # Scan models/ dir
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        if os.path.exists(models_dir):
            files = glob.glob(os.path.join(models_dir, "*.pt"))
            for f in files:
                self.combo_model.addItem(os.path.basename(f))
                
    def change_yolo_model(self):
        model_name = self.combo_model.currentText()
        if self.load_yolo_model(model_name):
            self.lbl_status.setText(f"Loaded: {model_name}")
            # If running, restart
            if self.worker and self.worker.isRunning():
                self.stop_processing()
                self.start_processing()
        
    def upload_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.stop_processing()
            self.source_type = 'image'
            self.source_path = path
            self.show_preview(path)
            self.btn_start.setEnabled(True)
            self.lbl_status.setText("Status: Image Loaded")
            # Auto-start processing for immediate feedback
            self.start_processing()
            
    def upload_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi)")
        if path:
            self.stop_processing()
            self.source_type = 'video'
            self.source_path = path
            self.btn_start.setEnabled(True)
            self.lbl_status.setText("Status: Video Loaded")
            
    def start_camera(self):
        self.stop_processing()
        self.source_type = 'camera'
        self.source_path = 0
        self.btn_start.setEnabled(True)
        self.lbl_status.setText("Status: Camera Selected")
        self.start_processing()
        
    def start_screen(self):
        self.stop_processing()
        self.source_type = 'screen'
        self.source_path = None
        self.btn_start.setEnabled(True)
        self.lbl_status.setText("Status: Screen Selected")
        self.start_processing()

    def show_preview(self, path):
        pixmap = QPixmap(path)
        self.display_label.setPixmap(pixmap.scaled(self.display_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
        
    def change_algorithm(self):
        if self.worker and self.worker.isRunning():
            self.stop_processing()
            self.start_processing()
            
    def start_processing(self):
        if not self.source_type: return
        algo_map = {0: 'PNN', 1: 'YOLO', 2: 'FUSION'}
        algo = algo_map[self.combo_algo.currentIndex()]
        
        self.worker = AlgorithmWorker(
            self.source_type, self.source_path, algo, 
            self.pnn_model, self.yolo_model
        )
        self.worker.result_signal.connect(self.update_display)
        self.worker.start()
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save_img.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.lbl_status.setText(f"Status: Running ({algo})")
        self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
        
    def stop_processing(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        if self.recording:
            self.toggle_recording()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Status: Stopped")
        self.lbl_status.setStyleSheet("color: gray; font-weight: bold;")
        
    def update_display(self, frame, fps, has_fire):
        self.current_image = frame
        self.lbl_fps.setText(f"FPS: {fps:.2f}")
        
        # Fire Alert Overlay
        if has_fire:
            # Draw semi-transparent red box
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (300, 50), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Draw text
            cv2.putText(frame, "WARNING: FIRE DETECTED!", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.display_label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.display_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
        
        if self.recording and self.video_writer:
            self.video_writer.write(frame)

    def save_image(self):
        if self.current_image is not None:
            path = self.output_manager.save_prediction(self.current_image, [])
            QMessageBox.information(self, "Success", f"Image saved to {path}")

    def toggle_recording(self):
        if not self.recording:
            if self.current_image is None: return
            run_dir = self.output_manager.get_run_dir()
            path = os.path.join(run_dir, "recording.avi")
            h, w = self.current_image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(path, fourcc, 20.0, (w, h))
            self.recording = True
            self.btn_record.setText("Stop Recording")
            self.btn_record.setStyleSheet("background-color: red; color: white;")
        else:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.btn_record.setText("Start Recording")
            self.btn_record.setStyleSheet("")
            QMessageBox.information(self, "Success", "Recording saved!")

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
