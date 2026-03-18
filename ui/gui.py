import sys
import cv2
import numpy as np
import time
import os
import gc
import config
try:
    import mss
except ImportError:
    mss = None
    print("Warning: 'mss' library not found. Screen capture disabled.")
import glob
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QAction, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QSlider, QFileDialog, QGroupBox,
    QProgressBar, QSplitter, QFrame, QMessageBox, QScrollArea, QSizePolicy,
    QTabWidget
)

# 导入算法模块
# 将父目录添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.preprocess import preprocess_image
from algorithm.features import extract_features
from algorithm.pnn import PNN
from algorithm.yolo_detector import YoloDetector
from algorithm.fusion import FusionDetector
from core.output_manager import OutputManager
from core.alarm_manager import AlarmManager
from core.hardware_booster import HardwareBooster
from ui.config_ui import ConfigDialog
from ui.converter_ui import ModelConverterDialog
import pickle
import glob

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Warning: Ultralytics not installed. YOLO disabled.")

"""
主界面模块
    提供基于 PyQt6 的图形用户界面，集成视频流显示、算法控制、报警状态反馈等功能。
    包含两个主要类：
    1. AlgorithmWorker: 后台线程，负责视频流读取、处理和算法推理，避免阻塞 UI。
    2. MainWindow: 主窗口，负责 UI 布局和交互逻辑。
"""

class AlgorithmWorker(QThread):
    """
    算法工作线程
        在后台执行耗时的图像处理和模型推理任务。
        通过信号将处理结果（图像、FPS、检测状态）发送回主线程进行显示。
    """
    # 信号定义: 图像数据, FPS, 是否发现火灾
    result_signal = pyqtSignal(object, object, bool) 
    
    def __init__(self, source_type, source_path, algorithm_type, pnn_model, yolo_detector):
        super().__init__()
        self.source_type = source_type # 来源类型: 'image', 'video', 'camera', 'screen'
        self.source_path = source_path
        self.algorithm_type = algorithm_type # 算法类型: 'PNN', 'YOLO', 'FUSION'
        self.pnn_model = pnn_model
        self.yolo_detector = yolo_detector
        self.fusion_detector = FusionDetector(pnn_model, yolo_detector)
        self.output_manager = OutputManager()
        self.alarm_manager = AlarmManager()
        self.running = True
        self.paused = False
        self.frame_count = 0
        self.output_interval = 3 # 每 3 帧保存一次处理后的图像
        
        # 缓存上一帧的检测结果，用于跳帧时的绘制
        self.last_detections = [] 
        self.last_has_fire = False
        self.last_detect_time = 0
        
    def _setup_opencv_camera(self, camera_index):
        """
        初始化 OpenCV 摄像头 (H.264 优化模式)
        """
        # 针对 H.264 流优化探测参数
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "probesize;32|analyzeduration;0"
        
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        
        # 尝试设置 H.264 格式
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return None
        return cap

    def run(self):
        """线程主入口"""
        cap = None
        sct = None
        
        # 1. 初始化输入源
        if self.source_type == 'video':
            cap = cv2.VideoCapture(self.source_path)
        elif self.source_type == 'camera':
             if config.USE_LIBCAMERA:
                 try:
                     from core.camera_wrapper import LibCameraWrapper
                     cap = LibCameraWrapper(config.FRAME_WIDTH, config.FRAME_HEIGHT, config.FPS)
                     print("GUI: Using LibCameraWrapper")
                 except ImportError as e:
                     print(f"GUI: Failed to load LibCameraWrapper: {e}")
                     cap = self._setup_opencv_camera(self.source_path)
             else:
                cap = self._setup_opencv_camera(self.source_path)
                
        elif self.source_type == 'screen':
            if mss is None:
                print("Error: Screen capture requires 'mss' library.")
                self.running = False
                return
            if not os.environ.get('DISPLAY') and sys.platform != 'win32':
                print("Error: Screen capture requires X11 DISPLAY. Disabling screen capture.")
                self.running = False
                return
            try:
                sct = mss.mss()
                monitor = sct.monitors[1] # 主显示器
            except Exception as e:
                print(f"Screen capture init error: {e}")
                self.running = False
                return
            
        elif self.source_type == 'image':
            img = cv2.imread(self.source_path)
            if img is not None:
                vis, has_fire = self.process_frame(img)
                # 自动保存处理结果
                self.output_manager.save_prediction(vis, [], metadata={"source": self.source_path, "has_fire": has_fire})
                self.result_signal.emit(vis, 0.0, has_fire)
            self.running = False
            return

        print(f"Worker started with algorithm: {self.algorithm_type}")

        # 2. 主循环
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
                
            start_time = time.time()
            frame = None
            
            # 读取帧
            if self.source_type in ['video', 'camera']:
                # --- 关键优化：解决高延迟 (Clear Buffer Logic) ---
                if self.source_type == 'camera':
                    # 丢弃堆积在缓冲区的老帧，只处理最新帧
                    for _ in range(config.GRAB_DROP_COUNT):
                        if not cap.grab(): break
                    ret, frame = cap.retrieve()
                else:
                    ret, frame = cap.read()

                if not ret:
                    if self.source_type == 'video':
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 视频循环播放
                        continue
                    else:
                        # 对于摄像头，持续尝试读取 (LibCameraWrapper 会自动重启)
                        time.sleep(0.05)
                        continue
            elif self.source_type == 'screen':
                try:
                    sct_img = sct.grab(monitor)
                except Exception as e:
                    print(f"Screen capture error: {e}")
                    time.sleep(0.5)
                    continue
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                # 调整大小以提高性能
                frame = cv2.resize(frame, (1280, 720))
                
            if frame is not None:
                # 处理帧
                # 每隔 2 帧进行一次深度学习检测，以提高实时性 (Frame Skipping)
                do_detect = (self.frame_count % 2 == 0)
                res_frame, has_fire = self.process_frame(frame, do_detect)
                
                fps = 1.0 / (time.time() - start_time)
                self.result_signal.emit(res_frame, fps, has_fire)
                
                # --- 新增：每处理 3 帧保存一次包含检测框的图像和 JSON ---
                if has_fire and (self.frame_count % self.output_interval == 0):
                    self.output_manager.save_prediction(
                        res_frame, 
                        self.last_detections, 
                        metadata={
                            "fps": round(fps, 2),
                            "algorithm": self.algorithm_type,
                            "frame_id": self.frame_count
                        }
                    )
                
                # 立即清理
                del frame
                
                # 记录指标
                self.frame_count += 1
                if self.frame_count % 100 == 0:
                    self.output_manager.log_metric("fps", fps)
                
                # 强制 GC
                if self.frame_count % config.GC_INTERVAL == 0:
                    gc.collect() 
            else:
                time.sleep(0.01)

        # 清理资源
        if sct:
            try:
                sct.close()
            except:
                pass
        if cap:
            cap.release()

    def process_frame(self, frame, do_detect=True):
        """
        处理单帧图像：检测与绘制
        """
        vis = frame.copy()
        
        # 1. 执行检测 (仅当 do_detect 为 True 时)
        if do_detect:
            self.last_detections = []
            current_has_fire = False
            
            try:
                if self.algorithm_type == 'PNN':
                    dets, _ = self.detect_pnn(frame)
                    if dets: 
                        current_has_fire = True
                        for (x, y, w, h) in dets:
                            self.last_detections.append((x, y, w, h, "FIRE (PNN)", (0, 0, 255)))
                        
                elif self.algorithm_type == 'YOLO':
                    # 调试日志：确认进入了 YOLO 分支
                    if self.frame_count % 30 == 0:
                        print(f"[Worker Debug] Current Algorithm: {self.algorithm_type}, Detector OK: {self.yolo_detector is not None}")
                    
                    dets = self.detect_yolo(frame)
                    if dets: 
                        current_has_fire = True
                        for (x, y, w, h) in dets:
                            self.last_detections.append((x, y, w, h, "FIRE (YOLO)", (0, 0, 255)))
                        # 调试日志：确认检测到火灾
                        print(f"[Worker Debug] YOLO detected fire! Added {len(dets)} boxes.")
                
                elif self.algorithm_type == 'FUSION':
                    results = self.fusion_detector.detect(frame)
                    if results: 
                        current_has_fire = True
                        for (x, y, w, h, conf, src) in results:
                            color = (0, 255, 0)
                            if "PNN" in src and "YOLO" not in src: color = (0, 0, 255)
                            if "YOLO" in src and "PNN" not in src: color = (255, 0, 0)
                            label = f"{src} {conf:.2f}"
                            self.last_detections.append((x, y, w, h, label, color))
            except Exception as e:
                print(f"Detection Error: {e}")

            # 报警逻辑重构：引入冷却机制 (Cooldown Logic)
            if current_has_fire:
                self.last_has_fire = True
                self.last_detect_time = time.time()
                self.alarm_manager.trigger()
            else:
                # 检查是否超过冷却时间 (Check Cooldown)
                if self.last_has_fire:
                    if (time.time() - self.last_detect_time) > config.ALARM_COOLDOWN:
                        self.last_has_fire = False
                        self.alarm_manager.stop()

        # 2. 绘制结果
        if self.last_detections:
            for (x, y, w, h, label, color) in self.last_detections:
                cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
                cv2.putText(vis, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        return vis, self.last_has_fire

    def detect_pnn(self, img):
        """PNN 检测流程"""
        try:
            target_w, target_h = config.PNN_TARGET_WIDTH, config.PNN_TARGET_HEIGHT
            h0, w0 = img.shape[:2]
            small = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            mask = preprocess_image(small)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            sx = w0 / float(target_w)
            sy = h0 / float(target_h)
            detections = []
            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                if area < 12: 
                    continue
                component_mask = np.zeros_like(mask)
                component_mask[labels == i] = 255
                roi = small[y:y+h, x:x+w]
                roi_mask = component_mask[y:y+h, x:x+w]
                try:
                    feats = extract_features(roi, roi_mask)
                    pred = self.pnn_model.predict(feats)[0]
                    if pred == 1:
                        xr = int(x * sx)
                        yr = int(y * sy)
                        wr = int(w * sx)
                        hr = int(h * sy)
                        detections.append((xr, yr, wr, hr))
                except: 
                    continue
            return detections, mask
        except: return [], None

    def detect_yolo(self, img):
        """
        YOLO 火灾检测流程 (代理到 YoloDetector)
        返回: list of (x, y, w, h)
        """
        if self.yolo_detector is None:
            return []
        return self.yolo_detector.detect(img)

    def stop(self):
        """停止线程"""
        self.running = False
        self.wait()
        self.alarm_manager.cleanup()

class MainWindow(QMainWindow):
    """
    主窗口类
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DBFD System - Drone Based Fire Detector (无人机火灾检测系统)")
        self.setGeometry(100, 100, 1280, 800)
        
        # 0. 基础变量初始化 (必须最先执行，防止信号触发导致 AttributeError)
        self.output_manager = OutputManager()
        self.hardware_booster = HardwareBooster()
        self.worker = None
        self.pnn_model = None
        self.yolo_detector = None
        self.current_image = None
        self.recording = False
        self.video_writer = None
        
        # 1. 模型加载
        self.load_pnn_model()
        
        # 优先使用配置中的 YOLO 模型路径，如果没有则尝试从 models 目录自动发现
        initial_yolo = getattr(config, 'YOLO_MODEL_PATH', "best.pt")
        self.load_yolo_model(initial_yolo)
        
        # 2. UI 初始化
        self.init_ui()
        
        # 3. 定时器：硬件监控 (Hardware Monitoring Timer)
        self.hw_timer = QTimer(self)
        self.hw_timer.timeout.connect(self.update_hw_stats)
        self.hw_timer.start(3000) # 每 3 秒更新一次
        
        # 4. 根据配置设置默认算法
        if getattr(config, 'USE_YOLO', False):
            self.combo_algo.setCurrentIndex(1) # YOLO
        elif getattr(config, 'USE_PNN', False):
            self.combo_algo.setCurrentIndex(0) # PNN
            
    def update_hw_stats(self):
        """更新状态栏硬件信息"""
        stats = self.hardware_booster.get_stats()
        msg = f"Temp: {stats['temp']} | CPU: {stats['cpu_clock']} | GPU: {stats['gpu_clock']} | Vulkan: {stats['vulkan']} | Status: {stats['throttled']}"
        self.statusBar().showMessage(msg)
        
    def load_pnn_model(self, path=None):
        """加载 PNN 模型"""
        try:
            if not path:
                # 尝试加载最新的 pnn_latest.pkl，如果不存在则加载 model_pnn.pkl
                base_dir = os.path.dirname(os.path.dirname(__file__))
                # 先检查 models 目录
                path = os.path.join(base_dir, "models", "pnn_latest.pkl")
                if not os.path.exists(path):
                     # 回退到旧路径
                     path = os.path.join(base_dir, "model_pnn.pkl")
            
            # 处理从下拉框传入的纯文件名
            if not os.path.exists(path):
                alt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", path)
                if os.path.exists(alt_path):
                    path = alt_path

            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.pnn_model = pickle.load(f)
                print(f"Loaded PNN model: {path}")
                return True
            else:
                self.pnn_model = None
                return False
        except Exception as e:
            print(f"Error loading PNN: {e}")
            self.pnn_model = None
            return False

    def load_yolo_model(self, path):
        """加载 YOLO 检测器"""
        try:
            # 1. 路径自动补全与检索 (Path retrieval)
            if not path or not os.path.exists(path):
                base_dir = os.path.dirname(os.path.dirname(__file__))
                models_dir = os.path.join(base_dir, "models")
                
                # 尝试 A: 搜索 models 目录下的同名文件
                alt_path = os.path.join(models_dir, os.path.basename(path))
                if os.path.exists(alt_path):
                    path = alt_path
                else:
                    # 尝试 B: 如果默认模型不存在，搜索 models 目录下任何 .pt 文件
                    pt_files = glob.glob(os.path.join(models_dir, "*.pt"))
                    if pt_files:
                        path = pt_files[0]
                        print(f"Warning: {os.path.basename(path)} not found, using first available: {os.path.basename(path)}")
                    else:
                        print(f"Error: No YOLO model files found in {models_dir}")
                        self.yolo_detector = None
                        return False
            
            # 2. 实例化加载 (Initialization)
            self.yolo_detector = YoloDetector(path)
            if self.yolo_detector.model:
                print(f"Loaded YOLO model successfully from: {path}")
                return True
            else:
                self.yolo_detector = None
                return False
        except Exception as e:
            print(f"Error initializing YOLO detector: {e}")
            self.yolo_detector = None
            return False

    def init_ui(self):
        """初始化 UI 布局"""
        # 标签页容器
        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        
        # 实时检测
        detection_widget = QWidget()
        self.setup_detection_ui(detection_widget)
        tabs.addTab(detection_widget, "Real-time Detection (实时检测)")
        
        # 数据管理
        # tabs.addTab(data_widget, "Data Management")
        pass
        
    def setup_detection_ui(self, widget):
        main_layout = QHBoxLayout(widget)
        
        # --- 左侧面板: 控制区 ---
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # 0. System Config
        self.btn_config = QPushButton("System Settings (系统设置)")
        self.btn_config.clicked.connect(self.open_settings)
        left_layout.addWidget(self.btn_config)

        # 1. 输入源选择 (Media Input)
        gb_input = QGroupBox("Media Input (输入源)")
        input_layout = QVBoxLayout()
        self.btn_upload = QPushButton("Upload Image (上传图片)")
        self.btn_upload.clicked.connect(self.upload_image)
        self.btn_camera = QPushButton("Open Camera (打开摄像头)")
        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_screen = QPushButton("Screen Capture (屏幕捕获)")
        self.btn_screen.clicked.connect(self.start_screen)
        if mss is None:
            self.btn_screen.setEnabled(False)
            self.btn_screen.setToolTip("Install 'mss' to enable screen capture")
        
        self.btn_video_file = QPushButton("Open Video File (打开视频)")
        self.btn_video_file.clicked.connect(self.upload_video)
        input_layout.addWidget(self.btn_upload)
        input_layout.addWidget(self.btn_video_file)
        input_layout.addWidget(self.btn_camera)
        input_layout.addWidget(self.btn_screen)
        gb_input.setLayout(input_layout)
        
        # 1. 算法选择 (Algorithm Select)
        algo_group = QGroupBox("算法控制 (Algorithm Control)")
        algo_layout = QVBoxLayout()
        
        # Initialize status label early to avoid AttributeError during model loading
        self.lbl_status = QLabel("Status: Idle")
        
        algo_layout.addWidget(QLabel("Select Algorithm:"))
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["PNN (Texture/Color)", "Deep Learning (YOLO/NCNN)", "FUSION (Hybrid)"])
        self.combo_algo.currentIndexChanged.connect(self.change_algorithm)
        algo_layout.addWidget(self.combo_algo)
        
        # PNN 模型选择器
        algo_layout.addWidget(QLabel("PNN Model:"))
        self.combo_pnn = QComboBox()
        self.refresh_pnn_list()
        self.combo_pnn.currentIndexChanged.connect(self.change_pnn_model)
        algo_layout.addWidget(self.combo_pnn)
        # Load initial PNN model
        if self.combo_pnn.count() > 0:
             self.change_pnn_model()

        # 深度学习模型选择 (Deep Learning Model Select)
        algo_layout.addWidget(QLabel("DL Model (PT/NCNN):"))
        self.combo_yolo = QComboBox()
        self.refresh_yolo_list()
        self.combo_yolo.currentIndexChanged.connect(self.change_yolo_model)
        algo_layout.addWidget(self.combo_yolo)
        # Load initial YOLO model
        if self.combo_yolo.count() > 0:
             self.change_yolo_model()

        # 模型转换按钮 (Model Converter Button)
        self.btn_convert_ui = QPushButton("Convert Model (模型转换器)")
        self.btn_convert_ui.clicked.connect(self.open_model_converter)
        self.btn_convert_ui.setStyleSheet("background: #2196F3; color: white; margin-top: 5px;")
        algo_layout.addWidget(self.btn_convert_ui)
        
        self.btn_start = QPushButton("Start Processing (开始)")
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_start.setEnabled(False)
        self.btn_stop = QPushButton("Stop (停止)")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setEnabled(False)
        algo_layout.addWidget(self.btn_start)
        algo_layout.addWidget(self.btn_stop)
        algo_layout.addWidget(self.lbl_status)
        algo_group.setLayout(algo_layout)
        
        # 3. 输出控制 (Export)
        gb_export = QGroupBox("Output (输出)")
        export_layout = QVBoxLayout()
        self.btn_save_img = QPushButton("Save Current Frame (保存当前帧)")
        self.btn_save_img.clicked.connect(self.save_image)
        self.btn_save_img.setEnabled(False)
        self.btn_record = QPushButton("Start Recording (开始录制)")
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_record.setEnabled(False)
        export_layout.addWidget(self.btn_save_img)
        export_layout.addWidget(self.btn_record)
        gb_export.setLayout(export_layout)
        
        left_layout.addWidget(gb_input)
        left_layout.addWidget(algo_group)
        left_layout.addWidget(gb_export)
        left_layout.addStretch()
        
        # --- 中间面板: 可视化 ---
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

    def open_model_converter(self):
        """
        打开模型转换器对话框 (Open Model Converter Dialog)
        """
        dialog = ModelConverterDialog(self)
        dialog.exec()
        # 转换完成后刷新模型列表
        self.refresh_yolo_list()
        self.refresh_pnn_list()
        
        # 状态
        self.source_type = None
        self.source_path = None

    def open_settings(self):
        """打开设置对话框"""
        dialog = ConfigDialog(self)
        if dialog.exec():
            # 设置保存后，某些参数可能需要重启生效，
            # 但部分参数（如 DETECT_INTERVAL）会立即生效，因为 Worker 直接读取 config
            self.lbl_status.setText("Status: Settings Updated")

    def refresh_yolo_list(self):
        """扫描 models 目录，获取可用的深度学习模型 (PT 和 NCNN)"""
        self.combo_yolo.clear()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "models")
        
        if not os.path.exists(models_dir):
            try:
                os.makedirs(models_dir)
            except OSError:
                pass

        if os.path.exists(models_dir):
            # 1. 扫描 .pt 文件
            pt_files = glob.glob(os.path.join(models_dir, "*.pt"))
            for f in pt_files:
                name = os.path.basename(f)
                self.combo_yolo.addItem(f"PyTorch: {name}", f)
            
            # 2. 扫描 NCNN 模型目录 (通常以 _ncnn_model 结尾)
            ncnn_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.endswith('_ncnn_model')]
            for d in ncnn_dirs:
                path = os.path.join(models_dir, d)
                self.combo_yolo.addItem(f"NCNN: {d}", path)
                
            if self.combo_yolo.count() == 0:
                self.combo_yolo.addItem("No models found", "")

    def refresh_pnn_list(self):
        """刷新 PNN 模型列表"""
        self.combo_pnn.clear()
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "models")
        
        # 1. Scan models/ directory for .pkl files
        if os.path.exists(models_dir):
            files = glob.glob(os.path.join(models_dir, "*.pkl"))
            for f in files:
                name = os.path.basename(f)
                self.combo_pnn.addItem(name, f)

        # 2. Check for legacy model_pnn.pkl in root
        root_pnn = os.path.join(base_dir, "model_pnn.pkl")
        if os.path.exists(root_pnn):
             self.combo_pnn.addItem("model_pnn.pkl (Root)", "model_pnn.pkl")

        if self.combo_pnn.count() == 0:
             self.combo_pnn.addItem("No PNN models found", "")
                
    def change_yolo_model(self):
        path = self.combo_yolo.currentData()
        if not path:
             # If no path in data, it might be the "No models found" item or text-only
             return

        if self.load_yolo_model(path):
            self.lbl_status.setText(f"Loaded: {os.path.basename(path)}")
            # 如果正在运行，重启
            if self.worker and self.worker.isRunning():
                self.stop_processing()
                self.start_processing()

    def change_pnn_model(self):
        model_name = self.combo_pnn.currentText()
        # Handle custom data (e.g. legacy root path)
        model_data = self.combo_pnn.currentData()
        path = model_data if model_data else model_name
        
        if not path: return

        if self.load_pnn_model(path):
            self.lbl_status.setText(f"Loaded PNN: {model_name}")
            # 如果正在运行，重启
            if self.worker and self.worker.isRunning():
                self.stop_processing()
                self.start_processing()
        
    def open_settings(self):
        """打开设置对话框"""
        dialog = ConfigDialog(self)
        if dialog.exec():
            # 设置保存后，某些参数可能需要重启生效，
            # 但部分参数（如 DETECT_INTERVAL）会立即生效，因为 Worker 直接读取 config
            self.lbl_status.setText("Status: Settings Updated")

    def upload_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.stop_processing()
            self.source_type = 'image'
            self.source_path = path
            self.show_preview(path)
            self.btn_start.setEnabled(True)
            self.lbl_status.setText("Status: Image Loaded")
            # 自动开始处理以获得即时反馈
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
        
        # --- CPU + GPU 混合加速优化 (CPU + GPU Hybrid Acceleration) ---
        # 1. 开启 OpenCL 进行硬件级图像处理和渲染 (GPU Offloading)
        try:
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                print("GUI: GPU Acceleration (OpenCL) enabled for rendering.")
        except:
            cv2.ocl.setUseOpenCL(False)
            
        # 2. 显式设置 CPU 推理线程数 (Max Performance)
        try:
            import torch
            # 设置 CPU 推理线程数为核心数的一半，预留核心给 GPU 渲染和系统任务
            num_cores = os.cpu_count() or 4
            torch.set_num_threads(max(1, num_cores // 2))
            print(f"GUI: Torch CPU threads optimized to {torch.get_num_threads()}")
        except:
            pass
        
        algo_map = {0: 'PNN', 1: 'YOLO', 2: 'FUSION'}
        algo = algo_map[self.combo_algo.currentIndex()]
        
        self.worker = AlgorithmWorker(
            self.source_type, self.source_path, algo, 
            self.pnn_model, self.yolo_detector
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
        """更新 UI 显示"""
        # 优化: 不要强引用当前帧，或者确保正确替换
        self.current_image = frame 
        self.lbl_fps.setText(f"FPS: {fps:.2f}")
        
        # 火灾警告叠加层
        if has_fire:
            # 绘制半透明红色框
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (300, 50), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            del overlay
            
            # 绘制文字
            cv2.putText(frame, "WARNING: FIRE DETECTED!", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 将 BGR 转为 RGB 用于 Qt 显示
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # 创建 QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 缩放并显示 (使用 FastTransformation 保证性能)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.display_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )
        self.display_label.setPixmap(pixmap)
        
        if self.recording and self.video_writer:
            self.video_writer.write(frame)
        
        # 清理局部变量
        del rgb_image
        del qt_image
        del pixmap

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

