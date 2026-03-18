from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, 
    QGroupBox, QFormLayout, QMessageBox, QComboBox,
    QFileDialog, QLineEdit
)
from PyQt6.QtCore import Qt
import config

class ConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Configuration (系统设置)")
        self.resize(500, 400)
        self.init_ui()
        self.load_current_values()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Hardware & Camera Settings
        gb_hardware = QGroupBox("Camera & Hardware (摄像头与硬件)")
        form_hw = QFormLayout()
        
        self.cb_use_libcamera = QCheckBox("Use LibCamera (Pi Camera)")
        form_hw.addRow("Camera Driver:", self.cb_use_libcamera)
        
        self.sb_camera_index = QSpinBox()
        self.sb_camera_index.setRange(0, 10)
        form_hw.addRow("Camera Index:", self.sb_camera_index)
        
        self.sb_width = QSpinBox()
        self.sb_width.setRange(160, 1920)
        self.sb_width.setSingleStep(160)
        form_hw.addRow("Frame Width:", self.sb_width)
        
        self.sb_height = QSpinBox()
        self.sb_height.setRange(120, 1080)
        self.sb_height.setSingleStep(120)
        form_hw.addRow("Frame Height:", self.sb_height)
        
        self.sb_fps = QSpinBox()
        self.sb_fps.setRange(1, 120)
        form_hw.addRow("Target FPS:", self.sb_fps)

        self.sb_grab_drop = QSpinBox()
        self.sb_grab_drop.setRange(0, 10)
        self.sb_grab_drop.setToolTip("Number of frames to drop to reduce latency (丢帧以降低延迟)")
        form_hw.addRow("Grab Drop Count:", self.sb_grab_drop)
        
        gb_hardware.setLayout(form_hw)
        layout.addWidget(gb_hardware)

        # 2. YOLO Settings
        gb_yolo = QGroupBox("YOLO Detection Settings (YOLO 检测设置)")
        form_yolo = QFormLayout()
        
        self.cb_use_yolo = QCheckBox("Enable YOLO (启用 YOLO)")
        form_yolo.addRow("YOLO Active:", self.cb_use_yolo)

        self.edit_yolo_path = QLineEdit()
        btn_browse_yolo = QPushButton("Browse")
        btn_browse_yolo.clicked.connect(self.browse_yolo_model)
        h_yolo_path = QHBoxLayout()
        h_yolo_path.addWidget(self.edit_yolo_path)
        h_yolo_path.addWidget(btn_browse_yolo)
        form_yolo.addRow("Model Path:", h_yolo_path)

        self.sb_yolo_conf = QDoubleSpinBox()
        self.sb_yolo_conf.setRange(0.1, 1.0)
        self.sb_yolo_conf.setSingleStep(0.05)
        form_yolo.addRow("Conf Threshold:", self.sb_yolo_conf)

        self.sb_yolo_area = QSpinBox()
        self.sb_yolo_area.setRange(10, 2000)
        self.sb_yolo_area.setSingleStep(50)
        form_yolo.addRow("Min Area (px):", self.sb_yolo_area)

        gb_yolo.setLayout(form_yolo)
        layout.addWidget(gb_yolo)

        # 3. PNN Settings
        gb_pnn = QGroupBox("PNN Detection Settings (PNN 检测设置)")
        form_pnn = QFormLayout()
        
        self.cb_use_pnn = QCheckBox("Enable PNN (启用 PNN)")
        form_pnn.addRow("PNN Active:", self.cb_use_pnn)

        self.sb_pnn_width = QSpinBox()
        self.sb_pnn_width.setRange(32, 640)
        form_pnn.addRow("PNN Process Width:", self.sb_pnn_width)
        
        self.sb_pnn_height = QSpinBox()
        self.sb_pnn_height.setRange(24, 480)
        form_pnn.addRow("PNN Process Height:", self.sb_pnn_height)

        self.sb_pnn_samples = QSpinBox()
        self.sb_pnn_samples.setRange(10, 5000)
        form_pnn.addRow("Max Samples:", self.sb_pnn_samples)

        gb_pnn.setLayout(form_pnn)
        layout.addWidget(gb_pnn)

        # 4. Alarm Settings
        gb_alarm = QGroupBox("Alarm Settings (报警设置)")
        form_alarm = QFormLayout()
        
        self.sb_cooldown = QDoubleSpinBox()
        self.sb_cooldown.setRange(1.0, 60.0)
        self.sb_cooldown.setSingleStep(0.5)
        form_alarm.addRow("Alarm Cooldown (sec):", self.sb_cooldown)
        
        self.sb_alarm_pin = QSpinBox()
        self.sb_alarm_pin.setRange(0, 40)
        form_alarm.addRow("Alarm GPIO Pin (BCM):", self.sb_alarm_pin)

        self.combo_alarm_level = QComboBox()
        self.combo_alarm_level.addItems(["High (高电平触发)", "Low (低电平触发)"])
        form_alarm.addRow("Trigger Level:", self.combo_alarm_level)
        
        gb_alarm.setLayout(form_alarm)
        layout.addWidget(gb_alarm)

        # 5. Advanced / Performance
        gb_adv = QGroupBox("Performance & Advanced (性能与高级)")
        form_adv = QFormLayout()

        self.sb_detect_interval = QSpinBox()
        self.sb_detect_interval.setRange(1, 30)
        form_adv.addRow("Detect Interval (Frames):", self.sb_detect_interval)
        
        self.combo_device = QComboBox()
        self.combo_device.addItems(["cpu", "0 (GPU)", "auto"])
        form_adv.addRow("Inference Device:", self.combo_device)
        
        self.sb_max_memory = QSpinBox()
        self.sb_max_memory.setRange(128, 8192)
        self.sb_max_memory.setSingleStep(128)
        self.sb_max_memory.setSuffix(" MB")
        form_adv.addRow("Max Memory Limit:", self.sb_max_memory)
        
        self.sb_gc_interval = QSpinBox()
        self.sb_gc_interval.setRange(5, 600)
        self.sb_gc_interval.setSuffix(" sec")
        form_adv.addRow("GC Interval:", self.sb_gc_interval)
        
        gb_adv.setLayout(form_adv)
        layout.addWidget(gb_adv)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Save (保存)")
        btn_save.clicked.connect(self.save_settings)
        btn_cancel = QPushButton("Cancel (取消)")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    def browse_yolo_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", str(config.MODELS_DIR), 
            "YOLO Models (*.pt *.ncnn *.bin *.param);;All Files (*)"
        )
        if file_path:
            self.edit_yolo_path.setText(file_path)

    def load_current_values(self):
        """从 config 加载当前值"""
        try:
            # Hardware
            self.cb_use_libcamera.setChecked(config.USE_LIBCAMERA)
            self.sb_camera_index.setValue(config.CAMERA_INDEX)
            self.sb_width.setValue(config.FRAME_WIDTH)
            self.sb_height.setValue(config.FRAME_HEIGHT)
            self.sb_fps.setValue(config.FPS)
            self.sb_grab_drop.setValue(config.GRAB_DROP_COUNT)
            
            # YOLO
            self.cb_use_yolo.setChecked(config.USE_YOLO)
            self.edit_yolo_path.setText(str(config.YOLO_MODEL_PATH))
            self.sb_yolo_conf.setValue(config.YOLO_CONF_THRESH)
            self.sb_yolo_area.setValue(config.YOLO_MIN_AREA)

            # PNN
            self.cb_use_pnn.setChecked(config.USE_PNN)
            self.sb_pnn_width.setValue(config.PNN_TARGET_WIDTH)
            self.sb_pnn_height.setValue(config.PNN_TARGET_HEIGHT)
            self.sb_pnn_samples.setValue(config.PNN_MAX_SAMPLES)

            # Alarm
            self.sb_cooldown.setValue(config.ALARM_COOLDOWN)
            self.sb_alarm_pin.setValue(config.ALARM_GPIO_PIN)
            self.combo_alarm_level.setCurrentIndex(0 if config.ALARM_ACTIVE_HIGH else 1)

            # Performance
            self.sb_detect_interval.setValue(config.DETECT_INTERVAL)
            device_map = {"cpu": 0, "0": 1, "auto": 2}
            self.combo_device.setCurrentIndex(device_map.get(config.DEVICE, 0))
            self.sb_max_memory.setValue(config.MAX_MEMORY_MB)
            self.sb_gc_interval.setValue(config.GC_INTERVAL)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load config: {e}")

    def save_settings(self):
        """保存设置到 config 并写入文件"""
        try:
            # Hardware
            config.cfg.USE_LIBCAMERA = self.cb_use_libcamera.isChecked()
            config.cfg.CAMERA_INDEX = self.sb_camera_index.value()
            config.cfg.FRAME_WIDTH = self.sb_width.value()
            config.cfg.FRAME_HEIGHT = self.sb_height.value()
            config.cfg.FPS = self.sb_fps.value()
            config.cfg.GRAB_DROP_COUNT = self.sb_grab_drop.value()
            
            # YOLO
            config.cfg.USE_YOLO = self.cb_use_yolo.isChecked()
            config.cfg.YOLO_MODEL_PATH = self.edit_yolo_path.text()
            config.cfg.YOLO_CONF_THRESH = self.sb_yolo_conf.value()
            config.cfg.YOLO_MIN_AREA = self.sb_yolo_area.value()

            # PNN
            config.cfg.USE_PNN = self.cb_use_pnn.isChecked()
            config.cfg.PNN_TARGET_WIDTH = self.sb_pnn_width.value()
            config.cfg.PNN_TARGET_HEIGHT = self.sb_pnn_height.value()
            config.cfg.PNN_MAX_SAMPLES = self.sb_pnn_samples.value()

            # Alarm
            config.cfg.ALARM_COOLDOWN = self.sb_cooldown.value()
            config.cfg.ALARM_GPIO_PIN = self.sb_alarm_pin.value()
            config.cfg.ALARM_ACTIVE_HIGH = (self.combo_alarm_level.currentIndex() == 0)

            # Performance
            config.cfg.DETECT_INTERVAL = self.sb_detect_interval.value()
            device_map_rev = {0: "cpu", 1: "0", 2: "auto"}
            config.cfg.DEVICE = device_map_rev.get(self.combo_device.currentIndex(), "cpu")
            config.cfg.MAX_MEMORY_MB = self.sb_max_memory.value()
            config.cfg.GC_INTERVAL = self.sb_gc_interval.value()
            
            # 保存到文件
            config.cfg.save_config()
            
            QMessageBox.information(self, "Success", "Settings saved successfully!\nSome changes may require restart.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")
