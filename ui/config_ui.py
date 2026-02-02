from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, 
    QGroupBox, QFormLayout, QMessageBox
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

        # 1. Hardware Settings
        gb_hardware = QGroupBox("Hardware Settings (硬件设置)")
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
        
        gb_hardware.setLayout(form_hw)
        layout.addWidget(gb_hardware)

        # 2. Algorithm Settings
        gb_algo = QGroupBox("Algorithm Settings (算法设置)")
        form_algo = QFormLayout()
        
        self.sb_detect_interval = QSpinBox()
        self.sb_detect_interval.setRange(1, 30)
        form_algo.addRow("Detect Interval (Frames):", self.sb_detect_interval)
        
        self.sb_pnn_width = QSpinBox()
        self.sb_pnn_width.setRange(32, 640)
        form_algo.addRow("PNN Process Width:", self.sb_pnn_width)
        
        self.sb_pnn_height = QSpinBox()
        self.sb_pnn_height.setRange(24, 480)
        form_algo.addRow("PNN Process Height:", self.sb_pnn_height)
        
        gb_algo.setLayout(form_algo)
        layout.addWidget(gb_algo)

        # 3. Alarm Settings
        gb_alarm = QGroupBox("Alarm Settings (报警设置)")
        form_alarm = QFormLayout()
        
        self.sb_cooldown = QDoubleSpinBox()
        self.sb_cooldown.setRange(1.0, 60.0)
        self.sb_cooldown.setSingleStep(0.5)
        form_alarm.addRow("Alarm Cooldown (sec):", self.sb_cooldown)
        
        gb_alarm.setLayout(form_alarm)
        layout.addWidget(gb_alarm)

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

    def load_current_values(self):
        """从 config 加载当前值"""
        try:
            self.cb_use_libcamera.setChecked(config.USE_LIBCAMERA)
            self.sb_camera_index.setValue(config.CAMERA_INDEX)
            self.sb_width.setValue(config.FRAME_WIDTH)
            self.sb_height.setValue(config.FRAME_HEIGHT)
            self.sb_fps.setValue(config.FPS)
            
            self.sb_detect_interval.setValue(config.DETECT_INTERVAL)
            self.sb_pnn_width.setValue(config.PNN_TARGET_WIDTH)
            self.sb_pnn_height.setValue(config.PNN_TARGET_HEIGHT)
            
            self.sb_cooldown.setValue(config.ALARM_COOLDOWN)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load config: {e}")

    def save_settings(self):
        """保存设置到 config 并写入文件"""
        try:
            # 更新内存中的配置
            config.cfg.USE_LIBCAMERA = self.cb_use_libcamera.isChecked()
            config.cfg.CAMERA_INDEX = self.sb_camera_index.value()
            config.cfg.FRAME_WIDTH = self.sb_width.value()
            config.cfg.FRAME_HEIGHT = self.sb_height.value()
            config.cfg.FPS = self.sb_fps.value()
            
            config.cfg.DETECT_INTERVAL = self.sb_detect_interval.value()
            config.cfg.PNN_TARGET_WIDTH = self.sb_pnn_width.value()
            config.cfg.PNN_TARGET_HEIGHT = self.sb_pnn_height.value()
            
            config.cfg.ALARM_COOLDOWN = self.sb_cooldown.value()
            
            # 保存到文件
            config.cfg.save_config()
            
            QMessageBox.information(self, "Success", "Settings saved successfully!\nSome changes may require restart.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")
