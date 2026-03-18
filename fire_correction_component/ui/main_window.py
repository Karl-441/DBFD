import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QListWidget, QListWidgetItem, QPushButton, 
                             QLabel, QComboBox, QMessageBox, QFileDialog)
from PyQt6.QtGui import QIcon, QPixmap, QAction, QKeySequence
from PyQt6.QtCore import Qt, QSize, QTimer
from .canvas import Canvas
from .config_ui import ConfigDialog

class MainWindow(QMainWindow):
    def __init__(self, config, config_path, db_manager, scanner, exporter, trainer):
        super().__init__()
        self.config = config
        self.config_path = config_path
        self.db_manager = db_manager
        self.scanner = scanner
        self.exporter = exporter
        self.trainer = trainer
        
        self.setWindowTitle("Fire Detection Correction Tool")
        self.resize(1200, 800)
        
        self._setup_ui()
        self._load_data()
        
        # Polling timer for new files
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._scan_new_files)
        self.poll_timer.start(self.config['input']['polling_interval'] * 1000)

    def _setup_ui(self):
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left Panel: File List
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Unprocessed Files:"))
        self.file_list = QListWidget()
        self.file_list.setIconSize(QSize(100, 100))
        self.file_list.itemClicked.connect(self._on_file_selected)
        left_panel.addWidget(self.file_list)
        
        # Reviewer Selection
        reviewer_layout = QHBoxLayout()
        reviewer_layout.addWidget(QLabel("Reviewer:"))
        self.reviewer_combo = QComboBox()
        self.reviewer_combo.addItems(self.config['gui']['reviewer_list'])
        self.reviewer_combo.setCurrentText(self.config['gui']['reviewer_name'])
        reviewer_layout.addWidget(self.reviewer_combo)
        left_panel.addLayout(reviewer_layout)
        
        # Batch Actions
        batch_layout = QVBoxLayout()
        btn_all_fire = QPushButton("Mark All as Fire")
        btn_all_fire.clicked.connect(self._mark_all_fire)
        btn_all_not_fire = QPushButton("Mark All as Not Fire")
        btn_all_not_fire.clicked.connect(self._mark_all_not_fire)
        batch_layout.addWidget(btn_all_fire)
        batch_layout.addWidget(btn_all_not_fire)
        left_panel.addLayout(batch_layout)
        
        main_layout.addLayout(left_panel, 1)
        
        # Right Panel: Canvas & Actions
        right_panel = QVBoxLayout()
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        self.lbl_current_file = QLabel("No file selected")
        toolbar_layout.addWidget(self.lbl_current_file)
        
        btn_export = QPushButton("Export Dataset")
        btn_export.clicked.connect(self._on_export)
        toolbar_layout.addWidget(btn_export)
        
        btn_train = QPushButton("Retrain Model")
        btn_train.clicked.connect(self._on_retrain)
        toolbar_layout.addWidget(btn_train)

        btn_settings = QPushButton("Settings")
        btn_settings.clicked.connect(self._on_settings)
        toolbar_layout.addWidget(btn_settings)
        
        right_panel.addLayout(toolbar_layout)
        
        # Canvas
        self.canvas = Canvas()
        self.canvas.label_changed.connect(self._on_label_changed)
        right_panel.addWidget(self.canvas, 5)
        
        main_layout.addLayout(right_panel, 4)
        
        # Shortcuts
        self._setup_shortcuts()

    def _setup_shortcuts(self):
        # Shortcut Y for Fire
        act_fire = QAction(self)
        act_fire.setShortcut(QKeySequence(Qt.Key.Key_Y))
        act_fire.triggered.connect(lambda: self._apply_label_to_selected("fire"))
        self.addAction(act_fire)
        
        # Shortcut N for Not Fire
        act_not_fire = QAction(self)
        act_not_fire.setShortcut(QKeySequence(Qt.Key.Key_N))
        act_not_fire.triggered.connect(lambda: self._apply_label_to_selected("not_fire"))
        self.addAction(act_not_fire)

    def _load_data(self):
        """Initial load from DB for unprocessed files."""
        # For simplicity, just get all files from DB where user_label is "pending"
        all_corrections = self.db_manager.get_corrections()
        pending_files = sorted(list(set(c['file_path'] for c in all_corrections if c['user_label'] == 'pending')))
        
        self.file_list.clear()
        for fpath in pending_files:
            item = QListWidgetItem(os.path.basename(fpath))
            item.setData(Qt.ItemDataRole.UserRole, fpath)
            # Try to add a thumbnail
            pixmap = QPixmap(fpath).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
            item.setIcon(QIcon(pixmap))
            self.file_list.addItem(item)

    def _on_file_selected(self, item):
        fpath = item.data(Qt.ItemDataRole.UserRole)
        self.lbl_current_file.setText(fpath)
        
        # Get all boxes for this file
        boxes_data = self.db_manager.get_corrections(fpath)
        self.canvas.set_image(fpath, boxes_data)
        self.canvas.setFocus()

    def _on_label_changed(self, box_id, new_label):
        fpath = self.lbl_current_file.text()
        if fpath == "No file selected":
            return
            
        reviewer = self.reviewer_combo.currentText()
        self.db_manager.update_correction(fpath, box_id, new_label, reviewer)
        
        # Update canvas
        for box in self.canvas.boxes:
            if box.id == box_id:
                box.label = new_label
        self.canvas.update()
        
        # Check if all boxes for this file are done
        all_boxes = self.db_manager.get_corrections(fpath)
        if all(b['user_label'] != 'pending' for b in all_boxes):
            # Move to next file? Or just refresh list
            self._load_data()

    def _apply_label_to_selected(self, label):
        if self.canvas.selected_box_id != -1:
            self._on_label_changed(self.canvas.selected_box_id, label)

    def _mark_all_fire(self):
        fpath = self.lbl_current_file.text()
        if fpath == "No file selected":
            return
        boxes_data = self.db_manager.get_corrections(fpath)
        for box in boxes_data:
            self._on_label_changed(box['box_id'], "fire")

    def _mark_all_not_fire(self):
        fpath = self.lbl_current_file.text()
        if fpath == "No file selected":
            return
        boxes_data = self.db_manager.get_corrections(fpath)
        for box in boxes_data:
            self._on_label_changed(box['box_id'], "not_fire")

    def _scan_new_files(self):
        new_data = self.scanner.scan_now()
        if new_data:
            self._load_data()

    def _on_export(self):
        try:
            output_dir = self.exporter.export()
            QMessageBox.information(self, "Export Successful", f"Dataset exported to:\n{output_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error during export: {e}")

    def _on_retrain(self):
        reply = QMessageBox.question(self, "Start Retraining", 
                                     "Do you want to start retraining the model with the corrected dataset?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            # Run in background or show a progress dialog?
            # For simplicity, just run and show message after
            success, msg = self.trainer.run_training()
            if success:
                QMessageBox.information(self, "Training Done", f"New model saved and evaluated.\n{msg}")
            else:
                QMessageBox.warning(self, "Training Failed/Rollback", f"Training did not meet requirements.\n{msg}")

    def _on_settings(self):
        dialog = ConfigDialog(self.config, self.config_path, self)
        if dialog.exec():
            # Config was saved, apply changes
            self.scanner.update_config(self.config['input']['base_dir'])
            self.exporter.update_config(self.config['export']['output_dir'], self.config['export']['format'])
            self.trainer.update_config(self.config)
            
            # Restart timer if interval changed
            self.poll_timer.stop()
            self.poll_timer.start(self.config['input']['polling_interval'] * 1000)
            
            QMessageBox.information(self, "Settings Applied", "Configuration has been updated successfully.")
