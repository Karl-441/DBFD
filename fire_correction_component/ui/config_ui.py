import yaml
import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit, 
                             QPushButton, QHBoxLayout, QLabel, QSpinBox, 
                             QDoubleSpinBox, QComboBox, QCheckBox, QFileDialog)

class ConfigDialog(QDialog):
    def __init__(self, config, config_path, parent=None):
        super().__init__(parent)
        self.config = config
        self.config_path = config_path
        self.setWindowTitle("Configuration Settings")
        self.resize(500, 600)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Input Section
        form.addRow(QLabel("<b>Input Settings</b>"), QLabel(""))
        self.base_dir_edit = QLineEdit(self.config['input']['base_dir'])
        btn_browse_input = QPushButton("Browse...")
        btn_browse_input.clicked.connect(lambda: self._browse_dir(self.base_dir_edit))
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.base_dir_edit)
        input_layout.addWidget(btn_browse_input)
        form.addRow("Input Base Directory:", input_layout)

        self.polling_interval = QSpinBox()
        self.polling_interval.setRange(1, 3600)
        self.polling_interval.setValue(self.config['input']['polling_interval'])
        form.addRow("Polling Interval (s):", self.polling_interval)

        # Export Section
        form.addRow(QLabel("<b>Export Settings</b>"), QLabel(""))
        self.export_dir_edit = QLineEdit(self.config['export']['output_dir'])
        btn_browse_export = QPushButton("Browse...")
        btn_browse_export.clicked.connect(lambda: self._browse_dir(self.export_dir_edit))
        export_layout = QHBoxLayout()
        export_layout.addWidget(self.export_dir_edit)
        export_layout.addWidget(btn_browse_export)
        form.addRow("Export Directory:", export_layout)

        self.export_format = QComboBox()
        self.export_format.addItems(["coco", "voc"])
        self.export_format.setCurrentText(self.config['export']['format'])
        form.addRow("Export Format:", self.export_format)

        # Training Section
        form.addRow(QLabel("<b>Training Settings</b>"), QLabel(""))
        self.base_model_edit = QLineEdit(self.config['training']['base_model_path'])
        btn_browse_model = QPushButton("Browse...")
        btn_browse_model.clicked.connect(lambda: self._browse_file(self.base_model_edit, "Model Files (*.pt *.weights)"))
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.base_model_edit)
        model_layout.addWidget(btn_browse_model)
        form.addRow("Base Model Path:", model_layout)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(self.config['training']['epochs'])
        form.addRow("Epochs:", self.epochs)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(self.config['training']['batch_size'])
        form.addRow("Batch Size:", self.batch_size)

        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(5)
        self.lr.setRange(0.00001, 0.1)
        self.lr.setSingleStep(0.0001)
        self.lr.setValue(self.config['training']['learning_rate'])
        form.addRow("Learning Rate:", self.lr)

        self.auto_tune = QCheckBox()
        self.auto_tune.setChecked(self.config['training']['auto_tune'])
        form.addRow("Enable Auto-Tuning:", self.auto_tune)

        layout.addLayout(form)

        # Buttons
        btns = QHBoxLayout()
        btn_save = QPushButton("Save & Apply")
        btn_save.clicked.connect(self._on_save)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_save)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

    def _browse_dir(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory", line_edit.text())
        if dir_path:
            line_edit.setText(dir_path)

    def _browse_file(self, line_edit, filter):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", line_edit.text(), filter)
        if file_path:
            line_edit.setText(file_path)

    def _on_save(self):
        # Update config dictionary
        self.config['input']['base_dir'] = self.base_dir_edit.text()
        self.config['input']['polling_interval'] = self.polling_interval.value()
        self.config['export']['output_dir'] = self.export_dir_edit.text()
        self.config['export']['format'] = self.export_format.currentText()
        self.config['training']['base_model_path'] = self.base_model_edit.text()
        self.config['training']['epochs'] = self.epochs.value()
        self.config['training']['batch_size'] = self.batch_size.value()
        self.config['training']['learning_rate'] = self.lr.value()
        self.config['training']['auto_tune'] = self.auto_tune.isChecked()

        # Save to file
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.accept()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Save Error", f"Could not save configuration: {e}")
