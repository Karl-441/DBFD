import sys
import yaml
import logging
import os
from PyQt6.QtWidgets import QApplication
from core.database import DatabaseManager
from core.scanner import Scanner
from core.exporter import Exporter
from core.trainer import Trainer
from ui.main_window import MainWindow

def setup_logging(config):
    log_cfg = config['logging']
    os.makedirs(os.path.dirname(log_cfg['file_path']), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_cfg['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_cfg['file_path']),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Starting Fire Detection Correction Tool...")
    
    # Initialize Core Components
    db_manager = DatabaseManager(os.path.join(os.path.dirname(__file__), "corrections.sqlite"))
    scanner = Scanner(config['input']['base_dir'], db_manager)
    exporter = Exporter(db_manager, config['export']['output_dir'], config['export']['format'])
    trainer = Trainer(config)
    
    # Run GUI
    app = QApplication(sys.argv)
    window = MainWindow(config, config_path, db_manager, scanner, exporter, trainer)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
