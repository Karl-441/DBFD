import sys
import argparse
import time
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from core.memory_monitor import MemoryMonitor

def main():
    parser = argparse.ArgumentParser(description="DBFD Raspberry Pi Edition")
    parser.add_argument("--headless", action="store_true", help="Run without GUI (recommended for low RAM)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index")
    args = parser.parse_args()

    print(f"DBFD-Raspberry Starting...")
    print(f"Mode: {'Headless' if args.headless else 'GUI'}")
    print(f"Memory Limit: {config.MAX_MEMORY_MB} MB")

    # Start Memory Monitor
    monitor = MemoryMonitor(threshold_mb=config.MAX_MEMORY_MB)
    monitor.start()

    try:
        if args.headless:
            try:
                from core.headless_runner import run_headless
                run_headless(args.camera)
            except ImportError as e:
                print("\nCRITICAL ERROR: Missing dependencies for Headless Mode.")
                print(f"Details: {e}")
                print("Please run: sudo apt install python3-opencv")
                print("Or: pip install opencv-python-headless\n")
                sys.exit(1)
        else:
            try:
                # Lazy import to save memory if not used
                from PyQt6.QtWidgets import QApplication
                from ui.gui import MainWindow
                
                app = QApplication(sys.argv)
                window = MainWindow()
                window.show()
                sys.exit(app.exec())
            except ImportError as e:
                print(f"GUI libraries missing or failed to load: {e}")
                print("Attempting fallback to headless mode...")
                try:
                    from core.headless_runner import run_headless
                    run_headless(args.camera)
                except ImportError as e2:
                    print("\nCRITICAL ERROR: Failed to fallback to headless mode.")
                    print(f"Details: {e2}")
                    print("Please ensure OpenCV is installed: sudo apt install python3-opencv\n")
                    sys.exit(1)
            except Exception as e:
                 print(f"Critical Error in GUI: {e}")
                 # Ensure monitor stops
                 monitor.stop()
                 sys.exit(1)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        monitor.stop()

if __name__ == "__main__":
    main()
