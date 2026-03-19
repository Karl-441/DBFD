import os

import importlib.util
import numpy as np
from PyQt6.QtWidgets import QApplication

import sys


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    gui_path = os.path.join(repo_root, "ui", "gui.py")

    spec = importlib.util.spec_from_file_location("dbfd_gui", gui_path)
    if spec is None or spec.loader is None:
        raise SystemExit("FAILED: cannot load ui/gui.py")
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)

    gui.MainWindow.load_pnn_model = lambda self, path=None: True
    gui.MainWindow.load_yolo_model = lambda self, path: True

    app = QApplication([])
    window = gui.MainWindow()

    worker = gui.AlgorithmWorker("video", "", "PNN", None, None)
    worker.result_signal.connect(window.update_display)

    raw = np.zeros((120, 160, 3), dtype=np.uint8)
    vis = raw.copy()
    worker.result_signal.emit(raw, vis, 12.3, False)

    app.processEvents()

    if window.display_label.pixmap() is None:
        raise SystemExit("FAILED: display_label pixmap is None")

    print("OK")


if __name__ == "__main__":
    main()
