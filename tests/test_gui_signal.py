import os

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_worker_signal_matches_update_display(qtbot, monkeypatch):
    import ui.gui as gui

    monkeypatch.setattr(gui.MainWindow, "load_pnn_model", lambda self, path=None: True)
    monkeypatch.setattr(gui.MainWindow, "load_yolo_model", lambda self, path: True)

    window = gui.MainWindow()
    qtbot.addWidget(window)

    worker = gui.AlgorithmWorker("video", "", "PNN", None, None)
    worker.result_signal.connect(window.update_display)

    raw = np.zeros((120, 160, 3), dtype=np.uint8)
    vis = raw.copy()

    worker.result_signal.emit(raw, vis, 12.3, False)
    qtbot.wait(50)

    assert window.display_label.pixmap() is not None

