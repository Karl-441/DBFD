from PyQt6.QtWidgets import QWidget, QMenu
from PyQt6.QtGui import QPainter, QPen, QColor, QImage, QPixmap, QAction
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QPointF

class BoundingBox:
    def __init__(self, id, rect, label=None, confidence=0.0):
        self.id = id
        self.rect = rect # QRectF
        self.label = label # "fire", "not_fire", or None
        self.confidence = confidence
        self.is_selected = False
        self.is_hovered = False

class Canvas(QWidget):
    label_changed = pyqtSignal(int, str) # box_id, new_label

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.pixmap = None
        self.boxes = []
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.setMouseTracking(True)
        self.selected_box_id = -1
        self.hovered_box_id = -1

    def set_image(self, img_path, boxes_data):
        self.image = QImage(img_path)
        self.pixmap = QPixmap.fromImage(self.image)
        self.boxes = []
        for det in boxes_data:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            rect = QRectF(x1, y1, x2-x1, y2-y1)
            self.boxes.append(BoundingBox(det['box_id'], rect, det['user_label'], det['confidence']))
        self.update()

    def paintEvent(self, event):
        if not self.pixmap:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw image
        target_rect = self._get_image_rect()
        painter.drawPixmap(target_rect.toRect(), self.pixmap)

        # Draw boxes
        for box in self.boxes:
            pen_color = QColor(255, 0, 0) # Default red
            if box.label == "fire":
                pen_color = QColor(0, 255, 0) # Green for confirmed fire
            elif box.label == "not_fire":
                pen_color = QColor(255, 255, 0) # Yellow for confirmed non-fire

            pen = QPen(pen_color, 2)
            if box.is_hovered or box.is_selected:
                pen.setWidth(4)
            
            painter.setPen(pen)
            
            # Map box coordinates to widget coordinates
            widget_rect = self._map_to_widget(box.rect)
            painter.drawRect(widget_rect)
            
            # Draw label and confidence
            label_text = f"ID: {box.id} ({box.confidence:.2f})"
            if box.label:
                label_text += f" - {box.label}"
            
            painter.drawText(widget_rect.topLeft() + QPointF(0, -5), label_text)

    def _get_image_rect(self):
        if not self.pixmap:
            return QRectF()
        
        # Calculate aspect ratio
        view_w = self.width()
        view_h = self.height()
        img_w = self.pixmap.width()
        img_h = self.pixmap.height()
        
        scale = min(view_w / img_w, view_h / img_h)
        self.scale = scale
        
        w = img_w * scale
        h = img_h * scale
        x = (view_w - w) / 2
        y = (view_h - h) / 2
        
        self.offset = QPointF(x, y)
        return QRectF(x, y, w, h)

    def _map_to_widget(self, rect):
        return QRectF(
            rect.x() * self.scale + self.offset.x(),
            rect.y() * self.scale + self.offset.y(),
            rect.width() * self.scale,
            rect.height() * self.scale
        )

    def _map_to_image(self, pos):
        return QPointF(
            (pos.x() - self.offset.x()) / self.scale,
            (pos.y() - self.offset.y()) / self.scale
        )

    def mouseMoveEvent(self, event):
        pos = self._map_to_image(event.position())
        old_hovered = self.hovered_box_id
        self.hovered_box_id = -1
        
        for box in self.boxes:
            box.is_hovered = box.rect.contains(pos)
            if box.is_hovered:
                self.hovered_box_id = box.id
        
        if old_hovered != self.hovered_box_id:
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self._map_to_image(event.position())
            self.selected_box_id = -1
            for box in self.boxes:
                box.is_selected = box.rect.contains(pos)
                if box.is_selected:
                    self.selected_box_id = box.id
            self.update()
        
        elif event.button() == Qt.MouseButton.RightButton:
            pos = self._map_to_image(event.position())
            for box in self.boxes:
                if box.rect.contains(pos):
                    self._show_context_menu(event.globalPosition().toPoint(), box)
                    break

    def _show_context_menu(self, pos, box):
        menu = QMenu(self)
        fire_action = menu.addAction("Confirm Fire (Y)")
        not_fire_action = menu.addAction("Confirm Not Fire (N)")
        
        action = menu.exec(pos)
        if action == fire_action:
            self.label_changed.emit(box.id, "fire")
        elif action == not_fire_action:
            self.label_changed.emit(box.id, "not_fire")

    def keyPressEvent(self, event):
        if self.selected_box_id != -1:
            if event.key() == Qt.Key.Key_Y:
                self.label_changed.emit(self.selected_box_id, "fire")
            elif event.key() == Qt.Key.Key_N:
                self.label_changed.emit(self.selected_box_id, "not_fire")
        super().keyPressEvent(event)
