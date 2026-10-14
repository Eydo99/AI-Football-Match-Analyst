import os
import sys
import cv2
import json
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QFileDialog, QTabWidget, QGraphicsView, 
                               QGraphicsScene, QMessageBox, QProgressBar, QLabel, QGraphicsEllipseItem, QGraphicsTextItem, QComboBox)
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QBrush, QPen, QFont
from PySide6.QtCore import Qt, QTimer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline", "core"))
try:
    from pitch_keypoint_detector import PitchKeypointDetector
except ImportError as e:
    PitchKeypointDetector = None

try:
    from worker import PipelineWorker
except ImportError as e:
    print(f"Failed to import PipelineWorker: {e}")
    PipelineWorker = None

LANDMARKS = {

    "Center": (52.5, 34.0),
    "Top Mid": (52.5, 0.0),
    "Bot Mid": (52.5, 68.0),
    "Top L": (0.0, 0.0),
    "Top R": (105.0, 0.0),
    "Bot R": (105.0, 68.0),
    "Bot L": (0.0, 68.0),

    "Circ L": (43.35, 34.0),
    "Circ R": (61.65, 34.0),
    "Circ T": (52.5, 24.85),
    "Circ B": (52.5, 43.15),

    "Pen L": (11.0, 34.0),
    "Pen R": (94.0, 34.0),

    "Box L-TL": (0.0, 13.84),
    "Box L-TR": (16.5, 13.84),
    "Box L-BR": (16.5, 54.16),
    "Box L-BL": (0.0, 54.16),

    "Box R-TL": (88.5, 13.84),
    "Box R-TR": (105.0, 13.84),
    "Box R-BR": (105.0, 54.16),
    "Box R-BL": (88.5, 54.16),

    "Arc L-Tip": (20.15, 34.0),
    "Arc L-Top": (16.5, 26.69),
    "Arc L-Bot": (16.5, 41.31),

    "Arc R-Tip": (84.85, 34.0),
    "Arc R-Top": (88.5, 26.69),
    "Arc R-Bot": (88.5, 41.31),
}

class DraggableMarker(QGraphicsEllipseItem):
    def __init__(self, x, y, radius=7, color=QColor(255, 0, 0), label="", real_coord=None):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.setPos(x, y)
        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.white, 1))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setZValue(100)
        self.real_coord = real_coord

        if label:
            txt = QGraphicsTextItem(label, self)
            txt.setDefaultTextColor(Qt.white)
            txt.setFont(QFont("Arial", 8, QFont.Bold))
            txt.setPos(radius + 2, -radius)

    def hoverEnterEvent(self, event):
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setPen(QPen(Qt.yellow, 2))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setPen(QPen(Qt.white, 1))

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.scene().removeItem(self)
            event.accept()
        else:
            super().mousePressEvent(event)

class ClickableAnchorMarker(QGraphicsEllipseItem):
    def __init__(self, n, rc, parent_view, *args):
        super().__init__(*args)
        self.name = n
        self.real_coord = rc
        self.parent_view = parent_view
        self.setAcceptHoverEvents(True)
        self.setToolTip(self.name)

    def hoverEnterEvent(self, e):
        self.setCursor(Qt.PointingHandCursor)

    def hoverLeaveEvent(self, e):
        self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, e):
        self.parent_view.main_app.set_active_anchor(self.name, self.real_coord, self)

class TutorialPitch(QGraphicsView):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)

        scale = 3
        self.w = 105 * scale
        self.h = 68 * scale

        self.scene.addRect(0, 0, self.w, self.h, QPen(Qt.white, 2), QBrush(QColor(34, 139, 34)))

        self.scene.addLine(self.w/2, 0, self.w/2, self.h, QPen(Qt.white, 2))
        r = 9.15 * scale
        self.scene.addEllipse(self.w/2 - r, self.h/2 - r, r*2, r*2, QPen(Qt.white, 2))

        pen_r = 9.15 * scale
        self.scene.addEllipse((11.0)*scale - pen_r, (34.0)*scale - pen_r, pen_r*2, pen_r*2, QPen(Qt.white, 2))
        self.scene.addEllipse((94.0)*scale - pen_r, (34.0)*scale - pen_r, pen_r*2, pen_r*2, QPen(Qt.white, 2))

        self.scene.addRect(0, (34 - 20.16)*scale, 16.5*scale, 40.32*scale, QPen(Qt.white, 2), QBrush(QColor(34, 139, 34)))
        self.scene.addRect(self.w - 16.5*scale, (34 - 20.16)*scale, 16.5*scale, 40.32*scale, QPen(Qt.white, 2), QBrush(QColor(34, 139, 34)))

        self.scene.addRect(0, (34 - 9.16)*scale, 5.5*scale, 18.32*scale, QPen(Qt.white, 2))
        self.scene.addRect(self.w - 5.5*scale, (34 - 9.16)*scale, 5.5*scale, 18.32*scale, QPen(Qt.white, 2))

        self.anchor_items = []
        for name, (rx, ry) in LANDMARKS.items():
            px, py = rx * scale, ry * scale
            cm = ClickableAnchorMarker(name, (rx, ry), self, -6, -6, 12, 12)
            cm.setPos(px, py)
            cm.setBrush(QBrush(Qt.yellow))
            self.scene.addItem(cm)
            self.anchor_items.append(cm)

        self.setFixedSize(self.w + 20, self.h + 20)

class VideoView(QGraphicsView):
    def __init__(self, scene, main_app, *args):
        super().__init__(scene, *args)
        self.main_app = main_app

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if isinstance(item, DraggableMarker):
            super().mousePressEvent(event)
            return

        if self.main_app.active_anchor_name and event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            marker = DraggableMarker(pos.x(), pos.y(), radius=8, color=QColor(255, 165, 0), 
                                     label=self.main_app.active_anchor_name[:3], 
                                     real_coord=self.main_app.active_anchor_coord)
            self.scene().addItem(marker)
            self.main_app.set_active_anchor(None, None, None)
            event.accept()
        else:
            super().mousePressEvent(event)

class UnifiedCalibrationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Football App - Full Feature Safe Mode")
        self.resize(1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.toolbar = QHBoxLayout()
        self.btn_load = QPushButton("1. Browse Video")
        self.btn_load.clicked.connect(self.load_video)
        self.btn_load.setStyleSheet("padding: 10px; font-weight: bold;")

        self.length_combo = QComboBox()
        self.length_combo.addItems(["Full Video (100%)", "Quarter (25%)", "Half (50%)", "Three Quarters (75%)"])
        self.length_combo.setStyleSheet("padding: 10px; font-weight: bold;")

        self.btn_confirm = QPushButton("2. Confirm Calibration (Run Tracker)")
        self.btn_confirm.clicked.connect(self.confirm_calibration)
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.setStyleSheet("padding: 10px; font-weight: bold;")

        self.toolbar.addWidget(self.btn_load)
        self.toolbar.addWidget(self.length_combo)
        self.toolbar.addWidget(self.btn_confirm)
        self.main_layout.addLayout(self.toolbar)

        self.split_layout = QHBoxLayout()
        self.main_layout.addLayout(self.split_layout)

        self.tabs = QTabWidget()
        self.split_layout.addWidget(self.tabs, stretch=3)

        self.sidebar_layout = QVBoxLayout()
        self.split_layout.addLayout(self.sidebar_layout, stretch=1)

        self.status_label = QLabel("Auto-calibration will run first. To add points manually:\n1. Click a yellow dot on the pitch below.\n2. Click the matching spot on the video frame.\n3. Right-click any marker to delete it.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        self.sidebar_layout.addWidget(self.status_label)

        self.tutorial_pitch = TutorialPitch(self)
        self.sidebar_layout.addWidget(self.tutorial_pitch)
        self.sidebar_layout.addStretch()

        self.progress_container = QWidget()
        self.progress_layout = QVBoxLayout(self.progress_container)
        self.progress_label = QLabel("Processing video...")
        self.progress_bar = QProgressBar()
        self.progress_layout.addWidget(self.progress_label)
        self.progress_layout.addWidget(self.progress_bar)
        self.progress_container.hide()
        self.main_layout.addWidget(self.progress_container)

        if PitchKeypointDetector:
            self.detector = PitchKeypointDetector()
        else:
            self.detector = None

        self._safe_image_refs = []
        self.current_video_path = None
        self.active_anchor_name = None
        self.active_anchor_coord = None

        self._roboflow_api_key = os.environ.get("ROBOFLOW_API_KEY")
        self._roboflow_model_id = "object-detection-traffic/1"
        config_path = os.path.join(os.path.dirname(__file__), "..", "pipeline", "config.json")
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
                if cfg.get("roboflow_model_id"):
                    self._roboflow_model_id = cfg["roboflow_model_id"]
        except Exception:
            pass

    def set_active_anchor(self, name, real_coord, marker_item):
        self.active_anchor_name = name
        self.active_anchor_coord = real_coord
        for m in self.tutorial_pitch.anchor_items:
            m.setBrush(QBrush(Qt.yellow))

        if marker_item:
            marker_item.setBrush(QBrush(Qt.red))
            self.status_label.setText(f"Active Anchor: {name}\nClick on the video frame to place it.")
        else:
            self.status_label.setText("Select an anchor from the pitch below to place manually.")

    def load_video(self):
        if not self.detector:
            QMessageBox.critical(self, "Error", "YOLO PitchKeypointDetector failed to load.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not file_path:
            return

        self.current_video_path = file_path
        self.btn_load.setText("Processing Models... Please wait")
        self.btn_load.setEnabled(False)
        self.btn_confirm.setEnabled(False)
        QApplication.processEvents() 

        self.tabs.clear()
        self._safe_image_refs.clear()

        cap = cv2.VideoCapture(file_path)
        indices = [0, 15, 30, 45, 60]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append((idx, frame))
        cap.release()

        detection_results = []
        for idx, frame in frames:
            kpts_dict = self.detector.detect(frame, conf_thresh=0.5)
            name_to_real = {v[0]: v[1] for v in self.detector.keypoint_mapping.values()}
            valid_kpts = {k: v for k, v in kpts_dict.items() if k in name_to_real}

            if len(valid_kpts) < 4 and self._roboflow_api_key:
                try:
                    from inference import get_model
                    model = get_model(model_id=self._roboflow_model_id, api_key=self._roboflow_api_key)
                    res = model.infer(frame)[0]
                except Exception:
                    pass

            detection_results.append((idx, frame, valid_kpts, name_to_real))

        for idx, frame, valid_kpts, name_to_real in detection_results:
            scene = QGraphicsScene()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._safe_image_refs.append(rgb_frame)

            h, w, ch = rgb_frame.shape
            bpl = ch * w

            img = QImage(rgb_frame.data, w, h, bpl, QImage.Format_RGB888).copy()
            pixmap_item = scene.addPixmap(QPixmap.fromImage(img))
            scene.setSceneRect(pixmap_item.boundingRect())

            for name, (px, py, conf) in valid_kpts.items():
                marker = DraggableMarker(px, py, radius=8, color=QColor(0, 255, 0), label=name[:3], real_coord=name_to_real[name])
                scene.addItem(marker)

            view = VideoView(scene, self)
            view._scene = scene
            view.setRenderHint(QPainter.Antialiasing)
            view._pixmap_item = pixmap_item
            view._scene_ref = scene 

            def _fit(v=view):
                v.fitInView(v._pixmap_item, Qt.KeepAspectRatio)

            view.resizeEvent = lambda e, v=view: (QGraphicsView.resizeEvent(v, e), _fit(v))
            QTimer.singleShot(0, _fit)

            self.tabs.addTab(view, f"Frame {idx}")

        self.btn_load.setText("1. Browse Video")
        self.btn_load.setEnabled(True)
        self.btn_confirm.setEnabled(True)
        self.set_active_anchor(None, None, None)

    def confirm_calibration(self):
        current_view = self.tabs.currentWidget()
        if not current_view:
            return

        scene = current_view._scene_ref
        pixel_pts = []
        real_pts = []
        for item in scene.items():
            if isinstance(item, DraggableMarker) and item.real_coord:
                pos = item.scenePos()
                pixel_pts.append([pos.x(), pos.y()])
                real_pts.append([item.real_coord[0], item.real_coord[1]])

        if len(pixel_pts) < 4:
            QMessageBox.warning(self, "Error", "Not enough points (minimum 4) to calculate Homography.")
            return

        try:
            H, _ = cv2.findHomography(np.array(pixel_pts, dtype=np.float32), np.array(real_pts, dtype=np.float32), cv2.RANSAC, 5.0)
            if H is None:
                QMessageBox.warning(self, "Error", "Failed to compute valid Homography from these points.")
                return
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Homography calculation failed: {e}")
            return

        if not PipelineWorker:
            QMessageBox.critical(self, "Error", "PipelineWorker not found!")
            return

        self.tabs.hide()
        for i in range(self.toolbar.count()):
            widget = self.toolbar.itemAt(i).widget()
            if widget:
                widget.hide()

        self.status_label.hide()
        self.tutorial_pitch.hide()
        self.progress_container.show()

        selected_text = self.length_combo.currentText()
        frame_fraction = 1.0
        if "Quarter" in selected_text and "Three" not in selected_text:
            frame_fraction = 0.25
        elif "Half" in selected_text:
            frame_fraction = 0.50
        elif "Three Quarters" in selected_text:
            frame_fraction = 0.75

        self.worker = PipelineWorker(self.current_video_path, H, frame_fraction)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.finished_successfully.connect(self.on_success)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    def update_progress(self, percent, text):
        self.progress_bar.setValue(percent)
        self.progress_label.setText(text)

    def on_success(self, csv_path, out_vid_path):
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "Success", f"Processing complete!\nVideo: {out_vid_path}")
        self.progress_label.setText("Done!")

    def on_error(self, err_msg):
        QMessageBox.critical(self, "Error", f"Processing failed: {err_msg}")
        self.progress_label.setText("Error!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UnifiedCalibrationApp()
    window.show()
    sys.exit(app.exec())
