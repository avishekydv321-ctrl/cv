"""PySide6 dark-themed desktop interface for VisionSuite.

Modern dashboard with:
  * compact single-page layout — every control reachable
  * scrollable sidebar (2-column AI module grid) + slidable splitter panels
  * source selector (webcam, IP/RTSP, video file, image)
  * settings page, analytics dashboard, notification center
  * full-screen mode (F11), toolbar, status bar, keyboard shortcuts
  * per-module toggles + base mode multi-select
"""
from __future__ import annotations

import logging
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot, QPropertyAnimation
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSlider, QSplitter, QStackedWidget,
    QStatusBar, QTextEdit, QVBoxLayout, QWidget, QSystemTrayIcon, QStyle,
    QTableWidget, QTableWidgetItem, QHeaderView, QStyleFactory
)

from app import FrameResult, VisionWorker
from camera import list_cameras
from config import AI_MODULES, FILTERS, MODES, RESOLUTIONS, SOURCE_TYPES, YOLO_MODELS, AppConfig
import database
import repositories
from face_db import FaceDB
from modules import get_module
from recorder import VideoRecorder, save_snapshot
from video_anpr import ANPRVideoWorker, SUPPORTED_VIDEO_EXT
from utils import OUTPUT_DIR, export_detections, timestamp

log = logging.getLogger("gui")

DARK_QSS = """
* { font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 12px; }
QMainWindow, QWidget { background: #0e1117; color: #e6edf3; }
QGroupBox {
  border: 1px solid #232b36; border-radius: 8px; margin-top: 12px; padding: 8px 8px 6px;
  background: #131922;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #7ee7c7; font-weight: 600; }
QPushButton {
  background: #1b2430; border: 1px solid #2b3543; border-radius: 6px; padding: 6px 10px; color: #e6edf3;
}
QPushButton:hover { background: #23303f; border-color: #3a94ff; }
QPushButton:disabled { color: #5a6472; background: #161c25; }
QPushButton#primary { background: #12a67c; border-color: #12a67c; color: #04140f; font-weight: 600; }
QPushButton#primary:hover { background: #16c493; }
QPushButton#danger { background: #b3372f; border-color: #b3372f; color: #fff; }
QComboBox, QTextEdit, QListWidget, QLineEdit {
  background: #0f1620; border: 1px solid #2b3543; border-radius: 6px; padding: 4px; selection-background-color: #12a67c;
}
QLineEdit { background: #0f1620; }
QSlider::groove:horizontal { height: 4px; background: #2b3543; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -5px 0; border-radius: 7px; background: #12a67c; }
QListWidget::item:selected { background: #12a67c; color: #04140f; }
QLabel#video { background: #05080c; border: 1px solid #232b36; border-radius: 10px; }
QLabel#metric { color: #7ee7c7; font-size: 18px; font-weight: 700; }
QLabel#sysmetric { color: #3a94ff; font-size: 13px; font-weight: 600; }
QStatusBar { background: #0b1017; color: #9fb0c3; border-top: 1px solid #232b36; }
QStackedWidget { background: #0e1117; }
QSplitter::handle { background: #1b2430; border-radius: 3px; }
QSplitter::handle:hover { background: #3a94ff; }
QScrollArea { background: transparent; border: none; }
QCheckBox { spacing: 6px; min-height: 18px; }
"""

LIGHT_QSS = """
* { font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 12px; }
QMainWindow, QWidget { background: #f8f9fa; color: #212529; }
QGroupBox {
  border: 1px solid #dee2e6; border-radius: 8px; margin-top: 12px; padding: 8px 8px 6px;
  background: #ffffff;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #12a67c; font-weight: 600; }
QPushButton {
  background: #f1f3f5; border: 1px solid #ced4da; border-radius: 6px; padding: 6px 10px; color: #212529;
}
QPushButton:hover { background: #e9ecef; border-color: #12a67c; }
QPushButton:disabled { color: #adb5bd; background: #f8f9fa; }
QPushButton#primary { background: #12a67c; border-color: #12a67c; color: #ffffff; font-weight: 600; }
QPushButton#primary:hover { background: #16c493; }
QPushButton#danger { background: #dc3545; border-color: #dc3545; color: #ffffff; }
QComboBox, QTextEdit, QListWidget, QLineEdit {
  background: #ffffff; border: 1px solid #ced4da; border-radius: 6px; padding: 4px; selection-background-color: #12a67c; selection-color: #ffffff;
}
QSlider::groove:horizontal { height: 4px; background: #dee2e6; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -5px 0; border-radius: 7px; background: #12a67c; }
QListWidget::item:selected { background: #12a67c; color: #ffffff; }
QLabel#video { background: #e9ecef; border: 1px solid #ced4da; border-radius: 10px; }
QLabel#metric { color: #12a67c; font-size: 18px; font-weight: 700; }
QLabel#sysmetric { color: #0d6efd; font-size: 13px; font-weight: 600; }
QStatusBar { background: #f1f3f5; color: #495057; border-top: 1px solid #dee2e6; }
QStackedWidget { background: #f8f9fa; }
QSplitter::handle { background: #dee2e6; border-radius: 3px; }
QSplitter::handle:hover { background: #12a67c; }
QScrollArea { background: transparent; border: none; }
QCheckBox { spacing: 6px; min-height: 18px; }
"""


class SettingsDialog(QDialog):
    """Settings page: source config, thresholds, performance, alerts, and preprocessors."""

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("VisionSuite — Settings")
        self.setMinimumWidth(460)
        self.setStyleSheet(DARK_QSS if config.theme == "dark" else LIGHT_QSS)
        layout = QVBoxLayout(self)

        source_box = QGroupBox("Media source")
        src = QVBoxLayout(source_box)
        self.source_combo = QComboBox()
        self.source_combo.addItems(SOURCE_TYPES)
        self.source_combo.setCurrentText(config.source)
        self.ip_edit = QLineEdit(config.ip_url)
        self.ip_edit.setPlaceholderText("rtsp://user:pass@host:554/stream")
        self.media_edit = QLineEdit(config.media_path)
        self.media_edit.setPlaceholderText("Path to video file or image")
        pick = QPushButton("Browse…")
        pick.clicked.connect(self._browse)
        media_row = QHBoxLayout()
        media_row.addWidget(self.media_edit)
        media_row.addWidget(pick)
        src.addWidget(QLabel("Source type"))
        src.addWidget(self.source_combo)
        src.addWidget(QLabel("IP / RTSP URL"))
        src.addWidget(self.ip_edit)
        src.addWidget(QLabel("Media file"))
        src.addLayout(media_row)
        layout.addWidget(source_box)

        # Performance & Settings
        perf_box = QGroupBox("Performance & Settings")
        perf = QVBoxLayout(perf_box)
        self.monitor_check = QCheckBox("System monitor (CPU / RAM / GPU)")
        self.monitor_check.setChecked(config.system_monitor)
        self.analytics_check = QCheckBox("Analytics dashboard & history")
        self.analytics_check.setChecked(config.analytics_enabled)
        self.fps_check = QCheckBox("Show FPS overlay")
        self.fps_check.setChecked(config.show_fps)
        self.attendance_check = QCheckBox("Face Recognition Attendance Mode")
        self.attendance_check.setChecked(config.attendance_mode)
        self.tracking_check = QCheckBox("Native Object Tracking")
        self.tracking_check.setChecked(config.tracking_enabled)
        self.face_history_check = QCheckBox("Store face recognition history")
        self.face_history_check.setChecked(config.face_history_enabled)
        self.face_threshold_label = QLabel(
            f"Face match threshold: {config.face_similarity_threshold:.2f}"
        )
        self.face_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.face_threshold_slider.setRange(25, 75)
        self.face_threshold_slider.setValue(round(config.face_similarity_threshold * 100))
        self.face_threshold_slider.valueChanged.connect(
            lambda value: self.face_threshold_label.setText(
                f"Face match threshold: {value / 100:.2f}"
            )
        )

        for c in (self.monitor_check, self.analytics_check, self.fps_check,
                  self.attendance_check, self.tracking_check, self.face_history_check,
                  self.face_threshold_label, self.face_threshold_slider):
            perf.addWidget(c)
        layout.addWidget(perf_box)

        # Image Preprocessing Enhancement Box
        prep_box = QGroupBox("Low-Quality Image Enhancement")
        prep = QGridLayout(prep_box)
        self.chk_sharp = QCheckBox("Sharpening")
        self.chk_sharp.setChecked(config.sharpening)
        self.chk_denoise = QCheckBox("Denoising")
        self.chk_denoise.setChecked(config.denoising)
        self.chk_clahe = QCheckBox("CLAHE Contrast")
        self.chk_clahe.setChecked(config.clahe)
        self.chk_exposure = QCheckBox("Auto Exposure")
        self.chk_exposure.setChecked(config.exposure_correction)
        self.chk_wb = QCheckBox("White Balance")
        self.chk_wb.setChecked(config.white_balance)
        self.chk_stab = QCheckBox("Stabilization")
        self.chk_stab.setChecked(config.stabilization)

        prep.addWidget(self.chk_sharp, 0, 0)
        prep.addWidget(self.chk_denoise, 0, 1)
        prep.addWidget(self.chk_clahe, 1, 0)
        prep.addWidget(self.chk_exposure, 1, 1)
        prep.addWidget(self.chk_wb, 2, 0)
        prep.addWidget(self.chk_stab, 2, 1)
        layout.addWidget(prep_box)

        # Alerts and notifications setup
        alert_box = QGroupBox("Smart Alerts & LPR Setup")
        alert_lay = QVBoxLayout(alert_box)
        self.chk_tray = QCheckBox("Desktop Notifications (System Tray)")
        self.chk_tray.setChecked(config.desktop_notifications)
        self.chk_sound = QCheckBox("Sound Alarm Beep")
        self.chk_sound.setChecked(config.sound_alerts)

        alert_lay.addWidget(self.chk_tray)
        alert_lay.addWidget(self.chk_sound)

        alert_lay.addWidget(QLabel("Blacklisted Plates (comma-separated)"))
        self.blacklist_edit = QLineEdit(config.blacklist_plates)
        self.blacklist_edit.setPlaceholderText("e.g. AB123CD, XYZ890")
        alert_lay.addWidget(self.blacklist_edit)

        self.chk_email = QCheckBox("Enable Email Alerts")
        self.chk_email.setChecked(config.email_alerts_enabled)
        alert_lay.addWidget(self.chk_email)

        # SMTP details
        smtp_lay = QHBoxLayout()
        self.smtp_server = QLineEdit(config.email_smtp_server)
        self.smtp_server.setPlaceholderText("SMTP server")
        self.smtp_port = QLineEdit(str(config.email_smtp_port))
        self.smtp_port.setPlaceholderText("Port")
        self.smtp_port.setFixedWidth(50)
        smtp_lay.addWidget(self.smtp_server)
        smtp_lay.addWidget(self.smtp_port)
        alert_lay.addLayout(smtp_lay)

        email_cred_lay = QHBoxLayout()
        self.email_sender = QLineEdit(config.email_sender)
        self.email_sender.setPlaceholderText("Sender email")
        self.email_recipient = QLineEdit(config.email_recipient)
        self.email_recipient.setPlaceholderText("Recipient email")
        email_cred_lay.addWidget(self.email_sender)
        email_cred_lay.addWidget(self.email_recipient)
        alert_lay.addLayout(email_cred_lay)

        self.email_pass = QLineEdit(config.email_password)
        self.email_pass.setPlaceholderText("Email app password")
        self.email_pass.setEchoMode(QLineEdit.EchoMode.Password)
        alert_lay.addWidget(self.email_pass)

        layout.addWidget(alert_box)

        buttons = QHBoxLayout()
        save = QPushButton("Save")
        save.setObjectName("primary")
        save.clicked.connect(self._apply)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select media", "", "Media (*.mp4 *.avi *.mov *.mkv *.png *.jpg *.jpeg)"
        )
        if path:
            self.media_edit.setText(path)

    def _apply(self) -> None:
        self.config.source = self.source_combo.currentText()
        if self.parent() is not None and hasattr(self.parent(), "source_combo"):
            self.parent().source_combo.setCurrentText(self.config.source)
        self.config.ip_url = self.ip_edit.text().strip()
        self.config.media_path = self.media_edit.text().strip()
        self.config.system_monitor = self.monitor_check.isChecked()
        self.config.analytics_enabled = self.analytics_check.isChecked()
        self.config.show_fps = self.fps_check.isChecked()
        self.config.attendance_mode = self.attendance_check.isChecked()
        self.config.tracking_enabled = self.tracking_check.isChecked()
        self.config.face_history_enabled = self.face_history_check.isChecked()
        self.config.face_similarity_threshold = self.face_threshold_slider.value() / 100.0

        # Preprocessing enhancement parameters
        self.config.sharpening = self.chk_sharp.isChecked()
        self.config.denoising = self.chk_denoise.isChecked()
        self.config.clahe = self.chk_clahe.isChecked()
        self.config.exposure_correction = self.chk_exposure.isChecked()
        self.config.white_balance = self.chk_wb.isChecked()
        self.config.stabilization = self.chk_stab.isChecked()

        # Alerts config
        self.config.desktop_notifications = self.chk_tray.isChecked()
        self.config.sound_alerts = self.chk_sound.isChecked()
        self.config.blacklist_plates = self.blacklist_edit.text().strip()
        self.config.email_alerts_enabled = self.chk_email.isChecked()
        self.config.email_smtp_server = self.smtp_server.text().strip()
        try:
            self.config.email_smtp_port = int(self.smtp_port.text().strip())
        except ValueError:
            pass
        self.config.email_sender = self.email_sender.text().strip()
        self.config.email_recipient = self.email_recipient.text().strip()
        self.config.email_password = self.email_pass.text().strip()

        self.accept()


class DatabaseSettingsDialog(QDialog):
    """Database settings page: PostgreSQL status, backup/restore, setup help."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VisionSuite — Local Database")
        self.setMinimumWidth(560)
        self.setStyleSheet(DARK_QSS if (parent is not None and parent.config.theme == "dark") else LIGHT_QSS)
        layout = QVBoxLayout(self)

        # Status card.
        status_box = QGroupBox("PostgreSQL status")
        st = QVBoxLayout(status_box)
        self.status_label = QLabel("Checking…")
        self.status_label.setObjectName("metric")
        self.status_label.setWordWrap(True)
        st.addWidget(self.status_label)
        self.tables_label = QLabel("")
        self.tables_label.setWordWrap(True)
        st.addWidget(self.tables_label)
        layout.addWidget(status_box)

        # Connection configuration (read-only display, no password shown).
        cfg_box = QGroupBox("Connection (from .env)")
        cfg_lay = QVBoxLayout(cfg_box)
        cfg = database.get_db_config()
        self.host_edit = QLineEdit(cfg["host"])
        self.port_edit = QLineEdit(str(cfg["port"]))
        self.db_edit = QLineEdit(cfg["db"])
        self.user_edit = QLineEdit(cfg["user"])
        self.password_edit = QLineEdit("")
        self.password_edit.setPlaceholderText("●●●● (never shown)")
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        row = QHBoxLayout()
        row.addWidget(QLabel("Host"))
        row.addWidget(self.host_edit, 1)
        row.addWidget(QLabel("Port"))
        row.addWidget(self.port_edit, 1)
        cfg_lay.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Database"))
        row2.addWidget(self.db_edit, 1)
        row2.addWidget(QLabel("User"))
        row2.addWidget(self.user_edit, 1)
        cfg_lay.addLayout(row2)
        cfg_lay.addWidget(QLabel("Password"))
        cfg_lay.addWidget(self.password_edit)
        layout.addWidget(cfg_box)

        # Actions.
        actions = QHBoxLayout()
        self.btn_test = QPushButton("Test Connection")
        self.btn_test.clicked.connect(self.test_connection)
        self.btn_refresh = QPushButton("Refresh Status")
        self.btn_refresh.clicked.connect(self.refresh_status)
        self.btn_backup = QPushButton("Backup…")
        self.btn_backup.clicked.connect(self.backup)
        self.btn_restore = QPushButton("Restore…")
        self.btn_restore.clicked.connect(self.restore)
        for b in (self.btn_test, self.btn_refresh, self.btn_backup, self.btn_restore):
            actions.addWidget(b)
        layout.addLayout(actions)

        # Setup instructions when unavailable.
        self.help_label = QLabel(
            "Setup: see SETUP_POSTGRES.md\n"
            "1. Install PostgreSQL (brew install postgresql@16)\n"
            "2. Create a database + user\n"
            "3. Copy .env.example → .env and fill in POSTGRES_PASSWORD"
        )
        self.help_label.setWordWrap(True)
        self.help_label.setStyleSheet("color:#9fb0c3;")
        layout.addWidget(self.help_label)

        close = QPushButton("Close")
        close.setObjectName("primary")
        close.clicked.connect(self.accept)
        layout.addWidget(close)

        self.refresh_status()

    def refresh_status(self) -> None:
        st = database.get_pg_status()
        if st.get("connected"):
            self.status_label.setText("● PostgreSQL Connected")
            self.status_label.setStyleSheet("color:#12a67c; font-size:18px; font-weight:700;")
            tables = database.list_tables()
            self.tables_label.setText("Tables: " + (", ".join(tables) if tables else "(none yet)"))
        else:
            self.status_label.setText("● Local PostgreSQL is unavailable. Keeping app usable.")
            err = st.get("error") or ""
            self.status_label.setStyleSheet("color:#e0a53a; font-size:15px; font-weight:600;")
            self.tables_label.setText(("Detail: " + err) if err else "")

    def test_connection(self) -> None:
        ok = database.probe_connection().get("connected")
        if ok:
            QMessageBox.information(self, "Test Connection", "Connected to local PostgreSQL.")
        else:
            QMessageBox.warning(self, "Test Connection", "Could not reach local PostgreSQL.\nSee SETUP_POSTGRES.md")

    def backup(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Backup Database", "visionsuite.backup", "PostgreSQL Backup (*.dump *.backup)")
        if not path:
            return
        err = database.backup_postgres(Path(path))
        if err:
            QMessageBox.critical(self, "Backup", f"Backup failed:\n{err}")
        else:
            QMessageBox.information(self, "Backup", f"Backup saved to:\n{path}")

    def restore(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Restore Database", "", "PostgreSQL Backup (*.dump *.backup)")
        if not path:
            return
        if QMessageBox.question(self, "Restore", "This will replace the current database. Continue?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        err = database.restore_postgres(Path(path))
        if err:
            QMessageBox.critical(self, "Restore", f"Restore failed:\n{err}")
        else:
            QMessageBox.information(self, "Restore", "Database restored successfully.")
            self.refresh_status()


class AnalyticsDialog(QDialog):
    """Analytics dashboard: history, confidence stats, export buttons."""

    def __init__(self, worker, parent=None) -> None:
        super().__init__(parent)
        self.worker = worker
        self.setWindowTitle("VisionSuite — Analytics")
        self.setMinimumSize(540, 500)
        self.setStyleSheet(DARK_QSS)
        layout = QVBoxLayout(self)

        self.summary = QLabel("No session data yet.")
        self.summary.setWordWrap(True)
        self.summary.setObjectName("metric")
        layout.addWidget(self.summary)

        self.hist_list = QListWidget()
        layout.addWidget(QLabel("Detection history"))
        layout.addWidget(self.hist_list)

        btn_row = QHBoxLayout()
        csv_btn = QPushButton("Export CSV")
        csv_btn.clicked.connect(self.export_csv)
        pdf_btn = QPushButton("Export PDF report")
        pdf_btn.clicked.connect(self.export_pdf)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        btn_row.addWidget(csv_btn)
        btn_row.addWidget(pdf_btn)
        btn_row.addWidget(refresh)
        layout.addLayout(btn_row)

    def refresh(self) -> None:
        if self.worker is None:
            return
        stats = self.worker.analytics.session_stats()
        conf = self.worker.analytics.confidence_stats()
        counts = self.worker.analytics.label_counts()
        self.summary.setText(
            f"Duration: {stats['duration_s']}s   •   "
            f"Detections: {stats['total_detections']}   •   "
            f"Unique labels: {stats['unique_labels']}\n"
            f"Confidence — min: {conf['min']:.2%}  max: {conf['max']:.2%}  "
            f"mean: {conf['mean']:.2%}"
        )
        self.hist_list.clear()
        for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            self.hist_list.addItem(f"{label}  —  {n}")
        for row in self.worker.analytics.recent(40):
            self.hist_list.addItem(
                f"[{row['source']}] {row['label']}  {row['confidence']:.0%}"
            )

    def export_csv(self) -> None:
        if self.worker is None:
            return
        path = self.worker.analytics.export_csv(OUTPUT_DIR)
        self.statusBarMsg(f"Exported CSV → {path.name}")

    def export_pdf(self) -> None:
        if self.worker is None:
            return
        path = self.worker.analytics.export_pdf(OUTPUT_DIR)
        self.statusBarMsg(f"Exported PDF report → {path.name}")

    def statusBarMsg(self, msg: str) -> None:
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(msg, 4000)


class FaceEnrollmentWorker(QThread):
    def __init__(self, db_path: Path, name: str, unique_id: str,
                 image_paths: list[str], captured_frame: np.ndarray | None) -> None:
        super().__init__()
        self.db_path = db_path
        self.name = name
        self.unique_id = unique_id
        self.image_paths = image_paths
        self.captured_frame = captured_frame
        self.success_message = ""
        self.error_message = ""

    def run(self) -> None:
        db = None
        try:
            from modules.face import FaceRecognition

            db = FaceDB(self.db_path)
            module = FaceRecognition(db)
            if not module.available:
                raise RuntimeError(module._model_error or "Face recognition models are unavailable.")
            embeddings = []
            for image_path in self.image_paths:
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not read image: {image_path}")
                embedding = module.embed_face(image)
                if embedding is None:
                    raise ValueError(f"No usable face found in {Path(image_path).name}.")
                embeddings.append(embedding)
            if self.captured_frame is not None:
                embedding = module.embed_face(self.captured_frame)
                if embedding is None:
                    raise ValueError("No usable face found in the captured camera frame.")
                embeddings.append(embedding)
            if not embeddings:
                raise ValueError("Select at least one face image or capture a live frame.")
            db.register_person(
                self.name,
                self.unique_id,
                embeddings,
                embedding_model="opencv_sface_2021dec",
            )
            self.success_message = (
                f"Registered {self.name} ({self.unique_id}) with {len(embeddings)} sample(s)."
            )
        except Exception as exc:
            self.error_message = str(exc)
        finally:
            if db is not None:
                db.close()


class FaceEnrollmentDialog(QDialog):
    """Consent-based enrollment; images are embedded and not retained."""

    def __init__(self, db: FaceDB, captured_frame: np.ndarray | None, parent=None) -> None:
        super().__init__(parent)
        self.db = db
        self.captured_frame = captured_frame.copy() if captured_frame is not None else None
        self.image_paths: list[str] = []
        self.worker: FaceEnrollmentWorker | None = None
        self.setWindowTitle("Register Individual Face")
        self.setMinimumWidth(440)
        self.setStyleSheet(DARK_QSS if parent.config.theme == "dark" else LIGHT_QSS)
        layout = QVBoxLayout(self)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Full name")
        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("Unique ID (employee/student/etc.)")
        layout.addWidget(QLabel("Name"))
        layout.addWidget(self.name_edit)
        layout.addWidget(QLabel("Unique ID"))
        layout.addWidget(self.id_edit)

        self.images_label = QLabel("No uploaded images")
        self.images_label.setWordWrap(True)
        choose = QPushButton("Choose face images…")
        choose.clicked.connect(self.choose_images)
        layout.addWidget(choose)
        layout.addWidget(self.images_label)
        capture = QPushButton("Capture current camera frame")
        capture.clicked.connect(self.capture_current_frame)
        layout.addWidget(capture)
        capture_status = "Live frame ready" if self.captured_frame is not None else "No live frame captured"
        self.capture_label = QLabel(capture_status)
        self.capture_label.setWordWrap(True)
        layout.addWidget(self.capture_label)

        self.consent_check = QCheckBox(
            "I am authorized to enroll this person and have their explicit consent "
            "to process these biometric images."
        )
        layout.addWidget(self.consent_check)
        buttons = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.enroll_button = QPushButton("Enroll")
        self.enroll_button.setObjectName("primary")
        self.enroll_button.clicked.connect(self.enroll)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.enroll_button)
        layout.addLayout(buttons)

    def choose_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select face images", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if paths:
            self.image_paths = paths
            self.images_label.setText(f"{len(paths)} image(s) selected")

    def capture_current_frame(self) -> None:
        frame = getattr(self.parent(), "last_frame", None)
        if frame is None:
            QMessageBox.warning(self, "Capture face", "Start a camera or video source first.")
            return
        self.captured_frame = frame.copy()
        self.capture_label.setText("Captured frame ready for enrollment")

    def enroll(self) -> None:
        name, unique_id = self.name_edit.text().strip(), self.id_edit.text().strip()
        if not name or not unique_id:
            QMessageBox.warning(self, "Registration", "Enter both a name and a unique ID.")
            return
        if not self.consent_check.isChecked():
            QMessageBox.warning(self, "Consent required", "Confirm enrollment authority and explicit consent.")
            return
        if not self.image_paths and self.captured_frame is None:
            QMessageBox.warning(self, "Images required", "Choose face images or start the camera before enrollment.")
            return
        self.enroll_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.worker = FaceEnrollmentWorker(
            self.db.path, name, unique_id, list(self.image_paths), self.captured_frame
        )
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    @Slot()
    def on_worker_finished(self) -> None:
        if self.worker is None:
            return
        if self.worker.error_message:
            self.enroll_button.setEnabled(True)
            self.cancel_button.setEnabled(True)
            QMessageBox.critical(self, "Registration failed", self.worker.error_message)
            return
        message = self.worker.success_message
        if self.parent() is not None:
            self.parent().statusBar().showMessage(message, 6000)
        QMessageBox.information(self, "Registration complete", message)
        self.accept()

    def reject(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self.worker is not None and self.worker.isRunning():
            event.ignore()
            return
        super().closeEvent(event)


class FaceDBManagerDialog(QDialog):
    """Interactive management tool for Face Identity Database."""

    def __init__(self, db: FaceDB, parent=None) -> None:
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Face Database Manager")
        self.resize(400, 500)
        self.setStyleSheet(DARK_QSS if parent.config.theme == "dark" else LIGHT_QSS)
        layout = QVBoxLayout(self)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search registered faces...")
        self.search_edit.textChanged.connect(self.load_identities)
        layout.addWidget(self.search_edit)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        self.btn_history = QPushButton("Recognition History")
        self.btn_history.clicked.connect(self.show_history)
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)

        btn_layout.addWidget(self.btn_history)
        btn_layout.addWidget(self.btn_delete)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

        self.load_identities()

    def load_identities(self) -> None:
        self.list_widget.clear()
        query = self.search_edit.text().strip().lower()
        for record in self.db.identity_records(query):
            item = QListWidgetItem(
                f"{record['name']}  ·  {record['unique_id']}  ·  {record['sample_count']} samples"
            )
            item.setData(Qt.ItemDataRole.UserRole, record["name"])
            self.list_widget.addItem(item)

    def delete_selected(self) -> None:
        item = self.list_widget.currentItem()
        if item:
            name = item.data(Qt.ItemDataRole.UserRole)
            reply = QMessageBox.question(
                self, "Delete Face", f"Are you sure you want to delete '{name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.db.remove(name)
                self.load_identities()

    def show_history(self) -> None:
        lines = [
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(row['recognized_at']))}  "
            f"{row['name']}  {row['confidence']:.0%}"
            for row in self.db.history(100)
        ]
        QMessageBox.information(
            self, "Recognition History", "\n".join(lines) if lines else "No history recorded."
        )


class PlateLogDialog(QDialog):
    """View and search history logs of license plate recognition events."""

    def __init__(self, db: FaceDB, parent=None) -> None:
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("License Plate Recognition Log")
        self.resize(720, 500)
        self.setStyleSheet(DARK_QSS if parent.config.theme == "dark" else LIGHT_QSS)
        layout = QVBoxLayout(self)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search plate numbers...")
        self.search_edit.textChanged.connect(self.load_plates)
        layout.addWidget(self.search_edit)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Plate Number", "Confidence", "Camera / Source", "Timestamp"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemDoubleClicked.connect(self.show_snapshot)
        layout.addWidget(self.table)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        layout.addWidget(self.btn_close)

        self.records = []
        self.load_plates()

    def load_plates(self) -> None:
        self.table.setRowCount(0)
        query = self.search_edit.text().strip()
        self.records = self.db.get_plate_history(query)
        self.table.setRowCount(len(self.records))
        for idx, row in enumerate(self.records):
            self.table.setItem(idx, 0, QTableWidgetItem(row["plate"]))
            self.table.setItem(idx, 1, QTableWidgetItem(f"{row['confidence']:.0%}"))
            self.table.setItem(idx, 2, QTableWidgetItem(row["camera_id"]))
            local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["recognized_at"]))
            self.table.setItem(idx, 3, QTableWidgetItem(local_time))

    def show_snapshot(self, item: QTableWidgetItem) -> None:
        row_idx = item.row()
        if row_idx < len(self.records):
            path_str = self.records[row_idx]["snapshot_path"]
            if path_str and Path(path_str).exists():
                dialog = QDialog(self)
                dialog.setWindowTitle(f"Plate Snapshot: {self.table.item(row_idx, 0).text()}")
                dlg_layout = QVBoxLayout(dialog)
                lbl = QLabel()
                pix = QPixmap(path_str).scaled(
                    600, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
                lbl.setPixmap(pix)
                dlg_layout.addWidget(lbl)
                dialog.exec()
            else:
                QMessageBox.information(self, "No Snapshot", "No snapshot file found for this detection.")


class BatchWorker(QThread):
    """Background processor for batch processing a folder of image files."""

    progress = Signal(int, int, str)  # current, total, filename
    finished = Signal(str)            # path to output CSV report

    def __init__(self, folder: Path, config: AppConfig) -> None:
        super().__init__()
        self.folder = folder
        self.config = config

    def run(self) -> None:
        from utils import OUTPUT_DIR, timestamp
        import csv
        from app import VisionWorker

        image_files = [p for p in self.folder.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        if not image_files:
            self.finished.emit("")
            return

        worker = VisionWorker(self.config)
        csv_rows = []

        for idx, img_path in enumerate(image_files):
            self.progress.emit(idx + 1, len(image_files), img_path.name)
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            result = worker._process(frame)

            # Save processed image outputs
            out_dir = OUTPUT_DIR / "batch_processed"
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / img_path.name), result.frame)

            for det in result.detections:
                row = det.as_row(idx)
                row["filename"] = img_path.name
                csv_rows.append(row)

        report_path = OUTPUT_DIR / f"batch_report_{timestamp()}.csv"
        fields = ["frame", "filename", "time", "source", "label", "confidence", "x1", "y1", "x2", "y2"]
        try:
            with report_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(csv_rows)
            self.finished.emit(str(report_path))
        except Exception:
            self.finished.emit("")


class BatchProcessDialog(QDialog):
    """Progress tracker UI for batch folder processing runs."""

    def __init__(self, folder: Path, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Processing Folder")
        self.setMinimumWidth(380)
        self.setStyleSheet(DARK_QSS if parent.config.theme == "dark" else LIGHT_QSS)
        layout = QVBoxLayout(self)

        self.lbl_status = QLabel("Initializing batch run...")
        layout.addWidget(self.lbl_status)

        from PySide6.QtWidgets import QProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel_processing)
        layout.addWidget(self.btn_cancel)

        self.worker = BatchWorker(folder, config)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, current: int, total: int, filename: str) -> None:
        pct = int((current / total) * 100)
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(f"Processing ({current}/{total}): {filename}")

    def on_finished(self, report_path: str) -> None:
        if report_path:
            QMessageBox.information(
                self, "Batch Processing Complete",
                f"Batch run completed successfully!\nResults saved to output/batch_processed/\nReport CSV: {report_path}"
            )
        else:
            QMessageBox.warning(self, "Batch Run Error", "No image files found or batch process encountered an error.")
        self.accept()

    def cancel_processing(self) -> None:
        if self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            QMessageBox.warning(self, "Batch Cancelled", "Batch run was terminated.")
        self.reject()


class MainWindow(QMainWindow):
    """Main application window: compact single-page dashboard."""

    request_stop = Signal()
    request_pause = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VisionSuite — Real-Time Computer Vision Studio")
        self.resize(1500, 920)

        self.config = AppConfig()
        self.setStyleSheet(DARK_QSS if self.config.theme == "dark" else LIGHT_QSS)

        self.recorder = VideoRecorder()
        self.worker: VisionWorker | None = None
        self.thread: QThread | None = None
        self.last_frame: np.ndarray | None = None
        self.paused = False
        self.face_db = FaceDB()
        self.sidebar_collapsed = False

        # System Tray setup
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        self.tray_icon.show()

        self.setAcceptDrops(True)

        self._build_ui()
        self._build_menu()
        self._refresh_cameras()

    # -- construction -------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Slidable dashboard: a horizontal QSplitter lets the user drag the
        # sidebar, video stage and results panel to any width.
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(6)
        self.splitter.setChildrenCollapsible(False)

        sidebar = self._sidebar()
        stage = self._stage()
        results = self._results_panel()

        self.splitter.addWidget(sidebar)
        self.splitter.addWidget(stage)
        self.splitter.addWidget(results)
        self.splitter.setStretchFactor(0, 0)   # sidebar fixed-ish
        self.splitter.setStretchFactor(1, 1)   # stage expands
        self.splitter.setStretchFactor(2, 0)   # results panel
        self.splitter.setSizes([360, 820, 320])
        root.addWidget(self.splitter, 1)

        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready — press Ctrl+R to start (Ctrl+Return for fullscreen)")

        # Periodic system-monitor refresh.
        self._sys_timer = QTimer(self)
        self._sys_timer.timeout.connect(self._update_system)
        self._sys_timer.start(1000)

    def _sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(300)
        self.sidebar_panel = panel

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(4)

        # Title row with collapse toggle and theme switch.
        title_row = QHBoxLayout()
        title = QLabel("VISIONSUITE")
        title.setStyleSheet("font-size:16px; font-weight:800; letter-spacing:2px; color:#12a67c;")
        self.btn_theme = QPushButton("☀️" if self.config.theme == "dark" else "🌙")
        self.btn_theme.setFixedWidth(28)
        self.btn_theme.setToolTip("Toggle Light/Dark Theme")
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.btn_collapse = QPushButton("«")
        self.btn_collapse.setFixedWidth(28)
        self.btn_collapse.clicked.connect(self.toggle_sidebar)
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.btn_theme)
        title_row.addWidget(self.btn_collapse)
        outer.addLayout(title_row)

        # Scrollable content area so EVERY control stays reachable on the
        # single-page dashboard.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 4, 0)

        # -- AI modules (compact 2-column grid) -------------------------
        ai_box = QGroupBox("AI modules")
        ai_grid = QGridLayout(ai_box)
        ai_grid.setSpacing(2)
        self.module_checks: dict[str, QCheckBox] = {}
        keys = list(AI_MODULES.keys())
        cols = 2
        for i, key in enumerate(keys):
            check = QCheckBox(AI_MODULES[key])
            check.setChecked(key in self.config.enabled_modules)
            check.stateChanged.connect(self._make_module_toggle(key))
            self.module_checks[key] = check
            ai_grid.addWidget(check, i // cols, i % cols)
        layout.addWidget(ai_box)

        # -- Base modes (compact 2-column grid) -------------------------
        mode_box = QGroupBox("Mode")
        mode_grid = QGridLayout(mode_box)
        mode_grid.setSpacing(2)
        self.mode_checks: dict[str, QCheckBox] = {}
        for i, mode in enumerate(MODES):
            check = QCheckBox(mode)
            check.setChecked(mode in self.config.enabled_modes)
            check.stateChanged.connect(self._make_mode_toggle(mode))
            self.mode_checks[mode] = check
            mode_grid.addWidget(check, i // 2, i % 2)
        layout.addWidget(mode_box)

        # -- Source ------------------------------------------------------
        source_box = QGroupBox("Source")
        source_layout = QVBoxLayout(source_box)
        src_row = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems(SOURCE_TYPES)
        self.source_combo.setCurrentText(self.config.source)
        self.camera_combo = QComboBox()
        src_row.addWidget(QLabel("Type"))
        src_row.addWidget(self.source_combo, 1)
        src_row.addWidget(self.camera_combo, 1)
        source_layout.addLayout(src_row)
        res_row = QHBoxLayout()
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(RESOLUTIONS.keys())
        self.resolution_combo.setCurrentText("1280 x 720 (HD)")
        res_row.addWidget(QLabel("Resolution"))
        res_row.addWidget(self.resolution_combo, 1)
        rescan = QPushButton("Rescan")
        rescan.setFixedWidth(64)
        rescan.clicked.connect(self._refresh_cameras)
        res_row.addWidget(rescan)
        source_layout.addLayout(res_row)
        layout.addWidget(source_box)

        # -- Detection settings ------------------------------------------
        settings_box = QGroupBox("Detection settings")
        settings = QVBoxLayout(settings_box)
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(YOLO_MODELS.keys())
        self.model_combo.currentTextChanged.connect(
            lambda text: setattr(self.config, "yolo_model", YOLO_MODELS[text])
        )
        model_row.addWidget(QLabel("YOLO"))
        model_row.addWidget(self.model_combo, 1)
        settings.addLayout(model_row)

        self.conf_label = QLabel("Confidence: 0.45")
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(45)
        self.conf_slider.valueChanged.connect(self._on_confidence)
        settings.addWidget(self.conf_label)
        settings.addWidget(self.conf_slider)

        filter_row = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(FILTERS)
        self.filter_combo.currentTextChanged.connect(
            lambda text: setattr(self.config, "image_filter", text)
        )
        filter_row.addWidget(QLabel("Filter"))
        filter_row.addWidget(self.filter_combo, 1)
        settings.addLayout(filter_row)

        toggles = QHBoxLayout()
        self.detect_check = QCheckBox("Detect")
        self.detect_check.setChecked(True)
        self.detect_check.toggled.connect(lambda v: setattr(self.config, "detection_enabled", v))
        self.mirror_check = QCheckBox("Mirror")
        self.mirror_check.setChecked(True)
        self.mirror_check.toggled.connect(lambda v: setattr(self.config, "mirror", v))
        self.csv_check = QCheckBox("CSV")
        self.csv_check.toggled.connect(lambda v: setattr(self.config, "record_csv", v))
        for check in (self.detect_check, self.mirror_check, self.csv_check):
            toggles.addWidget(check)
        settings.addLayout(toggles)
        layout.addWidget(settings_box)

        # -- Quick actions ----------------------------------------------
        quick = QHBoxLayout()
        settings_btn = QPushButton("⚙ Settings")
        settings_btn.clicked.connect(self.open_settings)
        analytics_btn = QPushButton("📊 Analytics")
        analytics_btn.clicked.connect(self.open_analytics)
        db_btn = QPushButton("🗄 Database")
        db_btn.clicked.connect(self.open_database_settings)
        quick.addWidget(settings_btn)
        quick.addWidget(analytics_btn)
        quick.addWidget(db_btn)
        layout.addLayout(quick)

        # PostgreSQL status indicator.
        self.db_status_label = QLabel("PostgreSQL: …")
        self.db_status_label.setObjectName("sysmetric")
        self.db_status_label.setWordWrap(True)
        layout.addWidget(self.db_status_label)

        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        return panel

    def _stage(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_capture = QPushButton("📷  Capture")
        self.btn_capture.clicked.connect(self.capture_image)
        self.btn_record = QPushButton("⏺  Record")
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_fullscreen = QPushButton("⛶  Fullscreen")
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        for button in (self.btn_start, self.btn_pause, self.btn_stop,
                       self.btn_capture, self.btn_record, self.btn_fullscreen):
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        self.fps_label = QLabel("0.0 FPS")
        self.fps_label.setObjectName("metric")
        toolbar.addWidget(self.fps_label)
        layout.addLayout(toolbar)

        self.video = QLabel("Camera is off")
        self.video.setObjectName("video")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(480, 380)
        layout.addWidget(self.video, 1)

        # Playback controls bar
        self.play_controls = QWidget()
        pc_layout = QHBoxLayout(self.play_controls)
        pc_layout.setContentsMargins(0, 0, 0, 0)

        # Seek slider
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.sliderMoved.connect(self.on_seek)

        # Frame / Time label
        self.lbl_time = QLabel("00:00 / 00:00")

        # Playback Speed selector
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.setFixedWidth(60)
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)

        # Frame Step Buttons
        self.btn_prev_frame = QPushButton("⏮")
        self.btn_prev_frame.setToolTip("Previous Frame")
        self.btn_prev_frame.setFixedWidth(30)
        self.btn_prev_frame.clicked.connect(self.prev_frame)
        self.btn_next_frame = QPushButton("⏭")
        self.btn_next_frame.setToolTip("Next Frame")
        self.btn_next_frame.setFixedWidth(30)
        self.btn_next_frame.clicked.connect(self.next_frame)

        pc_layout.addWidget(self.btn_prev_frame)
        pc_layout.addWidget(self.btn_next_frame)
        pc_layout.addWidget(self.seek_slider, 1)
        pc_layout.addWidget(self.lbl_time)
        pc_layout.addWidget(self.speed_combo)

        layout.addWidget(self.play_controls)
        self.play_controls.setVisible(False)

        self._set_running(False)
        return panel

    def _results_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(250)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        # Live metrics row.
        summary = QGroupBox("Live metrics")
        summary_layout = QVBoxLayout(summary)
        self.metric_objects = QLabel("Objects: 0")
        self.metric_extra = QLabel("—")
        self.metric_extra.setWordWrap(True)
        summary_layout.addWidget(self.metric_objects)
        summary_layout.addWidget(self.metric_extra)
        self.sys_label = QLabel("CPU —  RAM —  GPU —")
        self.sys_label.setObjectName("sysmetric")
        summary_layout.addWidget(self.sys_label)
        layout.addWidget(summary)

        # Detections list.
        detections = QGroupBox("Detections")
        det_layout = QVBoxLayout(detections)
        self.detection_list = QListWidget()
        det_layout.addWidget(self.detection_list)
        layout.addWidget(detections, 1)

        # Text / codes.
        console = QGroupBox("Text / codes")
        console_layout = QVBoxLayout(console)
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setFixedHeight(80)
        console_layout.addWidget(self.text_output)
        layout.addWidget(console)

        # Notification center.
        alerts = QGroupBox("Notifications")
        alerts_layout = QVBoxLayout(alerts)
        self.alerts_output = QTextEdit()
        self.alerts_output.setReadOnly(True)
        self.alerts_output.setFixedHeight(70)
        alerts_layout.addWidget(self.alerts_output)
        layout.addWidget(alerts)

        # AI actions.
        ai_actions = QGroupBox("AI actions")
        ai_layout = QVBoxLayout(ai_actions)
        self.btn_register = QPushButton("👤  Register Face")
        self.btn_register.clicked.connect(self.register_face)
        ai_layout.addWidget(self.btn_register)

        self.btn_manage_faces = QPushButton("👥  Manage Face DB")
        self.btn_manage_faces.clicked.connect(self.manage_face_db)
        ai_layout.addWidget(self.btn_manage_faces)

        self.btn_view_plates = QPushButton("🚗  View Plate Log")
        self.btn_view_plates.clicked.connect(self.view_plate_log)
        ai_layout.addWidget(self.btn_view_plates)

        self.btn_live_anpr = QPushButton("🚦  Live ANPR")
        self.btn_live_anpr.clicked.connect(self.open_live_anpr)
        ai_layout.addWidget(self.btn_live_anpr)

        self.btn_video_anpr = QPushButton("🎞  Video ANPR")
        self.btn_video_anpr.clicked.connect(self.open_video_anpr)
        ai_layout.addWidget(self.btn_video_anpr)

        self.btn_scan_doc = QPushButton("📄  Scan Document")
        self.btn_scan_doc.clicked.connect(self.scan_document)
        ai_layout.addWidget(self.btn_scan_doc)
        self.btn_plant = QPushButton("🌿  Classify Plant")
        self.btn_plant.clicked.connect(self.classify_plant)
        ai_layout.addWidget(self.btn_plant)
        layout.addWidget(ai_actions)

        export = QPushButton("⬇  Export detections (CSV)")
        export.clicked.connect(self.export_csv)
        layout.addWidget(export)
        return panel

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        
        # Add Batch Process action
        batch_act = QAction("Batch Process Folder…", self)
        batch_act.triggered.connect(self.batch_process_folder)
        file_menu.addAction(batch_act)
        file_menu.addSeparator()

        for text, shortcut, handler in (
            ("Capture image", "Ctrl+S", self.capture_image),
            ("Export CSV", "Ctrl+E", self.export_csv),
            ("Export PDF report", "Ctrl+P", self.export_pdf),
            ("Open output folder", "Ctrl+O", self.open_output),
        ):
            action = QAction(text, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(handler)
            file_menu.addAction(action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        camera_menu = self.menuBar().addMenu("&Camera")
        for text, shortcut, handler in (
            ("Start", "Ctrl+R", self.start_camera),
            ("Pause / Resume", "Space", self.toggle_pause),
            ("Stop", "Ctrl+T", self.stop_camera),
            ("Toggle recording", "Ctrl+D", self.toggle_recording),
            ("Fullscreen", "Ctrl+Return", self.toggle_fullscreen),
        ):
            action = QAction(text, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(handler)
            camera_menu.addAction(action)

        view_menu = self.menuBar().addMenu("&View")
        analytics_action = QAction("Analytics dashboard", self)
        analytics_action.triggered.connect(self.open_analytics)
        view_menu.addAction(analytics_action)
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.open_settings)
        view_menu.addAction(settings_action)
        fs_action = QAction("Fullscreen (F11)", self)
        fs_action.setShortcut(QKeySequence("F11"))
        fs_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fs_action)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("About VisionSuite", self)
        about.triggered.connect(
            lambda: QMessageBox.information(
                self, "VisionSuite",
                "VisionSuite — real-time computer vision studio.\n"
                "YOLO detection, face mesh, hand gestures, pose analytics,\n"
                "image processing, OCR, code scanning and 15+ AI modules.",
            )
        )
        help_menu.addAction(about)

    # -- camera lifecycle ---------------------------------------------
    def _refresh_cameras(self) -> None:
        self.camera_combo.clear()
        cameras = list_cameras()
        if not cameras:
            self.camera_combo.addItem("No camera detected", -1)
            self.statusBar().showMessage("No camera detected — connect one and rescan")
            return
        for index in cameras:
            self.camera_combo.addItem(f"Camera #{index}", index)

    @Slot()
    def start_camera(self) -> None:
        if self.thread is not None:
            return

        self.config.source = self.source_combo.currentText()
        if self.config.source == "webcam":
            index = self.camera_combo.currentData()
            if index is None or index < 0:
                QMessageBox.warning(self, "No camera", "No usable camera device was found.")
                return
            self.config.camera_index = int(index)
        elif self.config.source == "ip":
            if not self.config.ip_url:
                QMessageBox.warning(self, "IP camera", "Enter an RTSP/IP stream URL in Settings.")
                return
        elif self.config.source in ("video_file", "image"):
            if not self.config.media_path:
                QMessageBox.warning(self, "Media", "Select a video/image file in Settings.")
                return

        self.config.resolution = RESOLUTIONS[self.resolution_combo.currentText()]

        self.thread = QThread(self)
        self.worker = VisionWorker(self.config)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status.connect(self.statusBar().showMessage)
        self.worker.failed.connect(self.on_failure)
        self.worker.alert.connect(self.on_alert)
        self.request_stop.connect(self.worker.stop)
        self.request_pause.connect(self.worker.set_paused)
        self.thread.start()
        self._set_running(True)

    @Slot()
    def stop_camera(self) -> None:
        if self.worker is None or self.thread is None:
            return
        self.worker.stop()
        self.thread.quit()
        self.thread.wait(2000)
        self.thread = None
        self.worker = None
        self.paused = False
        if self.recorder.active:
            self.recorder.stop()
        self.video.setText("Camera is off")
        self.video.setPixmap(QPixmap())
        self._set_running(False)
        self.statusBar().showMessage("Camera stopped")

    @Slot()
    def toggle_pause(self) -> None:
        if self.worker is None:
            return
        self.paused = not self.paused
        self.request_pause.emit(self.paused)
        self.btn_pause.setText("▶  Resume" if self.paused else "⏸  Pause")

    def _set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        for button in (self.btn_pause, self.btn_stop, self.btn_capture, self.btn_record):
            button.setEnabled(running)

    # -- slots ---------------------------------------------------------
    @Slot(object)
    def on_frame(self, result: FrameResult) -> None:
        self.last_frame = result.frame
        self.fps_label.setText(f"{result.fps:4.1f} FPS")
        self._render(result.frame)
        self._update_panel(result)
        if self.recorder.active:
            self.recorder.write(result.frame)

        # Update video playback controls
        if self.worker and self.worker._camera and self.config.source == "video_file":
            stats = self.worker._camera.get_video_stats()
            if stats:
                total = stats["total_frames"]
                curr = stats["current_frame"]
                duration = stats["duration_s"]
                position = stats["position_s"]

                self.seek_slider.blockSignals(True)
                self.seek_slider.setRange(0, total)
                self.seek_slider.setValue(curr)
                self.seek_slider.blockSignals(False)

                fmt_curr = f"{int(position)//60:02d}:{int(position)%60:02d}"
                fmt_total = f"{int(duration)//60:02d}:{int(duration)%60:02d}"
                self.lbl_time.setText(f"{fmt_curr} / {fmt_total} ({curr}/{total})")
                self.play_controls.setVisible(True)
        else:
            self.play_controls.setVisible(False)

    @Slot(str)
    def on_failure(self, message: str) -> None:
        QMessageBox.critical(self, "Source error", message)
        self.stop_camera()

    @Slot(str)
    def on_alert(self, message: str) -> None:
        self.alerts_output.append(f"[{timestamp()}] {message}")
        self.statusBar().showMessage(message, 4000)

        # 1. Desktop notification
        if getattr(self.config, "desktop_notifications", True):
            self.tray_icon.showMessage(
                "VisionSuite Alert", message, QSystemTrayIcon.MessageIcon.Warning, 3000
            )

        # 2. Sound alarm (with 3-second cooldown)
        if getattr(self.config, "sound_alerts", True):
            now = time.time()
            if not hasattr(self, "_last_sound_time") or (now - self._last_sound_time) > 3.0:
                self._last_sound_time = now
                QApplication.beep()

        # 3. Email alerts (in a background thread to prevent UI freezing, with 60-second cooldown)
        if getattr(self.config, "email_alerts_enabled", False):
            now = time.time()
            if not hasattr(self, "_last_email_time") or (now - self._last_email_time) > 60.0:
                self._last_email_time = now
                threading.Thread(target=self._send_email_alert, args=(message,), daemon=True).start()

    def _send_email_alert(self, message: str) -> None:
        import smtplib
        from email.mime.text import MIMEText
        try:
            msg = MIMEText(
                f"VisionSuite detected an alert event:\n\n{message}\n\nTimestamp: "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            msg["Subject"] = "VisionSuite Security Alert"
            msg["From"] = self.config.email_sender
            msg["To"] = self.config.email_recipient

            with smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port) as server:
                server.starttls()
                server.login(self.config.email_sender, self.config.email_password)
                server.sendmail(self.config.email_sender, self.config.email_recipient, msg.as_string())
            log.info("Alert email sent successfully.")
        except Exception as e:
            log.warning("Failed to send alert email: %s", e)

    def _render(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.video.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video.setPixmap(pixmap)

    def _update_panel(self, result: FrameResult) -> None:
        self.detection_list.clear()
        for det in result.detections[:60]:
            self.detection_list.addItem(f"{det.label}  —  {det.confidence:.0%}")

        info = result.info
        self.metric_objects.setText(f"Detections: {len(result.detections)}")

        lines: list[str] = []
        if "hands" in info:
            summary = ", ".join(
                f"{h['side']}: {h['fingers']} fingers ({h['gesture']})" for h in info["hands"]
            )
            lines.append(summary or "No hands detected")
        if "squats" in info:
            lines.append(
                f"Squats: {info['squats']}  Push-ups: {info['pushups']}\n"
                f"Knee {info.get('knee_angle', '—')}°  Elbow {info.get('elbow_angle', '—')}°"
            )
        if info.get("counts"):
            lines.append("  ".join(f"{k}×{v}" for k, v in sorted(info["counts"].items())))
        if "faces" in info:
            lines.append(f"Faces: {info['faces']}")
        face_status = info.get("face_recognition", {})
        if face_status.get("error"):
            lines.append(f"Face recognition unavailable: {face_status['error']}")
        elif face_status.get("recognition"):
            lines.append(face_status["recognition"])
        if "signs" in info and info["signs"] != "—":
            lines.append(f"Signs: {info['signs']}")
        self.metric_extra.setText("\n".join(lines) if lines else "—")

        text = info.get("text")
        if text:
            self.text_output.setPlainText(text)
        codes = [d.label for d in result.detections if d.source == "code"]
        if codes:
            self.text_output.setPlainText("\n".join(codes))

        if getattr(result, "modules", None):
            timing = "  ".join(f"{k}:{v:.0f}ms" for k, v in result.modules.items())
            self.setWindowTitle(f"VisionSuite — {timing}")

    def _update_system(self) -> None:
        if self.worker is None or not self.config.system_monitor:
            self.sys_label.setText("CPU —  RAM —  GPU —")
            return
        try:
            load = self.worker.analytics.system_load()
            self.sys_label.setText(
                f"CPU {load['cpu']:3.0f}%   RAM {load['ram']:3.0f}%   "
                f"GPU {load['gpu']:3.0f}%"
            )
        except Exception:  # pragma: no cover
            self.sys_label.setText("CPU —  RAM —  GPU —")

    def refresh_db_status(self) -> None:
        """Refresh the PostgreSQL status indicator in the sidebar."""
        try:
            label = database.connection_status_label()
            self.db_status_label.setText(f"PostgreSQL: {label}")
            if database.get_pg_status().get("connected"):
                self.db_status_label.setStyleSheet("color:#12a67c;")
            else:
                self.db_status_label.setStyleSheet("color:#e0a53a;")
        except Exception:  # pragma: no cover
            self.db_status_label.setText("PostgreSQL: unavailable")

    # -- actions ---------------------------------------------------------
    @Slot()
    def capture_image(self) -> None:
        if self.last_frame is None:
            return
        path = save_snapshot(self.last_frame)
        self.statusBar().showMessage(f"Saved {path}")

    @Slot()
    def register_face(self) -> None:
        dialog = FaceEnrollmentDialog(self.face_db, self.last_frame, self)
        dialog.exec()

    @Slot()
    def scan_document(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open document", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        try:
            module = get_module("document_scan")
            text = module.scan_file(Path(path))
            self.text_output.setPlainText(text or "No text recognised.")
            self.statusBar().showMessage("Document scanned")
        except Exception as exc:  # pragma: no cover
            log.exception("Document scan failed")
            QMessageBox.critical(self, "Document scanner", str(exc))

    @Slot()
    def classify_plant(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open plant image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        try:
            module = get_module("plant_disease")
            result = module.classify_file(Path(path))
            self.metric_extra.setText(result)
            self.statusBar().showMessage("Plant classified")
        except Exception as exc:  # pragma: no cover
            log.exception("Plant classification failed")
            QMessageBox.critical(self, "Plant disease", str(exc))

    @Slot()
    def toggle_recording(self) -> None:
        if self.last_frame is None:
            return
        if self.recorder.active:
            path = self.recorder.stop()
            self.btn_record.setText("⏺  Record")
            self.statusBar().showMessage(f"Recording saved: {path}")
        else:
            self.recorder = VideoRecorder(fps=25.0)
            self.recorder.start(self.last_frame)
            self.btn_record.setText("⏹  Stop recording")
            self.statusBar().showMessage("Recording…")

    @Slot()
    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    @Slot()
    def toggle_sidebar(self) -> None:
        self.sidebar_collapsed = not self.sidebar_collapsed
        self.sidebar_panel.setFixedWidth(44 if self.sidebar_collapsed else 320)

    @Slot()
    def toggle_theme(self) -> None:
        """Switch between dark and light themes."""
        self.config.theme = "light" if self.config.theme == "dark" else "dark"
        self.setStyleSheet(LIGHT_QSS if self.config.theme == "light" else DARK_QSS)
        self.btn_theme.setText("🌙" if self.config.theme == "light" else "☀️")
        self.statusBar().showMessage(
            f"Theme switched to {'light' if self.config.theme == 'light' else 'dark'}"
        )

    @Slot()
    def open_settings(self) -> None:
        dialog = SettingsDialog(self.config, self)
        dialog.exec()

    @Slot()
    def open_analytics(self) -> None:
        dialog = AnalyticsDialog(self.worker, self)
        dialog.refresh()
        dialog.exec()

    @Slot()
    def open_database_settings(self) -> None:
        dialog = DatabaseSettingsDialog(self)
        dialog.exec()
        self.refresh_db_status()

    @Slot()
    def manage_face_db(self) -> None:
        dialog = FaceDBManagerDialog(self.face_db, self)
        dialog.exec()

    @Slot()
    def view_plate_log(self) -> None:
        dialog = PlateLogDialog(self.face_db, self)
        dialog.exec()

    @Slot()
    def open_live_anpr(self) -> None:
        from live_anpr_window import LiveANPRWindow
        dialog = LiveANPRWindow(self)
        dialog.exec()

    @Slot()
    def open_video_anpr(self) -> None:
        from anpr_window import ANPRVideoWindow
        dialog = ANPRVideoWindow(self)
        dialog.exec()

    @Slot()
    def batch_process_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder to batch process")
        if folder:
            dialog = BatchProcessDialog(Path(folder), self.config, self)
            dialog.exec()

    # -- video playback controls --------------------------------------
    @Slot()
    def on_seek(self, value: int) -> None:
        if self.worker and self.worker._camera:
            self.worker._camera.seek(value)

    @Slot()
    def on_speed_changed(self, text: str) -> None:
        if self.worker and self.worker._camera:
            speed = float(text.replace("x", ""))
            self.worker._camera.set_speed(speed)

    @Slot()
    def prev_frame(self) -> None:
        if self.worker and self.worker._camera:
            self.worker._camera.step_frame(forward=False)

    @Slot()
    def next_frame(self) -> None:
        if self.worker and self.worker._camera:
            self.worker._camera.step_frame(forward=True)

    @Slot()
    def export_csv(self) -> None:
        rows = self.worker.csv_rows if self.worker else []
        if not rows:
            QMessageBox.information(self, "Nothing to export",
                                    "Enable 'Log detections to CSV' and run detection first.")
            return
        path = export_detections(rows, OUTPUT_DIR / f"detections-{timestamp()}.csv")
        self.statusBar().showMessage(f"Exported {len(rows)} rows to {path}")

    @Slot()
    def export_pdf(self) -> None:
        if self.worker is None:
            QMessageBox.information(self, "No session", "Start the camera and detect first.")
            return
        try:
            path = self.worker.analytics.export_pdf(OUTPUT_DIR)
            self.statusBar().showMessage(f"Exported PDF report → {path.name}")
        except Exception as exc:  # pragma: no cover
            log.exception("PDF export failed")
            QMessageBox.critical(self, "PDF export", str(exc))

    @Slot()
    def open_output(self) -> None:
        QFileDialog.getOpenFileName(self, "Output folder", str(OUTPUT_DIR))

    def _make_mode_toggle(self, mode: str):
        def toggle(state: int) -> None:
            if state:
                self.config.enabled_modes.add(mode)
            else:
                self.config.enabled_modes.discard(mode)
            self.config.mode = next(iter(self.config.enabled_modes), "Objects")
            self.statusBar().showMessage(
                f"Mode: {mode} {'enabled' if state else 'disabled'} "
                f"({len(self.config.enabled_modes)} active)"
            )
        return toggle

    def _make_module_toggle(self, key: str):
        def toggle(state: int) -> None:
            if state:
                self.config.enabled_modules.add(key)
            else:
                self.config.enabled_modules.discard(key)
            self.statusBar().showMessage(
                f"{AI_MODULES[key]} {'enabled' if state else 'disabled'}"
            )
        return toggle

    def _on_confidence(self, value: int) -> None:
        self.config.confidence = value / 100.0
        self.conf_label.setText(f"Confidence: {self.config.confidence:.2f}")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.stop_camera()
        super().closeEvent(event)
