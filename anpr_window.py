"""Dedicated Vehicle ANPR video page.

A self-contained QDialog that lets the user upload (or drag-and-drop) a video,
process it in a background thread for vehicle detection / tracking and license
plate recognition, watch a live preview, inspect results in a searchable table,
and export them to CSV / Excel / PDF.

This module is imported lazily by gui.py so the main dashboard stays light.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QPushButton, QSlider, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from video_anpr import ANPRVideoWorker, SUPPORTED_VIDEO_EXT

log = logging.getLogger("anpr_window")


class ANPRVideoWindow(QDialog):
    """Vehicle ANPR page: upload video, detect/track vehicles, read plates."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VisionSuite — Vehicle ANPR (Video)")
        self.resize(1280, 860)

        from gui import DARK_QSS, LIGHT_QSS
        theme = parent.config.theme if parent is not None else "dark"
        self.setStyleSheet(DARK_QSS if theme == "dark" else LIGHT_QSS)

        self.worker: ANPRVideoWorker | None = None
        self.video_path = ""
        self.current_frame: np.ndarray | None = None
        self.records: list[Any] = []
        self._total_frames = 0

        self.setAcceptDrops(True)
        self._build_ui()

    # -- construction -------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # Top toolbar: upload + playback controls.
        toolbar = QHBoxLayout()
        self.btn_upload = QPushButton("⬆  Upload Video")
        self.btn_upload.setObjectName("primary")
        self.btn_upload.clicked.connect(self.upload_video)
        toolbar.addWidget(self.btn_upload)

        self.btn_play = QPushButton("▶  Play")
        self.btn_play.clicked.connect(self.play_video)
        self.btn_play.setEnabled(False)
        self.btn_pause = QPushButton("⏸  Pause / Resume")
        self.btn_pause.clicked.connect(self.pause_resume)
        self.btn_pause.setEnabled(False)
        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setEnabled(False)
        self.btn_prev = QPushButton("⏮ Frame")
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_prev.setEnabled(False)
        self.btn_next = QPushButton("Frame ⏭")
        self.btn_next.clicked.connect(self.next_frame)
        self.btn_next.setEnabled(False)

        for b in (self.btn_upload, self.btn_play, self.btn_pause, self.btn_stop,
                  self.btn_prev, self.btn_next):
            toolbar.addWidget(b)
        toolbar.addStretch(1)

        # ANPR Debug toggle — raw OCR, plate/enhanced crops, model status.
        self.btn_debug = QCheckBox("🐞 ANPR Debug")
        self.btn_debug.setToolTip("Show raw OCR, plate crop, enhanced crop, model status")
        toolbar.addWidget(self.btn_debug)

        self.fps_label = QLabel("FPS 0")
        self.fps_label.setObjectName("metric")
        toolbar.addWidget(self.fps_label)
        root.addLayout(toolbar)

        # Stats row.
        stats = QHBoxLayout()
        self.lbl_vehicles = QLabel("Vehicles: 0")
        self.lbl_vehicles.setObjectName("metric")
        self.lbl_plates = QLabel("Plates: 0")
        self.lbl_plates.setObjectName("metric")
        self.lbl_status = QLabel("No video loaded")
        self.lbl_status.setWordWrap(True)
        stats.addWidget(self.lbl_vehicles)
        stats.addWidget(self.lbl_plates)
        stats.addWidget(self.lbl_status, 1)

        # Model status label (debug diagnostics).
        self.lbl_model = QLabel("Plate model: —")
        self.lbl_model.setStyleSheet("color:#9fb0c3;")
        stats.addWidget(self.lbl_model)
        root.addLayout(stats)

        # Progress bar.
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        root.addWidget(self.progress)

        # Video preview.
        self.video = QLabel("Upload or drop a video to start")
        self.video.setObjectName("video")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumHeight(360)
        root.addWidget(self.video, 1)

        # Seek slider.
        seek_row = QHBoxLayout()
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.sliderMoved.connect(self.on_seek)
        self.lbl_time = QLabel("00:00 / 00:00")
        seek_row.addWidget(self.seek_slider, 1)
        seek_row.addWidget(self.lbl_time)
        root.addLayout(seek_row)

        # Search + export row.
        action_row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search plate number, vehicle type, track ID...")
        self.search_edit.textChanged.connect(self.refresh_table)
        action_row.addWidget(self.search_edit, 1)

        self.btn_copy = QPushButton("📋  Copy Plate")
        self.btn_copy.clicked.connect(self.copy_plate)
        action_row.addWidget(self.btn_copy)
        self.btn_history = QPushButton("🗂  View History")
        self.btn_history.clicked.connect(self.view_history)
        action_row.addWidget(self.btn_history)

        self.btn_csv = QPushButton("Export CSV")
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_xlsx = QPushButton("Export Excel")
        self.btn_xlsx.clicked.connect(self.export_excel)
        self.btn_pdf = QPushButton("Export PDF")
        self.btn_pdf.clicked.connect(self.export_pdf)
        for b in (self.btn_copy, self.btn_history, self.btn_csv, self.btn_xlsx, self.btn_pdf):
            action_row.addWidget(b)
        root.addLayout(action_row)

        # Results table.
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Plate Number", "OCR %", "Vehicle Type", "Track ID",
            "Timestamp", "Frame", "Vehicle Image", "Plate Image",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemDoubleClicked.connect(self.preview_images)
        root.addWidget(self.table, 1)

    # -- drag & drop ----------------------------------------------------
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.video.setStyleSheet("border: 2px solid #12a67c;")

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self.video.setStyleSheet("")
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        self.video.setStyleSheet("")
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and path.lower().endswith(SUPPORTED_VIDEO_EXT):
                self.load_video(path)
                break

    # -- video loading / playback --------------------------------------
    def upload_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.mpeg *.mpg)"
        )
        if path:
            self.load_video(path)

    def load_video(self, path: str) -> None:
        self.stop_video()
        self.video_path = path
        self.records.clear()
        self.table.setRowCount(0)
        self.lbl_vehicles.setText("Vehicles: 0")
        self.lbl_plates.setText("Plates: 0")
        self.progress.setValue(0)
        self.lbl_status.setText(f"Loaded: {Path(path).name}")
        self.video.setText(f"Loaded: {Path(path).name}\nPress Play to analyse")
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)

        # Show the first frame as a static preview.
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                self.current_frame = frame
                self._render(frame)
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()

    def play_video(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.video_path:
            return
        self.records.clear()
        self.table.setRowCount(0)

        self.worker = ANPRVideoWorker(
            self.video_path,
            confidence=0.35,
            debug=self.btn_debug.isChecked(),
            parent=self,
        )
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.progress.connect(self.on_progress)
        self.worker.status.connect(self.on_status)
        self.worker.result_ready.connect(self.on_result)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        self.lbl_status.setText("Processing…")

    def pause_resume(self) -> None:
        if self.worker is None:
            return
        if self.worker.is_paused:
            self.worker.resume()
            self.btn_pause.setText("⏸  Pause / Resume")
        else:
            self.worker.pause()
            self.btn_pause.setText("▶  Resume")

    def stop_video(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.worker.wait(2000)
            self.worker = None
        self.btn_pause.setText("⏸  Pause / Resume")
        self.lbl_status.setText("Stopped")

    def prev_frame(self) -> None:
        if self.worker is not None:
            self.worker.step_frame()

    def next_frame(self) -> None:
        if self.worker is not None:
            self.worker.step_frame()

    def on_seek(self, value: int) -> None:
        if self.worker is not None:
            self.worker.seek(value)

    # -- worker signals -------------------------------------------------
    @Slot(object)
    def on_frame(self, result) -> None:
        from modules.anpr_video import ANPRFrameResult
        if not isinstance(result, ANPRFrameResult):
            return
        if result.frame is not None and result.frame.size > 4:
            self.current_frame = result.frame
            self._render(result.frame)
        self.fps_label.setText(f"FPS {result.fps:4.1f}")
        self.lbl_vehicles.setText(f"Vehicles: {result.vehicle_count}")
        self.lbl_plates.setText(f"Plates: {result.plate_count}")
        # Show plate-model status in debug diagnostics.
        model = result.debug.get("model", {}) if isinstance(result.debug, dict) else {}
        if model:
            status = model.get("status", "")
            err = model.get("error", "")
            if status == "loaded":
                self.lbl_model.setText(f"Plate model: {model.get('name', '')} ✓")
            elif err:
                self.lbl_model.setText("Plate model: ERROR")
            else:
                self.lbl_model.setText(f"Plate model: {status}")

    @Slot(int, int)
    def on_progress(self, current: int, total: int) -> None:
        if total > 0:
            pct = int((current / total) * 100)
            self.progress.setValue(pct)
            self.lbl_status.setText(f"Processing frame {current}/{total}")

    @Slot(str)
    def on_status(self, message: str) -> None:
        self.lbl_status.setText(message)
        if "model" in message.lower():
            self.lbl_model.setText(message)

    @Slot(object)
    def on_result(self, record) -> None:
        self.records.append(record)
        self.refresh_table()

    @Slot(str)
    def on_error(self, message: str) -> None:
        self.lbl_status.setText(f"Error: {message}")
        self.lbl_model.setText("Plate model: ERROR")
        QMessageBox.critical(self, "ANPR Error", message)

    @Slot()
    def on_finished(self) -> None:
        self.lbl_status.setText("Processing finished")
        self.progress.setValue(100)
        self.btn_pause.setText("⏸  Pause / Resume")

    # -- rendering -------------------------------------------------------
    def _render(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.video.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video.setPixmap(pixmap)

    # -- results table ---------------------------------------------------
    def refresh_table(self) -> None:
        query = self.search_edit.text().strip().lower()
        self.table.setRowCount(0)
        for rec in self.records:
            if query and query not in rec.plate_number.lower() \
               and query not in rec.vehicle_type.lower() \
               and str(rec.track_id) not in query:
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(rec.plate_number))
            self.table.setItem(row, 1, QTableWidgetItem(f"{rec.ocr_confidence:.0%}"))
            self.table.setItem(row, 2, QTableWidgetItem(rec.vehicle_type))
            self.table.setItem(row, 3, QTableWidgetItem(str(rec.track_id)))
            ts = rec.timestamp_s
            m, s = divmod(int(ts), 60)
            self.table.setItem(row, 4, QTableWidgetItem(f"{m:02d}:{s:02d}"))
            self.table.setItem(row, 5, QTableWidgetItem(str(rec.frame_number)))
            self.table.setItem(row, 6, QTableWidgetItem(rec.vehicle_image))
            self.table.setItem(row, 7, QTableWidgetItem(rec.plate_image))
        self.lbl_plates.setText(f"Plates: {len(self.records)}")

    @Slot()
    def copy_plate(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            # No explicit selection: copy the most recent (first) result.
            row = 0
        if row >= self.table.rowCount():
            QMessageBox.information(self, "Copy Plate", "No plate results to copy yet.")
            return
        item = self.table.item(row, 0)
        if item is None:
            return
        plate = item.text().strip()
        if not plate:
            QMessageBox.warning(self, "Copy Plate", "The selected plate is empty.")
            return
        QGuiApplication.clipboard().setText(plate)
        self.lbl_status.setText(f"Copied plate: {plate}")
        # Visually select the row so the user knows which one was copied.
        self.table.selectRow(row)

    @Slot()
    def view_history(self) -> None:
        """Browse the shared ANPR history store — live data from all sessions."""
        from anpr_store import ANPRStore
        store = ANPRStore()
        try:
            rows = store.all(500)
        except Exception as exc:  # pragma: no cover
            log.warning("Failed to read history: %s", exc)
            rows = []
        finally:
            store.close()

        dlg = QDialog(self)
        dlg.setWindowTitle("ANPR History (Live Data)")
        dlg.resize(960, 520)
        lay = QVBoxLayout(dlg)
        search = QLineEdit()
        search.setPlaceholderText("Filter history...")
        lay.addWidget(search)
        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels([
            "Plate Number", "Confidence", "Vehicle Type", "Track ID",
            "Timestamp", "Source", "Frame",
        ])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        def populate(q: str = "") -> None:
            q = q.strip().lower()
            table.setRowCount(0)
            for r in rows:
                plate = str(r.get("plate", ""))
                vtype = str(r.get("vehicle_type", ""))
                track = str(r.get("track_id", ""))
                conf = f"{float(r.get('confidence') or 0.0):.0%}"
                local = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.get("recognized_at", 0)))
                if q and q not in plate.lower() and q not in vtype.lower() \
                   and q not in track:
                    continue
                row = table.rowCount()
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(plate))
                table.setItem(row, 1, QTableWidgetItem(conf))
                table.setItem(row, 2, QTableWidgetItem(vtype))
                table.setItem(row, 3, QTableWidgetItem(track))
                table.setItem(row, 4, QTableWidgetItem(local))
                table.setItem(row, 5, QTableWidgetItem(str(r.get("source", ""))))
                table.setItem(row, 6, QTableWidgetItem(str(r.get("frame_number", ""))))

        search.textChanged.connect(populate)
        populate()
        lay.addWidget(table, 1)
        close = QPushButton("Close")
        close.clicked.connect(dlg.accept)
        lay.addWidget(close)
        dlg.exec()

    def preview_images(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if not (0 <= row < self.table.rowCount()):
            return
        veh = self.table.item(row, 6).text()
        plat = self.table.item(row, 7).text()
        dialog = QDialog(self)
        dialog.setWindowTitle("Plate / Vehicle Preview")
        lay = QVBoxLayout(dialog)
        for path in (plat, veh):
            if path and Path(path).exists():
                lbl = QLabel()
                pix = QPixmap(path).scaled(
                    500, 300, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                lbl.setPixmap(pix)
                lay.addWidget(lbl)
            else:
                lay.addWidget(QLabel("(no image)"))
        close = QPushButton("Close")
        close.clicked.connect(dialog.accept)
        lay.addWidget(close)
        dialog.exec()

    # -- export ----------------------------------------------------------
    def _filtered_records(self) -> list[Any]:
        query = self.search_edit.text().strip().lower()
        if not query:
            return list(self.records)
        return [
            r for r in self.records
            if query in r.plate_number.lower() or query in r.vehicle_type.lower()
            or str(r.track_id) in query
        ]

    def _fmt_time(self, ts: float) -> str:
        m, s = divmod(int(ts), 60)
        return f"{m:02d}:{s:02d}"

    def export_csv(self) -> None:
        rows = self._filtered_records()
        if not rows:
            QMessageBox.information(self, "Export", "No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "anpr_results.csv", "CSV (*.csv)")
        if not path:
            return
        fields = ["Plate Number", "Plate Image", "Vehicle Type", "Track ID",
                  "Timestamp", "Frame", "OCR Confidence", "Vehicle Image"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for r in rows:
                writer.writerow([
                    r.plate_number, r.plate_image, r.vehicle_type, r.track_id,
                    self._fmt_time(r.timestamp_s), r.frame_number,
                    f"{r.ocr_confidence:.0%}", r.vehicle_image,
                ])
        QMessageBox.information(self, "Export", f"Exported {len(rows)} rows to:\n{path}")

    def export_excel(self) -> None:
        rows = self._filtered_records()
        if not rows:
            QMessageBox.information(self, "Export", "No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Excel", "anpr_results.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        try:
            from openpyxl import Workbook
            wb = Workbook()
            ws = wb.active
            ws.title = "ANPR Results"
            ws.append(["Plate Number", "Plate Image", "Vehicle Type", "Track ID",
                       "Timestamp", "Frame", "OCR Confidence", "Vehicle Image"])
            for r in rows:
                ws.append([
                    r.plate_number, r.plate_image, r.vehicle_type, r.track_id,
                    self._fmt_time(r.timestamp_s), r.frame_number,
                    f"{r.ocr_confidence:.0%}", r.vehicle_image,
                ])
            wb.save(path)
            QMessageBox.information(self, "Export", f"Exported {len(rows)} rows to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Excel Export", f"openpyxl required.\n{exc}")

    def export_pdf(self) -> None:
        rows = self._filtered_records()
        if not rows:
            QMessageBox.information(self, "Export", "No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF", "anpr_results.pdf", "PDF (*.pdf)")
        if not path:
            return
        try:
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
            from reportlab.lib.styles import getSampleStyleSheet

            doc = SimpleDocTemplate(path, pagesize=landscape(A4))
            styles = getSampleStyleSheet()
            data = [[Paragraph("Plate", styles["Normal"]),
                     Paragraph("Vehicle", styles["Normal"]),
                     Paragraph("Track", styles["Normal"]),
                     Paragraph("Conf.", styles["Normal"]),
                     Paragraph("Time", styles["Normal"]),
                     Paragraph("Frame", styles["Normal"])]]
            for r in rows:
                data.append([
                    r.plate_number, r.vehicle_type, str(r.track_id),
                    f"{r.ocr_confidence:.0%}", self._fmt_time(r.timestamp_s), str(r.frame_number),
                ])
            table = Table(data)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.07, 0.65, 0.49)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.92, 0.95, 0.95)]),
            ]))
            doc.build([Paragraph("VisionSuite — Vehicle ANPR Results", styles["Title"]), table])
            QMessageBox.information(self, "Export", f"Exported {len(rows)} rows to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "PDF Export", f"reportlab required.\n{exc}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_video()
        super().closeEvent(event)
