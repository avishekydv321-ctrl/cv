"""Dedicated Live ANPR page.

A QDialog that lets the user run real-time vehicle detection + license plate
recognition on a live camera source (USB webcam or IP / RTSP / CCTV stream).

Shows the annotated live feed with plate number / confidence / vehicle type /
track ID / timestamp, tracks a running vehicle & plate counter, exposes a
searchable results table with a Copy-Plate action, and exports the session to
CSV / Excel / PDF.

Imported lazily by gui.py so the main dashboard stays light.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from live_anpr import LiveANPRWorker
from config import RESOLUTIONS

log = logging.getLogger("live_anpr_window")


class LiveANPRWindow(QDialog):
    """Live ANPR page: real-time vehicle + plate recognition on camera feeds."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VisionSuite — Live Vehicle ANPR")
        self.resize(1280, 880)

        from gui import DARK_QSS, LIGHT_QSS
        theme = parent.config.theme if parent is not None else "dark"
        self.setStyleSheet(DARK_QSS if theme == "dark" else LIGHT_QSS)

        self.worker: LiveANPRWorker | None = None
        self.records: list[Any] = []
        self.current_frame: np.ndarray | None = None
        self._vehicles_seen = 0
        self._build_ui()

    # -- construction -----------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # Top toolbar: source selection + start/stop.
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Source"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["webcam", "ip"])
        self.source_combo.currentTextChanged.connect(self._on_source_change)
        toolbar.addWidget(self.source_combo)

        self.camera_combo = QComboBox()
        from camera import list_cameras
        for index in list_cameras():
            self.camera_combo.addItem(f"Camera #{index}", index)
        if self.camera_combo.count() == 0:
            self.camera_combo.addItem("No camera", -1)
        toolbar.addWidget(self.camera_combo)

        self.ip_edit = QLineEdit()
        self.ip_edit.setPlaceholderText("rtsp://user:pass@host:554/stream")
        self.ip_edit.setVisible(False)
        toolbar.addWidget(self.ip_edit, 1)

        toolbar.addWidget(QLabel("Res"))
        self.res_combo = QComboBox()
        self.res_combo.addItems(RESOLUTIONS.keys())
        self.res_combo.setCurrentText("1280 x 720 (HD)")
        toolbar.addWidget(self.res_combo)

        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.start_anpr)
        toolbar.addWidget(self.btn_start)
        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self.stop_anpr)
        self.btn_stop.setEnabled(False)
        toolbar.addWidget(self.btn_stop)
        toolbar.addStretch(1)

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
        self.lbl_status = QLabel("Configure a source and press Start")
        self.lbl_status.setWordWrap(True)
        stats.addWidget(self.lbl_vehicles)
        stats.addWidget(self.lbl_plates)
        stats.addWidget(self.lbl_status, 1)

        # Model status label (debug diagnostics).
        self.lbl_model = QLabel("Plate model: —")
        self.lbl_model.setStyleSheet("color:#9fb0c3;")
        stats.addWidget(self.lbl_model)
        root.addLayout(stats)

        # Live video feed.
        self.video = QLabel("Live ANPR feed")
        self.video.setObjectName("video")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumHeight(360)
        root.addWidget(self.video, 1)

        # Search + actions row.
        actions = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search plate, vehicle type, track ID...")
        self.search_edit.textChanged.connect(self.refresh_table)
        actions.addWidget(self.search_edit, 1)

        self.btn_copy = QPushButton("📋  Copy Plate")
        self.btn_copy.clicked.connect(self.copy_plate)
        actions.addWidget(self.btn_copy)
        self.btn_clear = QPushButton("🗑  Clear")
        self.btn_clear.clicked.connect(self.clear_records)
        actions.addWidget(self.btn_clear)

        self.btn_csv = QPushButton("Export CSV")
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_xlsx = QPushButton("Export Excel")
        self.btn_xlsx.clicked.connect(self.export_excel)
        self.btn_pdf = QPushButton("Export PDF")
        self.btn_pdf.clicked.connect(self.export_pdf)
        for b in (self.btn_csv, self.btn_xlsx, self.btn_pdf):
            actions.addWidget(b)
        root.addLayout(actions)

        # Results table.
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Plate Number", "OCR %", "Vehicle Type", "Track ID",
            "Timestamp", "Source", "Vehicle Image", "Plate Image",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemDoubleClicked.connect(self.preview_images)
        root.addWidget(self.table, 1)

    # -- source switching ------------------------------------------------
    def _on_source_change(self, text: str) -> None:
        is_ip = text == "ip"
        self.ip_edit.setVisible(is_ip)
        self.camera_combo.setVisible(not is_ip)

    # -- control ----------------------------------------------------------
    @Slot()
    def start_anpr(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        source = self.source_combo.currentText()
        if source == "ip":
            ip_url = self.ip_edit.text().strip()
            if not ip_url:
                QMessageBox.warning(self, "IP camera", "Enter an RTSP/IP stream URL.")
                return
        else:
            ip_url = ""
            index = self.camera_combo.currentData()
            if index is None or index < 0:
                QMessageBox.warning(self, "Camera", "No usable camera device was found.")
                return

        resolution = RESOLUTIONS[self.res_combo.currentText()]
        self.records.clear()
        self.table.setRowCount(0)
        self._vehicles_seen = 0
        self.lbl_vehicles.setText("Vehicles: 0")
        self.lbl_plates.setText("Plates: 0")

        self.worker = LiveANPRWorker(
            source=source,
            camera_index=self.camera_combo.currentData() if source == "webcam" else 0,
            ip_url=ip_url,
            resolution=resolution,
            confidence=0.35,
            debug=self.btn_debug.isChecked(),
            parent=self,
        )
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status.connect(self.on_status)
        self.worker.result_ready.connect(self.on_result)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("Starting live ANPR…")

    @Slot()
    def stop_anpr(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.worker.wait(3000)
            self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Stopped")

    # -- worker signals ---------------------------------------------------
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
        QMessageBox.critical(self, "Live ANPR Error", message)
        self.stop_anpr()

    @Slot()
    def on_finished(self) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Live ANPR stopped")

    # -- rendering ---------------------------------------------------------
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

    # -- results table ------------------------------------------------------
    def refresh_table(self) -> None:
        query = self.search_edit.text().strip().lower()
        self.table.setRowCount(0)
        for rec in self.records:
            if query:
                hay = f"{rec.plate_number} {rec.vehicle_type} {rec.track_id}".lower()
                if query not in hay:
                    continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(rec.plate_number))
            self.table.setItem(row, 1, QTableWidgetItem(f"{rec.ocr_confidence:.0%}"))
            self.table.setItem(row, 2, QTableWidgetItem(rec.vehicle_type))
            self.table.setItem(row, 3, QTableWidgetItem(str(rec.track_id)))
            local = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec.timestamp_s))
            self.table.setItem(row, 4, QTableWidgetItem(local))
            self.table.setItem(row, 5, QTableWidgetItem(rec.source))
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
    def clear_records(self) -> None:
        self.records.clear()
        self.table.setRowCount(0)
        self.lbl_plates.setText("Plates: 0")

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

    # -- export -------------------------------------------------------------
    def _filtered_records(self) -> list[Any]:
        query = self.search_edit.text().strip().lower()
        if not query:
            return list(self.records)
        return [
            r for r in self.records
            if query in f"{r.plate_number} {r.vehicle_type} {r.track_id}".lower()
        ]

    def export_csv(self) -> None:
        rows = self._filtered_records()
        if not rows:
            QMessageBox.information(self, "Export", "No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "live_anpr_results.csv", "CSV (*.csv)")
        if not path:
            return
        fields = ["Plate Number", "OCR Confidence", "Vehicle Type", "Track ID",
                  "Timestamp", "Source", "Frame", "Vehicle Image", "Plate Image"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for r in rows:
                writer.writerow([
                    r.plate_number, f"{r.ocr_confidence:.0%}", r.vehicle_type, r.track_id,
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.timestamp_s)),
                    r.source, r.frame_number, r.vehicle_image, r.plate_image,
                ])
        QMessageBox.information(self, "Export", f"Exported {len(rows)} rows to:\n{path}")

    def export_excel(self) -> None:
        rows = self._filtered_records()
        if not rows:
            QMessageBox.information(self, "Export", "No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Excel", "live_anpr_results.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        try:
            from openpyxl import Workbook
            wb = Workbook()
            ws = wb.active
            ws.title = "Live ANPR Results"
            ws.append(["Plate Number", "OCR Confidence", "Vehicle Type", "Track ID",
                       "Timestamp", "Source", "Frame", "Vehicle Image", "Plate Image"])
            for r in rows:
                ws.append([
                    r.plate_number, f"{r.ocr_confidence:.0%}", r.vehicle_type, r.track_id,
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.timestamp_s)),
                    r.source, r.frame_number, r.vehicle_image, r.plate_image,
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
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF", "live_anpr_results.pdf", "PDF (*.pdf)")
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
                     Paragraph("Source", styles["Normal"])]]
            for r in rows:
                data.append([
                    r.plate_number, r.vehicle_type, str(r.track_id),
                    f"{r.ocr_confidence:.0%}",
                    time.strftime("%H:%M:%S", time.localtime(r.timestamp_s)),
                    r.source,
                ])
            table = Table(data)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.07, 0.65, 0.49)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.92, 0.95, 0.95)]),
            ]))
            doc.build([Paragraph("VisionSuite — Live Vehicle ANPR Results", styles["Title"]), table])
            QMessageBox.information(self, "Export", f"Exported {len(rows)} rows to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "PDF Export", f"reportlab required.\n{exc}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_anpr()
        super().closeEvent(event)
