"""VisionSuite entry point."""
from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from gui import MainWindow
from utils import configure_logging, ensure_dirs


def main() -> int:
    configure_logging(logging.INFO)
    ensure_dirs()

    app = QApplication(sys.argv)
    app.setApplicationName("VisionSuite")
    app.setOrganizationName("VisionSuite Labs")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
