from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.gui_main import SpikeCurationMainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = SpikeCurationMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
