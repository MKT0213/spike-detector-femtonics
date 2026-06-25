from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.core.gui_main import SpikeCurationMainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    window = SpikeCurationMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
