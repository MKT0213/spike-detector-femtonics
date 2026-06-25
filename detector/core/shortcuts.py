from __future__ import annotations

from typing import Callable, Dict

from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QWidget


def bind_shortcuts(widget: QWidget, mapping: Dict[str, Callable[[], None]]) -> list[QShortcut]:
    shortcuts: list[QShortcut] = []
    for key_sequence, callback in mapping.items():
        shortcut = QShortcut(QKeySequence(key_sequence), widget)
        shortcut.activated.connect(callback)
        shortcuts.append(shortcut)
    return shortcuts
