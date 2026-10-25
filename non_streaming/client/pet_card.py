from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
)


class PetCard(QFrame):
    """单个桌宠卡片组件"""

    download_requested = Signal(str)   # pet_id
    uninstall_requested = Signal(str)  # pet_id
    start_requested = Signal(str)      # pet_id
    stop_requested = Signal(str)       # pet_id

    def __init__(self, pet_info: dict, parent=None):
        super().__init__(parent)
        self.pet_info = pet_info
        self.pet_id = pet_info["id"]
        self._installed = False
        self._active = False
        self._downloading = False
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedWidth(180)
        self.setStyleSheet("""
            PetCard {
                background: #1e1e2e;
                border-radius: 12px;
                border: 2px solid transparent;
            }
            PetCard:hover {
                border-color: #89b4fa;
            }
            PetCard.active {
                border-color: #a6e3a1;
                background: #1e2e1e;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Thumbnail placeholder
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(160, 100)
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setStyleSheet("background: #313244; border-radius: 8px; color: #6c7086; font-size: 12px;")
        self.thumb_label.setText("暂无图片")
        layout.addWidget(self.thumb_label, alignment=Qt.AlignCenter)

        # Name
        self.name_label = QLabel(self.pet_info.get("name", self.pet_id))
        self.name_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.name_label.setStyleSheet("color: #cdd6f4;")
        self.name_label.setWordWrap(True)
        layout.addWidget(self.name_label)

        # Description
        desc = self.pet_info.get("description", "")
        self.desc_label = QLabel(desc)
        self.desc_label.setFont(QFont("Microsoft YaHei", 8))
        self.desc_label.setStyleSheet("color: #a6adc8;")
        self.desc_label.setWordWrap(True)
        self.desc_label.setMaximumHeight(36)
        layout.addWidget(self.desc_label)

        # Meta row: rating + size
        meta_layout = QHBoxLayout()
        rating = self.pet_info.get("rating", 0)
        rating_count = self.pet_info.get("rating_count", 0)
        self.rating_label = QLabel(f"★ {rating:.1f} ({rating_count})")
        self.rating_label.setFont(QFont("Microsoft YaHei", 8))
        self.rating_label.setStyleSheet("color: #f9e2af;")
        meta_layout.addWidget(self.rating_label)

        size_mb = self.pet_info.get("size_mb", 0)
        self.size_label = QLabel(f"{size_mb:.1f} MB")
        self.size_label.setFont(QFont("Microsoft YaHei", 8))
        self.size_label.setStyleSheet("color: #6c7086;")
        meta_layout.addWidget(self.size_label, stretch=1)
        layout.addLayout(meta_layout)

        # Version tag
        self.version_label = QLabel(f"v{self.pet_info.get('version', '?')}")
        self.version_label.setFont(QFont("Microsoft YaHei", 7))
        self.version_label.setStyleSheet("color: #585b70;")
        layout.addWidget(self.version_label)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #45475a; border-radius: 4px;
                text-align: center; background: #313244; color: #cdd6f4;
            }
            QProgressBar::chunk { background: #89b4fa; border-radius: 3px; }
        """)
        layout.addWidget(self.progress_bar)

        # Two action buttons row: install/uninstall + start/stop
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        # Install / Uninstall button
        self.install_btn = QPushButton("下载")
        self.install_btn.setFixedHeight(30)
        self.install_btn.setCursor(Qt.PointingHandCursor)
        self.install_btn.setFont(QFont("Microsoft YaHei", 9))
        self.install_btn.setStyleSheet("""
            QPushButton {
                background: #89b4fa; color: #1e1e2e; border: none;
                border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background: #74c7ec; }
            QPushButton:disabled { background: #45475a; color: #6c7086; }
        """)
        self.install_btn.clicked.connect(self._on_install)
        btn_layout.addWidget(self.install_btn, stretch=1)

        # Start / Stop button
        self.start_btn = QPushButton("启动")
        self.start_btn.setFixedHeight(30)
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setFont(QFont("Microsoft YaHei", 9))
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #a6e3a1; color: #1e1e2e; border: none;
                border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background: #94e2b1; }
            QPushButton:disabled { background: #45475a; color: #6c7086; }
        """)
        self.start_btn.clicked.connect(self._on_start)
        btn_layout.addWidget(self.start_btn, stretch=1)

        layout.addLayout(btn_layout)

    def set_installed(self, installed: bool):
        self._installed = installed
        self._update_buttons()

    def set_active(self, active: bool):
        self._active = active
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)
        self._update_buttons()

    def set_downloading(self, downloading: bool):
        self._downloading = downloading
        self.progress_bar.setVisible(downloading)
        if downloading:
            self.progress_bar.setValue(0)
            self.install_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
        else:
            self.progress_bar.setValue(0)
            self._update_buttons()

    def set_progress(self, value: int):
        self.progress_bar.setValue(value)

    def set_thumbnail(self, pixmap):
        from PySide6.QtGui import QPixmap
        from PySide6.QtCore import Qt as QtAlign
        scaled = pixmap.scaled(160, 100, QtAlign.KeepAspectRatio, QtAlign.SmoothTransformation)
        self.thumb_label.setPixmap(scaled)
        self.thumb_label.setText("")

    def _update_buttons(self):
        """Update both button states based on current install/active state."""
        if self._downloading:
            self.install_btn.setEnabled(False)
            self.install_btn.setText("下载中...")
            self.start_btn.setEnabled(False)
            return

        # Install / Uninstall button
        if self._installed:
            self.install_btn.setText("卸载")
            self.install_btn.setEnabled(not self._active)
            self.install_btn.setStyleSheet("""
                QPushButton {
                    background: #f38ba8; color: #1e1e2e; border: none;
                    border-radius: 6px; font-weight: bold;
                }
                QPushButton:hover { background: #eba0ac; }
                QPushButton:disabled { background: #45475a; color: #6c7086; }
            """)
        else:
            self.install_btn.setText("下载")
            self.install_btn.setEnabled(True)
            self.install_btn.setStyleSheet("""
                QPushButton {
                    background: #89b4fa; color: #1e1e2e; border: none;
                    border-radius: 6px; font-weight: bold;
                }
                QPushButton:hover { background: #74c7ec; }
                QPushButton:disabled { background: #45475a; color: #6c7086; }
            """)

        # Start / Stop button
        if self._installed:
            self.start_btn.setEnabled(True)
            if self._active:
                self.start_btn.setText("停止")
                self.start_btn.setStyleSheet("""
                    QPushButton {
                        background: #eba0ac; color: #1e1e2e; border: none;
                        border-radius: 6px; font-weight: bold;
                    }
                    QPushButton:hover { background: #f38ba8; }
                """)
            else:
                self.start_btn.setText("启动")
                self.start_btn.setStyleSheet("""
                    QPushButton {
                        background: #a6e3a1; color: #1e1e2e; border: none;
                        border-radius: 6px; font-weight: bold;
                    }
                    QPushButton:hover { background: #94e2b1; }
                """)
        else:
            self.start_btn.setText("启动")
            self.start_btn.setEnabled(False)
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background: #45475a; color: #6c7086; border: none;
                    border-radius: 6px; font-weight: bold;
                }
                QPushButton:disabled { background: #45475a; color: #6c7086; }
            """)

    def _on_install(self):
        if self._downloading:
            return
        if self._installed:
            if self._active:
                return  # can't uninstall while active
            self.uninstall_requested.emit(self.pet_id)
        else:
            self.download_requested.emit(self.pet_id)

    def auto_launch(self):
        self.start_requested.emit(self.pet_id)
    def _on_start(self):
        if self._downloading:
            return
        if not self._installed:
            return
        if self._active:
            self.stop_requested.emit(self.pet_id)
        else:
            self.start_requested.emit(self.pet_id)
