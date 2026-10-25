from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QGridLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QMessageBox,
)

from api_client import ApiClient
from downloader import Downloader
from pet_manager import PetManager
from pet_card import PetCard


class PetStoreTab(QWidget):
    """宠物商店标签页：搜索栏 + 卡片网格"""

    pet_activated = Signal(str)  # pet_id

    def __init__(self, api_client: ApiClient, pet_manager: PetManager, parent=None):
        super().__init__(parent)
        self.api = api_client
        self.manager = pet_manager
        self.cards: dict[str, PetCard] = {}
        self.all_pets: list[dict] = []
        self.active_downloads: dict[str, Downloader] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # Top bar
        top_bar = QHBoxLayout()

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("搜索桌宠名称、描述或标签...")
        self.search_box.setFixedHeight(36)
        self.search_box.setStyleSheet("""
            QLineEdit {
                background: #313244; color: #cdd6f4; border: 1px solid #45475a;
                border-radius: 8px; padding: 0 12px; font-size: 13px;
            }
            QLineEdit:focus { border-color: #89b4fa; }
        """)
        self.search_box.textChanged.connect(self._filter_cards)
        top_bar.addWidget(self.search_box)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setFixedHeight(36)
        self.refresh_btn.setCursor(Qt.PointingHandCursor)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background: #89b4fa; color: #1e1e2e; border: none;
                border-radius: 8px; padding: 0 16px; font-weight: bold;
            }
            QPushButton:hover { background: #74c7ec; }
        """)
        self.refresh_btn.clicked.connect(self.load_catalog)
        top_bar.addWidget(self.refresh_btn)

        layout.addLayout(top_bar)

        # Status label
        self.status_label = QLabel("点击刷新加载桌宠目录")
        self.status_label.setStyleSheet("color: #6c7086; font-size: 12px; padding: 4px 0;")
        layout.addWidget(self.status_label)

        # Scroll area with grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setAlignment(Qt.AlignTop)
        self.grid_layout.setSpacing(12)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)

        scroll.setWidget(self.grid_widget)
        layout.addWidget(scroll)

    def load_catalog(self):
        """从服务端加载桌宠目录"""
        self.status_label.setText("正在加载...")
        try:
            self.all_pets = self.api.list_pets()
            self._build_cards()
            self.status_label.setText(f"共 {len(self.all_pets)} 个桌宠")
        except Exception as e:
            self.status_label.setText(f"加载失败: {e}")
            QMessageBox.warning(self, "错误", f"无法连接服务端:\n{e}")

    def _build_cards(self):
        """构建卡片网格"""
        # Clear existing
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.cards.clear()

        installed = self.manager.list_installed()
        active_id = self.manager.get_active_pet_id()

        for i, pet in enumerate(self.all_pets):
            card = PetCard(pet)
            card.download_requested.connect(self._on_download)
            card.uninstall_requested.connect(self._on_uninstall)
            card.start_requested.connect(self._on_start)
            card.stop_requested.connect(self._on_stop)

            pet_id = pet["id"]
            if pet_id in installed:
                card.set_installed(True)
            if pet_id == active_id:
                card.set_active(True)
                card.auto_launch()
            # Try loading thumbnail
            self._load_thumbnail(card, pet_id)

            self.cards[pet_id] = card
            row = i // 4
            col = i % 4
            self.grid_layout.addWidget(card, row, col)

    def _load_thumbnail(self, card: PetCard, pet_id: str):
        """尝试从服务端加载缩略图"""
        try:
            import requests
            url = f"{self.api.base_url}/api/pets/{pet_id}/thumbnail"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                pixmap = QPixmap()
                pixmap.loadFromData(resp.content)
                if not pixmap.isNull():
                    card.set_thumbnail(pixmap)
                    return
        except Exception:
            pass
        # Fallback: use placeholder color from pet info
        color_map = {
            "cat_miku": (255, 182, 193),
            "dog_hachi": (210, 180, 140),
            "dragon_kali": (178, 34, 34),
            "bunny_mochi": (255, 228, 181),
        }
        r, g, b = color_map.get(pet_id, (49, 50, 68))
        pixmap = QPixmap(160, 100)
        pixmap.fill(Qt.transparent)
        from PySide6.QtGui import QPainter, QColor
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(r, g, b))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, 160, 100, 8, 8)
        painter.end()
        card.set_thumbnail(pixmap)

    def _filter_cards(self, text: str):
        """按名称/描述/标签搜索"""
        text = text.lower().strip()
        row = 0
        col = 0
        for pet in self.all_pets:
            card = self.cards.get(pet["id"])
            if not card:
                continue
            visible = True
            if text:
                name = pet.get("name", "").lower()
                desc = pet.get("description", "").lower()
                tags = " ".join(pet.get("tags", [])).lower()
                visible = text in name or text in desc or text in tags
            card.setVisible(visible)
            if visible:
                self.grid_layout.addWidget(card, row, col)
                col += 1
                if col >= 4:
                    col = 0
                    row += 1

    def _on_download(self, pet_id: str):
        """开始下载桌宠"""
        card = self.cards.get(pet_id)
        if not card:
            return
        card.set_downloading(True)

        url = self.api.get_download_url(pet_id)
        pets_dir = str(self.manager.pets_dir)

        downloader = Downloader(pet_id, url, pets_dir)
        downloader.progress.connect(card.set_progress)
        downloader.finished_download.connect(self._on_download_finished)
        downloader.error.connect(self._on_download_error)
        downloader.start()
        self.active_downloads[pet_id] = downloader

    def _on_download_finished(self, pet_id: str):
        """下载完成回调"""
        card = self.cards.get(pet_id)
        if card:
            card.set_downloading(False)
            card.set_installed(True)
        try:
            self.manager.install_pet(pet_id, str(self.manager.pets_dir / f"{pet_id}.zip"))
            self.status_label.setText(f"✔ {pet_id} 安装成功")
        except Exception as e:
            self.status_label.setText(f"安装失败: {e}")

        # Clean up downloader
        self.active_downloads.pop(pet_id, None)

    def _on_download_error(self, pet_id: str, error: str):
        card = self.cards.get(pet_id)
        if card:
            card.set_downloading(False)
            card.set_progress(0)
        self.active_downloads.pop(pet_id, None)
        self.status_label.setText(f"下载错误: {error}")

    def _on_start(self, pet_id: str):
        """启动桌宠"""
        self.manager.set_active_pet(pet_id)
        for cid, card in self.cards.items():
            card.set_active(cid == pet_id)
        self.pet_activated.emit(pet_id)
        self.status_label.setText(f"已启动: {pet_id}")

    def _on_stop(self, pet_id: str):
        """停止桌宠"""
        self.manager.set_active_pet("")
        for cid, card in self.cards.items():
            card.set_active(False)
        self.pet_activated.emit("")
        self.status_label.setText(f"已停止: {pet_id}")

    def _on_uninstall(self, pet_id: str):
        """卸载桌宠"""
        reply = QMessageBox.question(
            self, "确认卸载", f"确定要卸载桌宠「{pet_id}」吗？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            was_active = self.manager.get_active_pet_id() == pet_id
            self.manager.uninstall_pet(pet_id)
            card = self.cards.get(pet_id)
            if card:
                card.set_installed(False)
                card.set_active(False)
            if was_active:
                self.pet_activated.emit("")
            self.status_label.setText(f"已卸载: {pet_id}")
