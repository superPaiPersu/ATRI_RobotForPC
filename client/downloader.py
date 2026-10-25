from __future__ import annotations

import os
import zipfile
from pathlib import Path
from zipfile import BadZipFile

import requests
from PySide6.QtCore import QThread, Signal


class Downloader(QThread):
    """后台线程下载器，带进度信号"""

    progress = Signal(int)          # 0-100
    finished_download = Signal(str) # pet_id 下载完成
    error = Signal(str, str)        # pet_id, 错误信息

    def __init__(self, pet_id: str, url: str, target_dir: str):
        super().__init__()
        self.pet_id = pet_id
        self.url = url
        self.target_dir = target_dir

    def run(self):
        os.makedirs(self.target_dir, exist_ok=True)
        tmp_path = os.path.join(self.target_dir, f"{self.pet_id}.tmp")
        final_path = os.path.join(self.target_dir, f"{self.pet_id}.zip")
        extract_dir = os.path.join(self.target_dir, self.pet_id)

        try:
            # Download
            with requests.get(self.url, stream=True, timeout=(10, 300)) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0

                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = min(int(downloaded / total * 100), 100)
                            self.progress.emit(pct)

            self.progress.emit(100)

            # Validate and extract
            try:
                with zipfile.ZipFile(tmp_path, "r") as zf:
                    zf.testzip()
            except BadZipFile:
                os.remove(tmp_path)
                self.error.emit(self.pet_id, "下载的文件已损坏")
                return

            # Extract — strip redundant top-level directory
            # e.g. zip contains "cat_miku/manifest.json" but we extract to
            # client/client_data/pets/cat_miku/, so we need to place files directly into
            # extract_dir, not into extract_dir/cat_miku/.
            # Also handles GBK-encoded filenames (common in Chinese zip files).
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(tmp_path, "r") as zf:
                entries = [n for n in zf.namelist() if n and n != "/"]
                top_dirs = {n.split("/")[0] for n in entries}
                common_top = top_dirs.pop() if len(top_dirs) == 1 else None

                def _fix_name(raw_name: str) -> str:
                    """尝试将 UTF-8 乱码文件名从 GBK 解码"""
                    try:
                        raw_name.encode('ascii')
                        return raw_name  # 纯 ASCII，无需修复
                    except UnicodeEncodeError:
                        pass
                    try:
                        return raw_name.encode('cp437').decode('gbk')
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        return raw_name

                if common_top:
                    prefix = common_top + "/"
                    for member in zf.namelist():
                        if not member or member == common_top or member == common_top + "/":
                            continue
                        if member.startswith(prefix):
                            rel = member[len(prefix):]
                            if not rel:
                                continue
                            rel = _fix_name(rel)
                            dest = os.path.join(extract_dir, rel)
                            if member.endswith("/"):
                                os.makedirs(dest, exist_ok=True)
                            else:
                                os.makedirs(os.path.dirname(dest), exist_ok=True)
                                with zf.open(member) as src, open(dest, "wb") as dst:
                                    dst.write(src.read())
                else:
                    for member in zf.namelist():
                        if not member or member.endswith("/"):
                            continue
                        fixed = _fix_name(member)
                        dest = os.path.join(extract_dir, fixed)
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        with zf.open(member) as src, open(dest, "wb") as dst:
                            dst.write(src.read())

            # Atomic rename
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(tmp_path, final_path)

            self.finished_download.emit(self.pet_id)

        except requests.RequestException as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            self.error.emit(self.pet_id, f"网络错误: {e}")
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            self.error.emit(self.pet_id, f"下载失败: {e}")
