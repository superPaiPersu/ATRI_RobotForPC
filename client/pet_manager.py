from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from schemas import LocalState, InstalledPetInfo


DEFAULT_STATE: LocalState = {
    "active_pet_id": None,
    "installed_pets": {},
    "server_url": "http://127.0.0.1:5000",
    "netease_cellphone": None,
}


class PetManager:
    """本地桌宠状态管理：安装、卸载、激活、持久化"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.pets_dir = self.data_dir / "pets"
        self.state_file = self.data_dir / "state.json"
        self.state = self._load_state()
        print(self.state)
    def _load_state(self) -> LocalState:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return dict(DEFAULT_STATE)

    def save(self):
        """原子写入：先写 .tmp 再 os.replace()"""
        os.makedirs(self.data_dir, exist_ok=True)
        tmp = str(self.state_file) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.state_file)

    def list_installed(self) -> dict[str, InstalledPetInfo]:
        return self.state.get("installed_pets", {})

    def is_installed(self, pet_id: str) -> bool:
        return pet_id in self.state.get("installed_pets", {})

    def get_active_pet_id(self) -> str | None:
        return self.state.get("active_pet_id")

    def set_active_pet(self, pet_id: str | None):
        self.state["active_pet_id"] = pet_id
        self.save()

    def install_pet(self, pet_id: str, zip_path: str):
        """从 zip 安装桌宠"""
        extract_dir = self.pets_dir / pet_id
        os.makedirs(extract_dir, exist_ok=True)

        with open(os.path.join(extract_dir, "manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)

        version = manifest.get("version", "unknown")
        self.state.setdefault("installed_pets", {})[pet_id] = {
            "version": version,
            "installed_at": datetime.now().isoformat(),
            "local_path": str(extract_dir),
        }
        self.save()

    def uninstall_pet(self, pet_id: str) -> bool:
        """卸载桌宠，删除本地文件"""
        installed = self.state.get("installed_pets", {})
        if pet_id not in installed:
            return False

        local_path = installed[pet_id].get("local_path", "")
        if local_path and os.path.isdir(local_path):
            shutil.rmtree(local_path)

        # Also remove zip file
        zip_path = self.pets_dir / f"{pet_id}.zip"
        if zip_path.exists():
            zip_path.unlink()

        del self.state["installed_pets"][pet_id]

        if self.state.get("active_pet_id") == pet_id:
            self.state["active_pet_id"] = None

        self.save()
        return True

    def get_server_url(self) -> str:
        return self.state.get("server_url", "http://127.0.0.1:5000")

    def set_server_url(self, url: str):
        self.state["server_url"] = url
        self.save()

    def get_netease_cellphone(self) -> str | None:
        return self.state.get("netease_cellphone")

    def set_netease_cellphone(self, cellphone_num: str):
        self.state["netease_cellphone"] = cellphone_num
        self.save()
