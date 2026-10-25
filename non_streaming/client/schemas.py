from __future__ import annotations

from typing import TypedDict


class PetInfo(TypedDict, total=False):
    """桌宠元数据"""
    id: str
    name: str
    description: str
    version: str
    size_mb: float
    rating: float
    rating_count: int
    thumbnail_url: str
    tags: list[str]
    author: str
    created_at: str


class InstalledPetInfo(TypedDict):
    """本地已安装桌宠信息"""
    version: str
    installed_at: str
    local_path: str


class LocalState(TypedDict, total=False):
    """客户端本地状态"""
    active_pet_id: str | None
    installed_pets: dict[str, InstalledPetInfo]
    server_url: str
    netease_cellphone: str | None
