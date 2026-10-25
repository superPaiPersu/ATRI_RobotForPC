# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

import MusicLibrary


SPEC_DIR = Path(SPECPATH)
MUSICLIB_DIR = Path(next(iter(MusicLibrary.__path__)))
MUSICLIB_BINARIES = [
    (str(dll_path), "MusicLibrary/lib")
    for dll_path in (MUSICLIB_DIR / "lib").glob("*.dll")
]


a = Analysis(
    [str(SPEC_DIR / 'main.py')],
    pathex=[str(SPEC_DIR)],
    binaries=MUSICLIB_BINARIES,
    datas=[],
    hiddenimports=[
        'MusicLibrary.common',
        'MusicLibrary.core',
        'MusicLibrary.kuGouMusicApi',
        'MusicLibrary.neteaseCloudMusicApi',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main',
)
