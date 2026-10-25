# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all


harness_datas, harness_binaries, harness_hiddenimports = collect_all('deepseek_harness')
runtime_datas, runtime_binaries, runtime_hiddenimports = collect_all('deepseek_harness_runtime')

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=harness_binaries + runtime_binaries,
    datas=[
        ('client_data', 'client_data'),
        ('pet_template', 'pet_template'),
        ('config.json', '.'),
        ('harness.patch.yml', '.'),
        ('skills', 'skills'),
    ] + harness_datas + runtime_datas,
    hiddenimports=['pypdf', 'docx', 'bocha_mcp_server'] + harness_hiddenimports + runtime_hiddenimports,
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
