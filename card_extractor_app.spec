# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


datas = []
datas += collect_data_files("PIL")
datas += collect_data_files("pypdfium2")

binaries = []
binaries += collect_dynamic_libs("cv2")
binaries += collect_dynamic_libs("numpy")
binaries += collect_dynamic_libs("pypdfium2")

hiddenimports = []
hiddenimports += collect_submodules("cv2")
hiddenimports += collect_submodules("PIL")
hiddenimports += collect_submodules("pypdfium2")


a = Analysis(
    ["card_extractor_app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="business-card-extractor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
