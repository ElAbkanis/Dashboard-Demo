# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_all

project_dir = Path(SPECPATH)
dash_datas, dash_binaries, dash_hiddenimports = collect_all("dash")
plotly_datas, plotly_binaries, plotly_hiddenimports = collect_all("plotly")

a = Analysis(
    [str(project_dir / "app.py")],
    pathex=[str(project_dir)],
    binaries=dash_binaries + plotly_binaries,
    datas=[
        (str(project_dir / "assets"), "assets"),
        *dash_datas,
        *plotly_datas,
    ],
    hiddenimports=dash_hiddenimports + plotly_hiddenimports,
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
    [],
    exclude_binaries=True,
    name="Pizarra",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="Pizarra",
)
