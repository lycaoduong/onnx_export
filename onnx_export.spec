# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['onnx_export.py'],
    pathex=['C:\\Users\\user\\anaconda3\\envs\\ui_onnx\\Lib\\site-packages', 'C:\\Users\\user\\AppData\\Local\\Temp\\tmpg_sy60u_\\application\\onnx_export\\torch\\lib'],
    binaries=[],
    datas=[],
    hiddenimports=['torchvision', 'torch', 'numpy', 'models', 'models.yolo', 'models.common', 'models.experimental'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='onnx_export',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='onnx_export',
)
