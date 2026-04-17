# -*- mode: python ; coding: utf-8 -*-
# Elephant Re-ID Standalone Build Spec (Stable Mirror Version)
# Combines successful startup with full runtime stability.

import os
import sys
import glob
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

block_cipher = None
BASE = os.path.abspath(os.path.dirname(SPEC))

# ── Deep Parallel Mirroring (Cloning 121+ DLLs from Working Backup) ──────────
# Mirroring all binaries from the known-working copy into the bundle.
WORKING_BACKUP = r'C:\Users\giris\Downloads\ElephantReID\ElephantReID\_internal'

binaries = []

# DLL Folders to mirror (source_folder, destination_subpath)
mirror_dirs = [
    (WORKING_BACKUP, '.'),                           # Root DLLs (tbb12.dll, vcomp140.dll)
    (os.path.join(WORKING_BACKUP, 'torch', 'lib'), os.path.join('torch', 'lib')),
    (os.path.join(WORKING_BACKUP, 'numpy.libs'),  'numpy.libs'),
    (os.path.join(WORKING_BACKUP, 'faiss_cpu.libs'), 'faiss_cpu.libs'),
    (os.path.join(WORKING_BACKUP, 'PyQt6', 'Qt6', 'bin'), os.path.join('PyQt6', 'Qt6', 'bin')),
]

for src_dir, dst_path in mirror_dirs:
    if os.path.exists(src_dir):
        for dll in glob.glob(os.path.join(src_dir, '*.dll')):
            binaries.append((dll, dst_path))

# ── Assets & Data ────────────────────────────────────────────────────────────
added_files = [
    (os.path.join(BASE, 'models', 'best_model v4.6.pth'), 'models'),
    (os.path.join(BASE, 'models', 'gallery_embeddings.pt'), 'models'),
    (os.path.join(BASE, 'app_config.json'), '.'),
    (os.path.join(BASE, 'src'), 'src'),
    *collect_data_files('torchvision'),
    *collect_data_files('timm'),
    *collect_data_files('cv2'),
]

# ── Hidden Imports ───────────────────────────────────────────────────────────
hidden = [
    'torch', 'torchvision', 'PIL', 'numpy', 'cv2', 'timm', 'PyQt6',
    'src', 'src.models', 'src.models.dual_branch_extractor',
    'sklearn', 'sklearn.cluster', 'sklearn.metrics', 'scipy'
]

# ── Analysis (NO CUSTOM RTHOOK to prevent startup crash 0xC0000409) ───────────
a = Analysis(
    ['app.py'],
    pathex=[BASE],
    binaries=binaries,
    datas=added_files,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],  # ← REMOVED conflicting rthook
    excludes=['matplotlib', 'notebook', 'IPython', 'pandas', 'tensorflow', 'keras'],
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
    name='ElephantReID',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # ← DISABLED per user request for final build
    disable_windowed_traceback=False,


    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(BASE, 'src', 'elephant.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ElephantReID',
)
