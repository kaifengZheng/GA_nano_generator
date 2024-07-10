from PyInstaller.utils.hooks import copy_metadata,collect_submodules

datas = copy_metadata('zstandard')
hiddenimports = collect_submodules('zstandard')