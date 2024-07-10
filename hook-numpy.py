
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
hiddenimports = collect_submodules('numpy')
datas = collect_data_files('numpy')



# Collect all dynamic libraries for Intel MKL
binaries = collect_dynamic_libs('numpy')