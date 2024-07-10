
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all data files from ase
datas = collect_data_files('ase')

# Collect all submodules from ase
hiddenimports = collect_submodules('ase.io')