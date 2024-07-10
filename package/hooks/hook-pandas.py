from PyInstaller.utils.hooks import collect_all

# Collect all pandas dependencies and data files
datas, binaries, hiddenimports = collect_all('pandas')