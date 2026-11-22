import os
import subprocess
import platform

def open_file_location(path: str):
    if not os.path.exists(path): return
    directory = os.path.dirname(path) if os.path.isfile(path) else path
    try:
        if platform.system() == "Windows": os.startfile(directory)
        elif platform.system() == "Darwin": subprocess.Popen(["open", directory])
        else: subprocess.Popen(["xdg-open", directory])
    except Exception: pass