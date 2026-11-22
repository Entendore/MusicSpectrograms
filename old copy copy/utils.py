import os
import threading
import subprocess
import platform

class StreamReader(threading.Thread):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.output = []
        self.daemon = True

    def run(self):
        try:
            for line in iter(self.stream.readline, b''):
                self.output.append(line.decode('utf-8', errors='replace'))
        finally:
            self.stream.close()

    def get_output(self) -> str:
        return ''.join(self.output)

def open_file_location(path: str):
    if not os.path.exists(path): return
    directory = os.path.dirname(path) if os.path.isfile(path) else path
    try:
        if platform.system() == "Windows": os.startfile(directory)
        elif platform.system() == "Darwin": subprocess.Popen(["open", directory])
        else: subprocess.Popen(["xdg-open", directory])
    except Exception: pass