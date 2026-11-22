import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    app.setOrganizationName("SpectrogramStudio")
    app.setApplicationName("Spectrogram Studio Pro")
    
    app.setStyle("Fusion")
    
    # Check for GPU availability early
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[Init] GPU Acceleration Available: {torch.cuda.get_device_name(0)}")
        else:
            print("[Init] PyTorch found, but no CUDA device detected. Using CPU.")
    except ImportError:
        print("[Init] PyTorch not installed. Running in CPU mode.")
        
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())