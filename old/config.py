import os
import json
from typing import Dict, Any
from PySide6.QtCore import QStandardPaths

CONFIG_FILE_NAME = "spectrogram_settings.json"
AUDIO_EXTS = ('.wav', '.ogg', '.mp3', '.flac', '.aac', '.m4a')

DEFAULT_SETTINGS = {
    'auto_highpass': True,
    'highpass_freq': 20,
    'resolution': 1,
    'fps': 30,
    'gpu': True,
    'merge': False,
    'suffix': '',
    'colormap': 'Vibrant (Turbo)',
    'color_mode': 'Static',
    'overlay': True,
    'window': 5.0,
    'auto_freq': True,
    'min_freq': 20,
    'max_freq': 20000,
    'dyn_range': 80,
    'gamma': 1.0,
    'blur': 0.0,
    'normalize': True,
    'bass': 0.0,
    'treble': 0.0,
    'last_input_dir': '',
    'default_load_folder': '',
    'visual_mode': 'Standard',
    'spec_type': 'Standard (STFT)',
    'n_fft': 4096,
    'hop_length': 512,
    'sharpen': True,
    'interpolation': 'Lanczos',
    'center_line': True,
    'analyzer_color_mode': 'Match Video',
    'line_base_width': 6,
    'line_magnitude_sensitivity': 800.0,
    'line_glow_intensity': 15.0,
    'line_bloom': 8.0,
    'analyzer_bar_count': 64,
    'analyzer_position': 'Center',
    'analyzer_show_center_line': True,
    'palette_folder': '',
}

def get_default_load_folder() -> str:
    music = QStandardPaths.writableLocation(QStandardPaths.MusicLocation)
    if music and os.path.isdir(music): return music
    home = QStandardPaths.writableLocation(QStandardPaths.HomeLocation)
    if home and os.path.isdir(home): return home
    return os.path.expanduser("~")

class ConfigManager:
    def __init__(self):
        config_dir = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        os.makedirs(config_dir, exist_ok=True)
        self.path = os.path.join(config_dir, CONFIG_FILE_NAME)

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {'settings': DEFAULT_SETTINGS}
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'settings' not in data: data['settings'] = {}
                for key, value in DEFAULT_SETTINGS.items():
                    if key not in data['settings']: data['settings'][key] = value
                return data
        except (json.JSONDecodeError, IOError):
            return {'settings': DEFAULT_SETTINGS}

    def save(self, config: Dict[str, Any]):
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
        except IOError as e:
            print(f"Error saving settings: {e}")