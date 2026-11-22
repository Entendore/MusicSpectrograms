#!/usr/bin/env python3
"""
Spectrogram Studio Pro - All-in-One Single File Version
Combines: app, config, download, main_window, namecleanup, palettes, renderer, utils, widgets, worker
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
import platform
import re
import threading
import traceback
from typing import Dict, Any, List, Tuple, Optional, Callable

import numpy as np
import cv2
import soundfile as sf
import librosa

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QProgressBar, QGroupBox, QComboBox, QDoubleSpinBox,
    QCheckBox, QSpinBox, QLineEdit, QMenuBar, QTextEdit, QMessageBox,
    QTableWidgetItem, QTableWidget, QAbstractItemView, QMenu, QHeaderView,
    QHeaderView, QSplitter, QFrame
)
from PySide6.QtCore import Qt, Signal, QThread, QStandardPaths
from PySide6.QtGui import QAction, QColor, QBrush, QTextCursor, QImage, QPixmap

# ── Optional Imports ──────────────────────────────────────────────────────────
try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None

try:
    from numba import cuda
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_FILE_NAME = "spectrogram_settings.json"
AUDIO_EXTS = ('.wav', '.ogg', '.mp3', '.flac', '.aac', '.m4a')

DEFAULT_SETTINGS = {
    'auto_highpass': True, 'highpass_freq': 20, 'resolution': 1, 'fps': 30,
    'gpu': True, 'merge': False, 'suffix': '', 'colormap': 'Vibrant (Turbo)',
    'color_mode': 'Static', 'overlay': True, 'window': 5.0, 'auto_freq': True,
    'min_freq': 20, 'max_freq': 20000, 'dyn_range': 80, 'gamma': 1.0,
    'blur': 0.0, 'normalize': True, 'bass': 0.0, 'treble': 0.0,
    'last_input_dir': '', 'default_load_folder': '', 'visual_mode': 'Standard',
    'spec_type': 'Standard (STFT)', 'n_fft': 4096, 'hop_length': 512,
    'sharpen': True, 'interpolation': 'Lanczos', 'center_line': True,
    'analyzer_color_mode': 'Match Video', 'line_base_width': 6,
    'line_magnitude_sensitivity': 800.0, 'line_glow_intensity': 15.0,
    'line_bloom': 8.0, 'analyzer_bar_count': 64, 'analyzer_position': 'Center',
    'analyzer_show_center_line': True, 'palette_folder': '',
}

def _get_default_load_folder() -> str:
    music = QStandardPaths.writableLocation(QStandardPaths.MusicLocation)
    if music and os.path.isdir(music):
        return music
    home = QStandardPaths.writableLocation(QStandardPaths.HomeLocation)
    if home and os.path.isdir(home):
        return home
    return os.path.expanduser("~")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class ConfigManager:
    def __init__(self):
        config_dir = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        os.makedirs(config_dir, exist_ok=True)
        self.path = os.path.join(config_dir, CONFIG_FILE_NAME)

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {'settings': DEFAULT_SETTINGS.copy()}
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'settings' not in data:
                data['settings'] = {}
            for key, value in DEFAULT_SETTINGS.items():
                if key not in data['settings']:
                    data['settings'][key] = value
            return data
        except (json.JSONDecodeError, IOError):
            return {'settings': DEFAULT_SETTINGS.copy()}

    def save(self, config: Dict[str, Any]):
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
        except IOError as e:
            print(f"Error saving settings: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def open_file_location(path: str):
    if not os.path.exists(path):
        return
    directory = os.path.dirname(path) if os.path.isfile(path) else path
    try:
        if platform.system() == "Windows":
            os.startfile(directory)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", directory])
        else:
            subprocess.Popen(["xdg-open", directory])
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# PALETTE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

PALETTE_SUBDIR = "palettes"

def _get_default_palette_dir() -> str:
    config_dir = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    palette_dir = os.path.join(config_dir, PALETTE_SUBDIR)
    return palette_dir

def _lerp_color(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)

def _generate_gradient(stops: List[Tuple[float, int, int, int]], n: int = 256) -> List[Tuple[int, int, int]]:
    colors = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.0
        matched = False
        for j in range(len(stops) - 1):
            if stops[j][0] <= t <= stops[j + 1][0]:
                span = stops[j + 1][0] - stops[j][0]
                t_local = (t - stops[j][0]) / span if span > 0 else 0.0
                r = _lerp_color(stops[j][1], stops[j + 1][1], t_local)
                g = _lerp_color(stops[j][2], stops[j + 1][2], t_local)
                b = _lerp_color(stops[j][3], stops[j + 1][3], t_local)
                colors.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
                matched = True
                break
        if not matched:
            colors.append((stops[-1][1], stops[-1][2], stops[-1][3]))
    return colors

_DEFAULT_PALETTE_DEFS = {
    'Neon_Cyber.map': {
        'desc': 'Cyberpunk neon - black purple pink cyan white',
        'stops': [
            (0.00, 0, 0, 0), (0.10, 15, 0, 60), (0.25, 120, 0, 160),
            (0.40, 220, 0, 120), (0.55, 255, 40, 80), (0.70, 0, 180, 255),
            (0.85, 120, 240, 255), (1.00, 255, 255, 255),
        ]
    },
    'Sunset_Fire.map': {
        'desc': 'Warm sunset - black red orange gold white',
        'stops': [
            (0.00, 0, 0, 0), (0.10, 60, 0, 0), (0.25, 160, 10, 0),
            (0.40, 220, 60, 0), (0.55, 255, 130, 0), (0.70, 255, 200, 20),
            (0.85, 255, 245, 140), (1.00, 255, 255, 255),
        ]
    },
    'Ocean_Depth.map': {
        'desc': 'Deep ocean - black navy teal aquamarine ice',
        'stops': [
            (0.00, 0, 0, 0), (0.10, 0, 0, 65), (0.25, 0, 20, 140),
            (0.40, 0, 70, 150), (0.55, 0, 130, 140), (0.70, 0, 200, 175),
            (0.85, 100, 235, 220), (1.00, 210, 255, 255),
        ]
    },
}

def _create_default_palettes(palette_dir: str):
    os.makedirs(palette_dir, exist_ok=True)
    for filename, defn in _DEFAULT_PALETTE_DEFS.items():
        filepath = os.path.join(palette_dir, filename)
        if os.path.exists(filepath):
            continue
        colors = _generate_gradient(defn['stops'])
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {defn['desc']}\n")
            f.write("# Auto-generated by Spectrogram Studio Pro\n")
            for r, g, b in colors:
                f.write(f"{r} {g} {b}\n")

def _load_map_file(filepath: str) -> Optional[np.ndarray]:
    try:
        colors = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        r = max(0, min(255, int(parts[0])))
                        g = max(0, min(255, int(parts[1])))
                        b = max(0, min(255, int(parts[2])))
                        colors.append([b, g, r])  # BGR for OpenCV
                    except ValueError:
                        continue
        if len(colors) == 0:
            return None
        while len(colors) < 256:
            colors.append(colors[-1] if colors else [0, 0, 0])
        colors = colors[:256]
        return np.array(colors, dtype=np.uint8).reshape(256, 1, 3)
    except Exception as e:
        print(f"Error loading .map file {filepath}: {e}")
        return None

def get_all_luts(palette_folder: str = None) -> Dict[str, np.ndarray]:
    luts = {}
    built_ins = {
        'Vibrant (Turbo)': cv2.COLORMAP_TURBO,
        'Rainbow (High Contrast)': cv2.COLORMAP_RAINBOW,
        'Thermal (Inferno)': cv2.COLORMAP_INFERNO,
        'Plasma (Deep)': cv2.COLORMAP_PLASMA,
        'Magma (Warm)': cv2.COLORMAP_MAGMA,
        'Viridis (Cool)': cv2.COLORMAP_VIRIDIS,
        'Jet (Classic)': cv2.COLORMAP_JET,
        'HSV': cv2.COLORMAP_HSV,
    }
    dummy = np.arange(256, dtype=np.uint8).reshape(1, 256)
    for name, cmap_id in built_ins.items():
        color_img = cv2.applyColorMap(dummy, cmap_id)
        luts[name] = color_img.reshape(256, 1, 3)

    if palette_folder is None or not palette_folder.strip():
        palette_folder = _get_default_palette_dir()
    _create_default_palettes(palette_folder)

    if os.path.isdir(palette_folder):
        for filename in sorted(os.listdir(palette_folder)):
            if filename.lower().endswith('.map'):
                path = os.path.join(palette_folder, filename)
                lut_data = _load_map_file(path)
                if lut_data is not None:
                    name = os.path.splitext(filename)[0]
                    luts[name] = lut_data
    return luts

# ══════════════════════════════════════════════════════════════════════════════
# STREAM READER (FFmpeg stderr)
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
# VIDEO ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class VideoEncoder:
    def __init__(self, target_w: int, target_h: int, fps: int,
                 audio_path: str, output_path: str, use_gpu: bool):
        self.target_w = target_w
        self.target_h = target_h
        self.fps = fps
        self.audio_path = audio_path
        self.output_path = output_path
        self.use_gpu = use_gpu
        self.process = None
        self.stderr_reader = None

    def _build_cmd(self):
        if self.use_gpu:
            codec, preset = 'h264_nvenc', 'p7'
            extra = ['-rc', 'vbr', '-cq', '18', '-b:v', '15M',
                     '-maxrate', '20M', '-profile:v', 'high',
                     '-movflags', '+faststart']
        else:
            codec, preset = 'libx264', 'medium'
            extra = ['-crf', '20', '-profile:v', 'high', '-level', '4.2',
                     '-movflags', '+faststart']
        return [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{self.target_w}x{self.target_h}', '-pix_fmt', 'rgb24',
            '-r', str(self.fps), '-i', '-',
            '-i', self.audio_path,
            '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', codec, '-preset', preset,
            *extra, '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '320k',
            '-shortest', self.output_path
        ]

    def start(self):
        self.process = subprocess.Popen(
            self._build_cmd(),
            stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )
        self.stderr_reader = StreamReader(self.process.stderr)
        self.stderr_reader.start()

    def write(self, data: bytes) -> bool:
        try:
            self.process.stdin.write(data)
            return True
        except (BrokenPipeError, OSError):
            return False

    def finish(self) -> str:
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except Exception:
            pass
        try:
            self.process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait()
        if self.stderr_reader:
            self.stderr_reader.join(timeout=5)
            return self.stderr_reader.get_output()
        return ""

# ══════════════════════════════════════════════════════════════════════════════
# SPECTROGRAM PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class SpectrogramProcessor:
    def __init__(self, settings: Dict):
        self.settings = settings
        self._filter_cache = {}
        self._remap_cache = {}
        self._gamma_lut = None
        self._gamma_cache_key = None

    def process_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        if len(y) == 0:
            return y

        # Smart noise detection / high-pass
        hp_freq = 0
        if self.settings.get('auto_highpass', True):
            hp_freq = self._detect_sub_bass_noise(y, sr)
        else:
            hp_freq = self.settings.get('highpass_freq', 20)

        if scipy_signal and hp_freq > 0 and hp_freq < sr / 2:
            try:
                b, a = scipy_signal.butter(5, hp_freq / (sr / 2), btype='high', analog=False)
                y = scipy_signal.filtfilt(b, a, y)
            except Exception:
                pass

        # Normalize
        if self.settings.get('normalize', True):
            peak = np.max(np.abs(y))
            if peak > 1e-5:
                y = librosa.util.normalize(y)

        # EQ
        if scipy_signal:
            if self.settings.get('bass_gain', 0) != 0:
                b, a = self._get_filter_coefficients(sr, 200, self.settings['bass_gain'], 'low')
                y = scipy_signal.lfilter(b, a, y)
            if self.settings.get('treble_gain', 0) != 0:
                b, a = self._get_filter_coefficients(sr, 4000, self.settings['treble_gain'], 'high')
                y = scipy_signal.lfilter(b, a, y)
        return y

    def _detect_sub_bass_noise(self, y: np.ndarray, sr: int) -> float:
        try:
            analysis_len = int(min(len(y), sr * 5))
            y_analysis = y[:analysis_len]
            n_fft = 4096
            D = np.abs(librosa.stft(y_analysis, n_fft=n_fft, hop_length=1024))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            sub_idx = np.where((freqs >= 0) & (freqs <= 60))[0]
            if len(sub_idx) == 0:
                return 20.0
            sub_energy = D[sub_idx, :]
            mean_e = np.mean(sub_energy, axis=0) + 1e-10
            geo_e = np.exp(np.mean(np.log(sub_energy + 1e-10), axis=0))
            flatness = np.mean(geo_e / mean_e)
            if flatness > 0.6:
                return 40.0
            elif flatness > 0.35:
                idx20 = np.argmin(np.abs(freqs - 20))
                idx40 = np.argmin(np.abs(freqs - 40))
                if np.mean(D[idx20, :]) > (np.mean(D[idx40, :]) * 0.8):
                    return 30.0
                return 20.0
            return 15.0
        except Exception:
            return 20.0

    def _get_filter_coefficients(self, sr, freq, gain_db, ftype):
        key = (sr, freq, gain_db, ftype)
        if key not in self._filter_cache:
            self._filter_cache[key] = self._design_shelf_filter(sr, freq, gain_db, ftype)
        return self._filter_cache[key]

    def _design_shelf_filter(self, sr, freq, gain_db, ftype):
        if sr <= 0 or freq <= 0 or freq >= sr / 2:
            return [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * freq / sr
        beta = np.sqrt(A + (1 / A))
        if ftype == 'low':
            b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + beta * np.sin(w0))
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - beta * np.sin(w0))
            a0 = (A + 1) + (A - 1) * np.cos(w0) + beta * np.sin(w0)
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
            a2 = (A + 1) + (A - 1) * np.cos(w0) - beta * np.sin(w0)
        else:
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + beta * np.sin(w0))
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - beta * np.sin(w0))
            a0 = (A + 1) - (A - 1) * np.cos(w0) + beta * np.sin(w0)
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - beta * np.sin(w0)
        return [b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0]

    def _get_gamma_lut(self, gamma: float):
        if self._gamma_cache_key != gamma or self._gamma_lut is None:
            inv = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
            self._gamma_lut = table
            self._gamma_cache_key = gamma
        return self._gamma_lut

    def generate_spectrogram_image(self, y: np.ndarray, sr: int, duration: float) -> Tuple[np.ndarray, int]:
        spec_type = self.settings.get('spec_type', 'stft')
        n_fft = self.settings.get('n_fft', 4096)
        hop_length = self.settings.get('hop_length', 512)
        dyn_range = self.settings.get('dynamic_range_db', 80)
        min_freq = self.settings.get('min_freq', 20.0)
        max_freq = self.settings.get('max_freq', 20000.0)

        if spec_type == 'mel':
            n_mels = min(256, self.settings.get('height', 1080))
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                                n_fft=n_fft, hop_length=hop_length,
                                                fmin=min_freq, fmax=max_freq)
            S_db = librosa.power_to_db(S, ref=np.max)
            if self.settings.get('auto_freq', True):
                min_freq, max_freq = self._detect_bounds_mel(S_db, sr, n_mels, min_freq, max_freq)
        elif spec_type == 'cqt':
            S = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length,
                                    fmin=librosa.note_to_hz('C1'),
                                    n_bins=84, bins_per_octave=12))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            if self.settings.get('auto_freq', True):
                min_freq, max_freq = self._detect_bounds_cqt(S_db, sr)
        else:
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            if self.settings.get('auto_freq', True):
                min_freq, max_freq = self._detect_bounds_stft(S_db, sr, n_fft)

        S_norm = (S_db + dyn_range) * (255.0 / dyn_range)
        gray = np.clip(S_norm, 0, 255).astype(np.uint8)

        gamma = self.settings.get('gamma', 1.0)
        if gamma != 1.0:
            gray = cv2.LUT(gray, self._get_gamma_lut(gamma))

        blur_amt = self.settings.get('blur', 0.0)
        if blur_amt > 0:
            ksize = int(blur_amt * 2) * 2 + 1
            gray = cv2.GaussianBlur(gray, (ksize, ksize), blur_amt)

        vis_mode = self.settings.get('visual_mode', 'standard')
        if vis_mode == 'radial':
            log_img = self._create_radial(gray, sr, n_fft, spec_type, min_freq, max_freq)
        elif vis_mode == 'classic':
            log_img = self._create_linear(gray)
        else:
            if spec_type == 'stft':
                log_img = self._create_log_freq(gray, sr, n_fft, min_freq, max_freq)
            else:
                log_img = self._create_linear(gray)

        if self.settings.get('sharpen', True):
            gauss = cv2.GaussianBlur(log_img, (0, 0), 3.0)
            log_img = cv2.addWeighted(log_img, 1.5, gauss, -0.5, 0)

        return log_img, hop_length

    # ── Bound detectors ──
    def _detect_bounds_stft(self, S_db, sr, n_fft):
        thr = -60.0
        mx = np.max(S_db)
        if not np.isfinite(mx):
            return 20.0, float(sr / 2)
        mask = np.max(S_db, axis=1) > (mx + thr)
        if not np.any(mask):
            return 20.0, float(sr / 2)
        idx = np.where(mask)[0]
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        return (float(freqs[max(0, idx[0] - 2)]),
                float(freqs[min(len(mask) - 1, idx[-1] + 2)]))

    def _detect_bounds_mel(self, S_db, sr, n_mels, fmin, fmax):
        thr = -60.0
        mx = np.max(S_db)
        if not np.isfinite(mx):
            return fmin, fmax
        mask = np.max(S_db, axis=1) > (mx + thr)
        if not np.any(mask):
            return fmin, fmax
        idx = np.where(mask)[0]
        mels = np.linspace(librosa.hz_to_mel(fmin), librosa.hz_to_mel(fmax), n_mels + 2)
        lo, hi = max(0, idx[0]), min(n_mels - 1, idx[-1])
        return (max(20.0, librosa.mel_to_hz(mels[lo])),
                min(float(sr/2), librosa.mel_to_hz(mels[hi + 2])))

    def _detect_bounds_cqt(self, S_db, sr):
        thr = -60.0
        mx = np.max(S_db)
        if not np.isfinite(mx):
            return 20.0, float(sr / 2)
        mask = np.max(S_db, axis=1) > (mx + thr)
        if not np.any(mask):
            return 20.0, float(sr / 2)
        idx = np.where(mask)[0]
        freqs = librosa.cqt_frequencies(S_db.shape[0], fmin=librosa.note_to_hz('C1'))
        return float(freqs[max(0, idx[0])]), float(freqs[min(len(freqs)-1, idx[-1])])

    # ── Image creators ──
    def _create_linear(self, img):
        target_h = self.settings.get('height', 1080)
        h, w = img.shape[:2]
        if w > 15000:
            img = cv2.resize(img, (15000, h), interpolation=cv2.INTER_AREA)
        return cv2.resize(img, (img.shape[1], target_h), interpolation=cv2.INTER_LANCZOS4)

    def _create_log_freq(self, img, sr, n_fft, min_freq, max_freq):
        target_h = self.settings.get('height', 1080)
        cache_key = (sr, n_fft, min_freq, max_freq, target_h, img.shape[1])
        if cache_key in self._remap_cache:
            mx, my = self._remap_cache[cache_key]
            if mx.shape[1] == img.shape[1]:
                return cv2.remap(img, mx, my, interpolation=cv2.INTER_LANCZOS4)
        if img.shape[1] > 20000:
            img = cv2.resize(img, (20000, img.shape[0]), interpolation=cv2.INTER_AREA)
        out_freqs = np.logspace(np.log10(max(1, min_freq)), np.log10(max_freq), target_h)
        bin_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        y_idx = np.interp(out_freqs, bin_freqs, np.arange(len(bin_freqs))).astype(np.float32)
        mx, my = np.meshgrid(np.arange(img.shape[1], dtype=np.float32), y_idx)
        self._remap_cache[cache_key] = (mx, my)
        log_img = cv2.remap(img, mx, my, interpolation=cv2.INTER_LANCZOS4)
        if log_img.shape[1] > 16000:
            log_img = cv2.resize(log_img, (16000, target_h), interpolation=cv2.INTER_AREA)
        return log_img

    def _create_radial(self, img, sr, n_fft, spec_type, min_freq, max_freq):
        dim = min(self.settings.get('width', 1920), self.settings.get('height', 1080))
        if dim < 100:
            dim = 800
        nyquist = sr / 2
        if spec_type == 'stft':
            bin_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        elif spec_type == 'mel':
            bin_freqs = np.linspace(0, nyquist, img.shape[0])
        else:
            bin_freqs = librosa.cqt_frequencies(n_bins=84, fmin=librosa.note_to_hz('C1'))
        out_freqs = np.logspace(np.log10(max(1, min_freq)), np.log10(max_freq), dim // 2)
        y_idx = np.interp(out_freqs, bin_freqs, np.arange(len(bin_freqs))).astype(np.float32)
        if img.shape[1] > 20000:
            img = cv2.resize(img, (20000, img.shape[0]), interpolation=cv2.INTER_AREA)
        mx, my = np.meshgrid(np.arange(img.shape[1], dtype=np.float32), y_idx)
        src_log = cv2.remap(img, mx, my, interpolation=cv2.INTER_LANCZOS4)
        src_polar = cv2.resize(src_log, (3600, dim // 2), interpolation=cv2.INTER_LANCZOS4)
        center = (dim // 2, dim // 2)
        max_radius = dim // 2
        flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP + cv2.INTER_LANCZOS4
        return cv2.warpPolar(src_polar, (dim, dim), center, max_radius, flags)

# ══════════════════════════════════════════════════════════════════════════════
# VIDEO RENDERER
# ══════════════════════════════════════════════════════════════════════════════

class VideoRenderer:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.luts = settings.get('luts', {})
        self.use_gpu = CUPY_AVAILABLE and settings.get('use_gpu', False)
        if self.use_gpu:
            try:
                cp.cuda.Device(0).use()
                print("[Renderer] GPU Acceleration Active (CuPy/CUDA)")
            except Exception as e:
                print(f"[Renderer] GPU Init Failed: {e}. Falling back to CPU.")
                self.use_gpu = False
        else:
            print("[Renderer] Running on CPU")

    def _get_lut(self, name: str) -> np.ndarray:
        lut = self.luts.get(name)
        if lut is not None:
            return lut
        for fb in ['Vibrant (Turbo)', 'Jet (Classic)']:
            if fb in self.luts:
                return self.luts[fb]
        return np.tile(np.arange(256, dtype=np.uint8).reshape(256, 1, 1), (1, 1, 3))

    def render_video_ffmpeg(self, audio_path, out_path, spec_img, y_audio, sr, dur,
                            seg_info, progress_cb, cancel_check):
        fps = self.settings['fps']
        window_sec = self.settings['window_size']
        tw, th = self.settings['width'], self.settings['height']
        if tw % 2 != 0: tw -= 1
        if th % 2 != 0: th -= 1

        total_frames = int(dur * fps)
        if total_frames == 0:
            raise ValueError("Zero duration")

        img_h, img_w = spec_img.shape[:2]
        px_per_sec = img_w / dur
        win_px = (window_sec / 2.0) * px_per_sec

        lut_data = self._get_lut(self.settings.get('colormap', 'Vibrant (Turbo)'))
        lut_rgb = cv2.cvtColor(lut_data, cv2.COLOR_BGR2RGB).reshape(256, 3)

        gpu_spec, gpu_lut = None, None
        if self.use_gpu:
            gpu_spec = cp.asarray(spec_img, dtype=cp.float32)
            gpu_lut = cp.asarray(lut_rgb, dtype=cp.float32) / 255.0
            batch_size = 30
        else:
            batch_size = 15

        encoder = VideoEncoder(tw, th, fps, audio_path, out_path, self.use_gpu)
        encoder.start()
        start_time = time.time()

        try:
            for batch_start in range(0, total_frames, batch_size):
                if cancel_check():
                    return

                batch_end = min(batch_start + batch_size, total_frames)

                if self.use_gpu:
                    frames = self._render_batch_gpu(
                        batch_start, batch_end, gpu_spec, gpu_lut,
                        tw, th, img_w, img_h, px_per_sec, win_px)
                    frames = frames.get()
                else:
                    frames = self._render_batch_cpu(
                        batch_start, batch_end, spec_img, lut_rgb,
                        tw, th, img_w, img_h, px_per_sec, win_px)

                for i, f_idx in enumerate(range(batch_start, batch_end)):
                    frame = frames[i]
                    if self.settings.get('draw_overlay', True):
                        self._draw_overlay(frame, f_idx, fps, dur)
                    if self.settings.get('center_line', True):
                        time_sec = f_idx / fps
                        cx_spec = int(time_sec * px_per_sec)
                        cx = max(0, min(img_w - 1, cx_spec))
                        spec_col = spec_img[:, cx]
                        self._draw_vertical_analyzer(frame, spec_col)

                if not encoder.write(frames.tobytes()):
                    raise Exception("FFmpeg process died.")

                if batch_start % (batch_size * 2) == 0:
                    elapsed = time.time() - start_time
                    speed = batch_end / elapsed if elapsed > 0 else 0
                    eta = int((total_frames - batch_end) / speed) if speed > 0 else 0
                    m, s = divmod(eta, 60)
                    progress_cb(int((batch_end / total_frames) * 100),
                                f"Rendering — ETA: {m:02d}:{s:02d}")

        finally:
            err = encoder.finish()
            if cancel_check():
                raise Exception("Cancelled")
            if encoder.process.returncode != 0:
                raise Exception(f"FFmpeg Error:\n{err[-500:]}")

    def _render_batch_cpu(self, start_idx, end_idx, spec_img, lut_rgb,
                          tw, th, sw, sh, pps, win_px):
        count = end_idx - start_idx
        results = np.zeros((count, th, tw, 3), dtype=np.uint8)
        fps = self.settings['fps']
        interp = cv2.INTER_LANCZOS4 if self.settings.get('interpolation') == 'lanczos' else cv2.INTER_CUBIC
        for i, frame_idx in enumerate(range(start_idx, end_idx)):
            time_sec = frame_idx / fps
            center_x = int(time_sec * pps)
            x1 = int(max(0, center_x - win_px))
            x2 = int(min(sw, center_x + win_px))
            if x2 <= x1:
                x2 = x1 + 1
            img_slice = spec_img[:, x1:x2]
            colored = lut_rgb[img_slice].astype(np.uint8)
            results[i] = cv2.resize(colored, (tw, th), interpolation=interp)
        return results

    def _render_batch_gpu(self, start_idx, end_idx, g_spec, g_lut,
                          tw, th, sw, sh, pps, win_px):
        if not CUPY_AVAILABLE:
            raise RuntimeError("GPU rendering not available (CuPy/Numba not installed)")
        count = end_idx - start_idx
        g_output = cp.zeros((count, th, tw, 3), dtype=cp.uint8)
        tpb = (8, 8, 1)
        bpg = (
            (tw + tpb[0] - 1) // tpb[0],
            (th + tpb[1] - 1) // tpb[1],
            count
        )
        fps = self.settings['fps']
        _cuda_render_kernel[bpg, tpb](
            g_spec, g_lut, g_output,
            start_idx, fps, tw, th, sw, sh, pps, win_px
        )
        return g_output

    def _draw_overlay(self, frame, frame_idx, fps, dur):
        current_time = frame_idx / fps
        m, s = divmod(int(current_time), 60)
        ms = int((current_time - int(current_time)) * 100)
        text = f"{m:02}:{s:02}.{ms:02}"
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_vertical_analyzer(self, frame, spec_column):
        h, w = frame.shape[:2]
        position = self.settings.get('analyzer_position', 'Center')
        bar_count = self.settings.get('analyzer_bar_count', 64)
        sensitivity = self.settings.get('line_magnitude_sensitivity', 800.0)
        base_width = self.settings.get('line_base_width', 6)
        glow_intensity = self.settings.get('line_glow_intensity', 15.0)
        bloom_val = self.settings.get('line_bloom', 8.0)
        show_center_line = self.settings.get('analyzer_show_center_line', True)

        x_old = np.linspace(0, 1, len(spec_column))
        x_new = np.linspace(0, 1, bar_count)
        bar_magnitudes = np.interp(x_new, x_old, spec_column)
        bar_h = h / bar_count

        mode = self.settings.get('analyzer_color_mode', 'Match Video')

        # Build bar layer
        bar_layer = np.zeros_like(frame)

        for i in range(bar_count):
            mag = bar_magnitudes[i]
            bar_len = max(base_width, int((mag / 255.0) * sensitivity))
            y1 = int(i * bar_h)
            y2 = int((i + 1) * bar_h) - 1

            if mode == 'Rainbow':
                hue = int(i * 180 / bar_count)
                bgr = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8),
                                   cv2.COLOR_HSV2BGR)[0][0].tolist()
            elif mode == 'Neon Gradient':
                bgr = (255, 0, 255)
            elif mode == 'Sunset Gradient':
                bgr = (0, 165, 255)
            elif mode == 'White':
                bgr = (255, 255, 255)
            elif mode == 'Match Video':
                bgr = frame[h // 2, w // 2].tolist()
            else:
                bgr = (255, 255, 255)

            if position == 'Center':
                xc = w // 2
                cv2.rectangle(bar_layer, (xc - bar_len // 2, y1),
                              (xc + bar_len // 2, y2), bgr, -1)
                cv2.rectangle(bar_layer, (xc - bar_len // 2, y1),
                              (xc + bar_len // 2, y2), (255, 255, 255), 1)
                if show_center_line:
                    cv2.line(bar_layer, (xc, y1), (xc, y2), (200, 200, 200), 1)
            else:
                cv2.rectangle(bar_layer, (0, y1), (bar_len, y2), bgr, -1)
                cv2.rectangle(bar_layer, (w - bar_len, y1), (w, y2), bgr, -1)

        # Glow
        if glow_intensity > 0:
            ksize_glow = int(glow_intensity * 2) * 2 + 1
            glow_layer = cv2.GaussianBlur(bar_layer, (ksize_glow, ksize_glow), glow_intensity)
            frame[:] = cv2.addWeighted(frame, 1.0, glow_layer, 0.6, 0)

        # Bloom
        if bloom_val > 0:
            ksize_bloom = int(bloom_val * 2) * 2 + 1
            bloom_layer = cv2.GaussianBlur(bar_layer, (ksize_bloom, ksize_bloom), bloom_val)
            frame[:] = cv2.addWeighted(frame, 1.0, bloom_layer, 0.35, 0)

        # Crisp bars on top
        mask = cv2.cvtColor(bar_layer, cv2.COLOR_BGR2GRAY) > 0
        frame[mask] = bar_layer[mask]


# ── CUDA Kernel (conditional) ────────────────────────────────────────────────
if CUPY_AVAILABLE:
    @cuda.jit
    def _cuda_render_kernel(spec_img, lut, output, start_frame, fps, tw, th, sw, sh, pps, win_px):
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        bz = cuda.blockIdx.z
        x = tx + bx * cuda.blockDim.x
        y = ty + by * cuda.blockDim.y
        batch_idx = bz
        if x >= tw or y >= th or batch_idx >= output.shape[0]:
            return
        frame_idx = start_frame + batch_idx
        time_sec = frame_idx / fps
        center_x = time_sec * pps
        norm_x = x / (tw - 1) if tw > 1 else 0.5
        src_x_f = center_x - win_px + (norm_x * 2 * win_px)
        norm_y = y / (th - 1) if th > 1 else 0.5
        src_y_f = norm_y * (sh - 1)
        if src_x_f < 0 or src_x_f >= sw or src_y_f < 0 or src_y_f >= sh:
            output[batch_idx, y, x, 0] = 0
            output[batch_idx, y, x, 1] = 0
            output[batch_idx, y, x, 2] = 0
            return
        x0 = int(src_x_f); y0 = int(src_y_f)
        x1 = x0 + 1;      y1 = y0 + 1
        x_w = src_x_f - x0; y_w = src_y_f - y0
        if x1 >= sw: x1 = x0
        if y1 >= sh: y1 = y0
        v00 = spec_img[y0, x0]; v01 = spec_img[y0, x1]
        v10 = spec_img[y1, x0]; v11 = spec_img[y1, x1]
        val = v00*(1-x_w)*(1-y_w) + v01*x_w*(1-y_w) + v10*(1-x_w)*y_w + v11*x_w*y_w
        idx = int(val)
        if idx > 255: idx = 255
        if idx < 0: idx = 0
        output[batch_idx, y, x, 0] = int(lut[idx, 2] * 255)
        output[batch_idx, y, x, 1] = int(lut[idx, 1] * 255)
        output[batch_idx, y, x, 2] = int(lut[idx, 0] * 255)

# ══════════════════════════════════════════════════════════════════════════════
# RENDER WORKER
# ══════════════════════════════════════════════════════════════════════════════

class RenderWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    log = Signal(str)
    error = Signal(str)
    batch_error = Signal(str, str)
    finished = Signal()
    file_started = Signal(str)
    file_finished = Signal(str)

    def __init__(self, file_list, output_target, settings):
        super().__init__()
        self.file_list = file_list
        self.output_target = output_target
        self.settings = settings
        self.is_cancelled = False
        self.processor = SpectrogramProcessor(settings)
        self.renderer = VideoRenderer(settings)
        if os.path.isfile(output_target):
            base_dir = os.path.dirname(output_target)
        else:
            base_dir = output_target
        self.local_temp_dir = os.path.join(base_dir if base_dir else os.getcwd(), "_temp_proc")
        try:
            os.makedirs(self.local_temp_dir, exist_ok=True)
        except OSError:
            self.local_temp_dir = None

    def run(self):
        try:
            if self.settings.get('merge_mode', False):
                self._process_concatenated()
            else:
                self._process_batch()
        except Exception as e:
            self.error.emit(f"Fatal Error:\n{traceback.format_exc()}")
        finally:
            self.progress.emit(100)
            self.finished.emit()

    def cancel(self):
        self.is_cancelled = True

    def _get_temp_dir(self):
        return self.local_temp_dir or tempfile.gettempdir()

    def _build_output_path(self, base_name, out_dir, index):
        suffix = self.settings.get('filename_suffix', '').strip()
        name = f"{base_name}{suffix}" if suffix else base_name
        out = os.path.join(out_dir, f"{name}.mp4")
        if os.path.exists(out):
            out = os.path.join(out_dir, f"{name}_{index}.mp4")
        counter = 1
        final_out = out
        while os.path.exists(final_out):
            stem, ext = os.path.splitext(out)
            final_out = f"{stem}_v{counter}{ext}"
            counter += 1
        return final_out

    def _cleanup_temp(self, *paths):
        for tmp in paths:
            try:
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass

    def _process_concatenated(self):
        self.log.emit("Merging files...")
        valid_files, total_dur, seg_info = [], 0.0, []
        for fpath in self.file_list:
            if self.is_cancelled:
                return
            try:
                info = sf.info(fpath)
                if info.duration > 0:
                    seg_info.append({'start': total_dur,
                                     'end': total_dur + info.duration,
                                     'name': os.path.basename(fpath)})
                    valid_files.append(fpath)
                    total_dur += info.duration
            except Exception:
                self.batch_error.emit(fpath, "Read Error")

        if not valid_files:
            self.log.emit("No valid files to merge.")
            return

        tmp = self._get_temp_dir()
        list_path = os.path.join(tmp, "concat.txt")
        temp_wav = os.path.join(tmp, "merged.wav")
        temp_proc = os.path.join(tmp, "merged_processed.wav")

        try:
            with open(list_path, 'w', encoding='utf-8') as wf:
                for p in valid_files:
                    escaped = p.replace('\\', '/').replace("'", "'\\''")
                    wf.write(f"file '{escaped}'\n")

            cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path,
                   '-ac', '1', '-ar', '44100', '-c:a', 'pcm_f32le', temp_wav]
            result = subprocess.run(cmd, stderr=subprocess.PIPE)
            if result.returncode != 0:
                err = result.stderr.decode('utf-8', errors='replace')[-500:]
                raise Exception(f"FFmpeg concat failed: {err}")

            y, sr = sf.read(temp_wav, dtype='float32')
            dur = len(y) / sr
            if dur == 0:
                raise Exception("Merged audio is empty")

            if self.settings.get('preview_mode', False):
                y = y[:int(min(15, dur) * sr)]
                dur = len(y) / sr

            y_proc = self.processor.process_audio(y, sr)
            sf.write(temp_proc, y_proc, sr)
            spec, _ = self.processor.generate_spectrogram_image(y_proc, sr, dur)

            out = (self.output_target if os.path.isfile(self.output_target)
                   else os.path.join(self.output_target, "Merged.mp4"))

            self.log.emit(f"Rendering merged video ({dur:.1f}s)...")
            self.renderer.render_video_ffmpeg(
                temp_proc, out, spec, y_proc, sr, dur, seg_info,
                lambda p, s: (self.progress.emit(p), self.status.emit(s)),
                lambda: self.is_cancelled
            )
            if not self.is_cancelled:
                self.log.emit(f"Merged output saved: {out}")
        finally:
            self._cleanup_temp(list_path, temp_wav, temp_proc)

    def _process_batch(self):
        out_dir = self.output_target
        if not os.path.isdir(out_dir):
            out_dir = os.path.dirname(out_dir) or os.getcwd()
            try:
                os.makedirs(out_dir, exist_ok=True)
            except OSError:
                out_dir = os.getcwd()

        total = len(self.file_list)
        for i, fpath in enumerate(self.file_list):
            if self.is_cancelled:
                return
            self.file_started.emit(fpath)
            self.log.emit(f"Processing ({i + 1}/{total}): {os.path.basename(fpath)}")

            temp_proc_wav = None
            try:
                y, sr = sf.read(fpath, dtype='float32')
                if y.ndim > 1:
                    y = y.mean(axis=1)
                dur = len(y) / sr
                if dur == 0:
                    raise Exception("Empty audio")

                y_proc = self.processor.process_audio(y, sr)

                if self.settings.get('preview_mode', False):
                    y_proc = y_proc[:int(min(15, dur) * sr)]
                    dur = len(y_proc) / sr

                temp_proc_wav = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, dir=self._get_temp_dir()).name
                sf.write(temp_proc_wav, y_proc, sr)

                spec, _ = self.processor.generate_spectrogram_image(y_proc, sr, dur)
                base_name = os.path.splitext(os.path.basename(fpath))[0]
                out = self._build_output_path(base_name, out_dir, i)

                self.log.emit(f"Rendering {base_name} ({dur:.1f}s)...")
                self.renderer.render_video_ffmpeg(
                    temp_proc_wav, out, spec, y_proc, sr, dur,
                    [{'start': 0, 'end': dur}],
                    lambda p, s: (self.progress.emit(p), self.status.emit(s)),
                    lambda: self.is_cancelled
                )
                if not self.is_cancelled:
                    self.file_finished.emit(fpath)
                    self.log.emit(f"Finished: {os.path.basename(out)}")

            except Exception as e:
                self.batch_error.emit(fpath, str(e))
            finally:
                self._cleanup_temp(temp_proc_wav)

            if not self.is_cancelled and total > 0:
                self.progress.emit(int(((i + 1) / total) * 100))

        try:
            if self.local_temp_dir and os.path.isdir(self.local_temp_dir):
                if not os.listdir(self.local_temp_dir):
                    os.rmdir(self.local_temp_dir)
        except OSError:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

class AudioTableWidget(QTableWidget):
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.CopyAction)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Filename", "Duration", "Status"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)

    def _show_context_menu(self, pos):
        item = self.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)
        menu.addAction("Remove Selected", self._remove_selected)
        path_item = self.item(item.row(), 0)
        if path_item:
            fpath = path_item.data(Qt.UserRole)
            menu.addAction("Open Containing Folder", lambda: open_file_location(fpath))
        menu.exec_(self.mapToGlobal(pos))

    def _remove_selected(self):
        for row in sorted({i.row() for i in self.selectedItems()}, reverse=True):
            self.removeRow(row)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in e.mimeData().urls()]
            self.files_dropped.emit(paths)
            e.acceptProposedAction()

# ══════════════════════════════════════════════════════════════════════════════
# YOUTUBE DOWNLOAD DIALOG
# ══════════════════════════════════════════════════════════════════════════════

class _DownloadThread(QThread):
    progress_signal = Signal(int)
    status_signal = Signal(str)
    log_signal = Signal(str)
    finished_signal = Signal(bool, list)  # success, list of downloaded files

    def __init__(self, url, output_dir):
        super().__init__()
        self.url = url
        self.output_dir = output_dir
        self.is_cancelled = False
        self._downloaded = []

    def run(self):
        if not YT_DLP_AVAILABLE:
            self.log_signal.emit("yt-dlp is not installed. Run: pip install yt-dlp")
            self.finished_signal.emit(False, [])
            return

        out_dir = self.output_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{out_dir}/%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '0',
            }],
            'sleep_interval': 5,
            'ignoreerrors': True,
            'geo_bypass': True,
            'progress_hooks': [self._progress_hook],
            'quiet': True,
            'no_warnings': True,
            'logger': self._YtdlLogger(self.log_signal),
        }

        try:
            self.status_signal.emit("Starting download...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url])
            self.finished_signal.emit(True, self._downloaded)
        except Exception as e:
            self.log_signal.emit(f"Critical Error: {str(e)}")
            self.finished_signal.emit(False, self._downloaded)

    class _YtdlLogger:
        def __init__(self, signal):
            self.signal = signal
        def debug(self, msg):
            if '[download]' in msg or '[info]' in msg:
                self.signal.emit(msg)
        def warning(self, msg):
            self.signal.emit(f"Warning: {msg}")
        def error(self, msg):
            self.signal.emit(f"Error: {msg}")

    def _progress_hook(self, d):
        if self.is_cancelled:
            raise Exception("Download cancelled by user")
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded = d.get('downloaded_bytes', 0)
            if total:
                self.progress_signal.emit(int(downloaded * 100 / total))
            self.status_signal.emit(f"Downloading: {os.path.basename(d.get('filename', 'file'))}")
        elif d['status'] == 'finished':
            self.status_signal.emit("Processing (Converting to MP3)...")
            self.progress_signal.emit(100)
            filename = d.get('filename', '')
            if filename:
                mp3_name = os.path.splitext(filename)[0] + '.mp3'
                if os.path.exists(mp3_name):
                    self._downloaded.append(mp3_name)


class DownloadDialog(QDialog):
    files_downloaded = Signal(list)

    def __init__(self, default_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YouTube Audio Downloader")
        self.resize(560, 380)
        self.download_thread = None
        self._default_dir = default_dir
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("YouTube URL:"))
        self.input_url = QLineEdit()
        self.input_url.setPlaceholderText("Paste link here (Video or Playlist)")
        layout.addWidget(self.input_url)

        row = QHBoxLayout()
        row.addWidget(QLabel("Save to:"))
        self.lbl_dir = QLabel(self._default_dir)
        self.lbl_dir.setStyleSheet("color: #5dade2; font-weight: bold;")
        self.lbl_dir.setWordWrap(True)
        btn_dir = QPushButton("Change...")
        btn_dir.setFixedWidth(80)
        btn_dir.clicked.connect(self._change_dir)
        row.addWidget(self.lbl_dir, 1)
        row.addWidget(btn_dir)
        layout.addLayout(row)

        self.btn_download = QPushButton("Download")
        self.btn_download.setStyleSheet(
            "background-color:#2ecc71; color:white; font-weight:bold; padding:10px;")
        self.btn_download.clicked.connect(self._start_download)
        layout.addWidget(self.btn_download)

        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)

        self.label_status = QLabel("Ready")
        self.label_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_status)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(120)
        layout.addWidget(self.log_area)

    def _change_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Download Folder", self._default_dir)
        if d:
            self._default_dir = d
            self.lbl_dir.setText(d)

    def _start_download(self):
        url = self.input_url.text().strip()
        if not url:
            self.log_area.append("Please enter a valid URL.")
            return
        self.btn_download.setEnabled(False)
        self.btn_download.setText("Downloading...")
        self.log_area.clear()
        self.log_area.append(f"Target: {url}")
        self.log_area.append(f"Files will be saved in: {self._default_dir}")

        self.download_thread = _DownloadThread(url, self._default_dir)
        self.download_thread.progress_signal.connect(self.progress_bar.setValue)
        self.download_thread.status_signal.connect(self.label_status.setText)
        self.download_thread.log_signal.connect(self.log_area.append)
        self.download_thread.finished_signal.connect(self._on_finished)
        self.download_thread.start()

    def _on_finished(self, success, downloaded_files):
        self.btn_download.setEnabled(True)
        self.btn_download.setText("Download")
        if success:
            self.label_status.setText("Process finished.")
            self.progress_bar.setValue(100)
            if downloaded_files:
                self.files_downloaded.emit(downloaded_files)
        else:
            self.label_status.setText("Download failed or cancelled.")

# ══════════════════════════════════════════════════════════════════════════════
# BULK FILE RENAMER DIALOG
# ══════════════════════════════════════════════════════════════════════════════

class FileRenamerDialog(QDialog):
    def __init__(self, default_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bulk File Renamer")
        self.resize(820, 560)
        self._folder_path = default_dir
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Folder row
        row_dir = QHBoxLayout()
        row_dir.addWidget(QLabel("Folder:"))
        self.lbl_folder = QLabel(self._folder_path)
        self.lbl_folder.setStyleSheet("color: #5dade2; font-weight: bold;")
        self.lbl_folder.setWordWrap(True)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._browse_folder)
        row_dir.addWidget(self.lbl_folder, 1)
        row_dir.addWidget(btn_browse)
        layout.addLayout(row_dir)

        # Options
        grp = QGroupBox("Renaming Options")
        grp_layout = QVBoxLayout(grp)
        self.chk_spaces = QCheckBox("Replace spaces with underscores (_)")
        self.chk_spaces.setChecked(True)
        self.chk_special = QCheckBox("Remove special characters (keep letters, numbers, dashes, dots)")
        self.chk_special.setChecked(True)
        self.chk_lowercase = QCheckBox("Convert filename to lowercase")
        self.chk_lowercase.setChecked(False)
        grp_layout.addWidget(self.chk_spaces)
        grp_layout.addWidget(self.chk_special)
        grp_layout.addWidget(self.chk_lowercase)
        layout.addWidget(grp)

        # Buttons
        row_btn = QHBoxLayout()
        btn_preview = QPushButton("Preview Changes")
        btn_preview.clicked.connect(self._preview_changes)
        btn_apply = QPushButton("Apply Changes")
        btn_apply.setStyleSheet("background-color:#2ecc71; color:white; font-weight:bold; padding:8px;")
        btn_apply.clicked.connect(self._apply_changes)
        row_btn.addWidget(btn_preview)
        row_btn.addWidget(btn_apply)
        row_btn.addStretch()
        layout.addLayout(row_btn)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Original Name", "New Name"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder", self._folder_path)
        if d:
            self._folder_path = d
            self.lbl_folder.setText(d)
            self._preview_changes()

    def _clean_filename(self, filename):
        name, ext = os.path.splitext(filename)
        if self.chk_special.isChecked():
            name = re.sub(r"[^\w\s\-\.]", "", name)
        name = re.sub(r"\s*-\s*", "-", name)
        if self.chk_spaces.isChecked():
            name = name.replace(" ", "_")
        if self.chk_lowercase.isChecked():
            name = name.lower()
            ext = ext.lower()

        def clean_sep(match):
            s = match.group()
            return '-' if '-' in s else '_'
        name = re.sub(r"[_\-]{2,}", clean_sep, name)
        name = name.strip("_- ")
        if not name:
            name = "file"
        return name + ext

    def _get_rename_map(self):
        folder = self._folder_path
        if not folder or not os.path.isdir(folder):
            return []
        result = []
        for f in os.listdir(folder):
            fp = os.path.join(folder, f)
            if not os.path.isfile(fp):
                continue
            new_f = self._clean_filename(f)
            if new_f != f:
                result.append((f, new_f, fp, os.path.join(folder, new_f)))
        return result

    def _preview_changes(self):
        self.table.setRowCount(0)
        rename_list = self._get_rename_map()
        if not rename_list:
            if self._folder_path:
                QMessageBox.information(self, "Info",
                                        "No files need renaming based on current options.")
            return
        for old, new, _, _ in rename_list:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(old))
            self.table.setItem(row, 1, QTableWidgetItem(new))

    def _apply_changes(self):
        rename_list = self._get_rename_map()
        if not rename_list:
            QMessageBox.information(self, "Info", "No files to rename.")
            return
        confirm = QMessageBox.question(
            self, "Confirm",
            f"{len(rename_list)} files will be renamed.\n\n"
            "WARNING: If a target filename already exists,\n"
            "the existing file will be DELETED (overwritten).\n\n"
            "Continue?")
        if confirm != QMessageBox.Yes:
            return

        ok, fail = 0, 0
        rename_list.sort(key=lambda x: x[0])
        for old_name, new_name, old_path, new_path in rename_list:
            try:
                if os.path.exists(new_path):
                    if os.path.normcase(old_path) == os.path.normcase(new_path):
                        tmp = new_path + ".tmp_rename_placeholder"
                        os.rename(old_path, tmp)
                        old_path = tmp
                    else:
                        os.remove(new_path)
                os.rename(old_path, new_path)
                ok += 1
            except Exception as e:
                fail += 1
                print(f"Error renaming {old_name}: {e}")

        QMessageBox.information(self, "Result",
                                f"Complete.\nRenamed: {ok}\nErrors: {fail}")
        self._preview_changes()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrogram Studio Pro v9.6")
        self.resize(1120, 1000)

        if not shutil.which("ffmpeg"):
            QMessageBox.critical(self, "Error",
                                 "FFmpeg not found! Please install FFmpeg and add it to your PATH.")
            sys.exit(1)

        self.worker = None
        self.config_manager = ConfigManager()

        saved_cfg = self.config_manager.load()
        saved_settings = saved_cfg.get('settings', {})
        self._palette_folder = saved_settings.get('palette_folder', '').strip()
        if not self._palette_folder:
            self._palette_folder = _get_default_palette_dir()
        self.available_luts = get_all_luts(self._palette_folder)

        saved_load = saved_settings.get('default_load_folder', '').strip()
        self.default_load_folder = (
            saved_load if saved_load and os.path.isdir(saved_load)
            else _get_default_load_folder())
        saved_last = saved_settings.get('last_input_dir', '').strip()
        self.last_input_dir = (
            saved_last if saved_last and os.path.isdir(saved_last)
            else self.default_load_folder)

        self.output_target_batch = os.path.join(os.getcwd(), "Output")
        self.output_target_merge = os.path.join(self.output_target_batch, "Merged_Output.mp4")
        self.output_target = self.output_target_batch
        self._preview_path = None
        self.error_count = 0

        os.makedirs(self.output_target_batch, exist_ok=True)

        self._init_ui()
        self._init_menu()
        self._load_settings()
        self._setup_autosave()

        if self.default_load_folder and os.path.isdir(self.default_load_folder):
            self._add_files_from_paths([self.default_load_folder])

    # ── UI ────────────────────────────────────────────────────────────────
    def _init_ui(self):
        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #e0e0e0;
                      font-family: 'Segoe UI', Arial; }
            QGroupBox { border: 1px solid #3a3a3a; border-radius: 5px;
                        margin-top: 20px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px;
                               padding: 0 5px; }
            QPushButton { background-color: #3c3c3c; border: 1px solid #555;
                          border-radius: 4px; padding: 8px; }
            QPushButton:hover { background-color: #4d4d4d; }
            QPushButton:disabled { background-color: #2a2a2a; color: #666; }
            QProgressBar::chunk { background-color: #2ecc71; border-radius: 3px; }
            QComboBox { padding: 4px; }
            QSpinBox, QDoubleSpinBox, QLineEdit { padding: 3px; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(8)

        # ── 1. Source Files ──
        grp_src = QGroupBox("Source Files")
        lsrc = QVBoxLayout(grp_src)

        r_def = QHBoxLayout()
        r_def.addWidget(QLabel("Default Load Folder:"))
        self.lbl_default_folder = QLabel(self.default_load_folder)
        self.lbl_default_folder.setStyleSheet("color:#5dade2; font-weight:bold;")
        self.lbl_default_folder.setWordWrap(True)
        btn_ch_def = QPushButton("Change...")
        btn_ch_def.setFixedWidth(80)
        btn_ch_def.clicked.connect(self._change_default_load_folder)
        r_def.addWidget(self.lbl_default_folder, 1)
        r_def.addWidget(btn_ch_def)
        lsrc.addLayout(r_def)

        ctrl = QHBoxLayout()
        self.btn_add_files = QPushButton("Add Files")
        self.btn_add_files.clicked.connect(self._add_files)
        self.btn_load_folder = QPushButton("Load Folder")
        self.btn_load_folder.clicked.connect(self._load_all_from_folder)
        self.btn_youtube = QPushButton("YouTube Download")
        self.btn_youtube.setStyleSheet("background-color:#c0392b; color:white;")
        self.btn_youtube.clicked.connect(self._open_youtube_dialog)
        self.chk_recursive = QCheckBox("Recursive")
        self.chk_recursive.setChecked(True)
        ctrl.addWidget(self.btn_add_files)
        ctrl.addWidget(self.btn_load_folder)
        ctrl.addWidget(self.btn_youtube)
        ctrl.addWidget(self.chk_recursive)
        ctrl.addStretch()
        lsrc.addLayout(ctrl)

        self.table_widget = AudioTableWidget()
        self.table_widget.files_dropped.connect(self._add_files_from_paths)
        lsrc.addWidget(self.table_widget)
        main_layout.addWidget(grp_src)

        # ── 2. Output & Video ──
        grp_vid = QGroupBox("Output Settings")
        lvid = QVBoxLayout(grp_vid)

        r_out = QHBoxLayout()
        self.btn_output = QPushButton("Save Location...")
        self.btn_output.clicked.connect(self._select_output)
        self.lbl_output = QLabel(self.output_target)
        self.lbl_output.setStyleSheet("font-weight:bold; color:#5dade2;")
        self.lbl_output.setWordWrap(True)
        r_out.addWidget(self.btn_output)
        r_out.addWidget(self.lbl_output, 1)
        lvid.addLayout(r_out)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Resolution:"))
        self.combo_res = QComboBox()
        self.combo_res.addItems(['720p', '1080p', '1440p', '2160p'])
        self.combo_res.setCurrentIndex(1)
        r1.addWidget(self.combo_res)
        r1.addWidget(QLabel("FPS:"))
        self.spin_fps = QSpinBox(); self.spin_fps.setRange(10, 60); self.spin_fps.setValue(30)
        r1.addWidget(self.spin_fps)
        self.chk_gpu = QCheckBox("GPU (NVENC)"); self.chk_gpu.setChecked(True)
        r1.addWidget(self.chk_gpu)
        r1.addStretch()
        lvid.addLayout(r1)

        r2 = QHBoxLayout()
        self.chk_merge = QCheckBox("Merge into One Video")
        self.chk_merge.stateChanged.connect(self._toggle_output_mode)
        r2.addWidget(self.chk_merge)
        r2.addWidget(QLabel("Filename Suffix:"))
        self.txt_suffix = QLineEdit(); self.txt_suffix.setMaximumWidth(150)
        self.txt_suffix.setPlaceholderText("_spectrogram")
        r2.addWidget(self.txt_suffix)
        r2.addStretch()
        lvid.addLayout(r2)
        main_layout.addWidget(grp_vid)

        # ── 3. Spectrogram Engine ──
        grp_spec = QGroupBox("Spectrogram Engine")
        lsp = QVBoxLayout(grp_spec)

        ra = QHBoxLayout()
        ra.addWidget(QLabel("Algorithm:"))
        self.combo_spec_type = QComboBox()
        self.combo_spec_type.addItems(['Standard (STFT)', 'Mel-Scaled', 'CQT'])
        ra.addWidget(self.combo_spec_type)
        ra.addWidget(QLabel("Display:"))
        self.combo_vis_mode = QComboBox()
        self.combo_vis_mode.addItems(['Standard', 'Classic', 'Radial'])
        ra.addWidget(self.combo_vis_mode)
        ra.addStretch()
        lsp.addLayout(ra)

        rc = QHBoxLayout()
        rc.addWidget(QLabel("Palette:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(sorted(self.available_luts.keys()))
        self.combo_cmap.currentIndexChanged.connect(self._update_palette_preview)
        rc.addWidget(self.combo_cmap)
        self.palette_preview = QLabel()
        self.palette_preview.setFixedHeight(22)
        self.palette_preview.setMinimumWidth(180)
        self.palette_preview.setStyleSheet("border:1px solid #555; border-radius:3px;")
        rc.addWidget(self.palette_preview, 1)
        rc.addWidget(QLabel("Color Cycle:"))
        self.combo_color_mode = QComboBox()
        self.combo_color_mode.addItems(['Static', 'Rainbow Cycle'])
        rc.addWidget(self.combo_color_mode)
        rc.addStretch()
        lsp.addLayout(rc)

        rg = QHBoxLayout()
        rg.addWidget(QLabel("FFT:"))
        self.combo_nfft = QComboBox()
        self.combo_nfft.addItems(['2048', '4096', '8192', '16384'])
        self.combo_nfft.setCurrentIndex(1)
        rg.addWidget(self.combo_nfft)
        rg.addWidget(QLabel("Hop:"))
        self.spin_hop = QSpinBox(); self.spin_hop.setRange(64, 4096); self.spin_hop.setValue(512)
        rg.addWidget(self.spin_hop)
        rg.addWidget(QLabel("Interp:"))
        self.combo_interp = QComboBox()
        self.combo_interp.addItems(['Linear', 'Cubic', 'Lanczos'])
        self.combo_interp.setCurrentIndex(2)
        rg.addWidget(self.combo_interp)
        self.chk_sharpen = QCheckBox("Sharpen"); self.chk_sharpen.setChecked(True)
        rg.addWidget(self.chk_sharpen)
        rg.addStretch()
        lsp.addLayout(rg)

        rf = QHBoxLayout()
        self.chk_auto_freq = QCheckBox("Auto Hz"); self.chk_auto_freq.setChecked(True)
        self.chk_auto_freq.stateChanged.connect(self._toggle_auto_freq)
        rf.addWidget(self.chk_auto_freq)
        rf.addWidget(QLabel("Min:"))
        self.spin_freq_min = QSpinBox(); self.spin_freq_min.setRange(1, 20000); self.spin_freq_min.setValue(20)
        rf.addWidget(self.spin_freq_min)
        rf.addWidget(QLabel("Max:"))
        self.spin_freq_max = QSpinBox(); self.spin_freq_max.setRange(20, 22000); self.spin_freq_max.setValue(20000)
        rf.addWidget(self.spin_freq_max)
        rf.addStretch()
        lsp.addLayout(rf)

        rfx = QHBoxLayout()
        self.chk_overlay = QCheckBox("Time Overlay"); self.chk_overlay.setChecked(True)
        rfx.addWidget(self.chk_overlay)
        rfx.addWidget(QLabel("Window:"))
        self.spin_window = QDoubleSpinBox(); self.spin_window.setRange(1, 60); self.spin_window.setValue(5.0)
        rfx.addWidget(self.spin_window)
        rfx.addWidget(QLabel("Dyn:"))
        self.spin_dyn_range = QSpinBox(); self.spin_dyn_range.setRange(20, 150); self.spin_dyn_range.setValue(80)
        rfx.addWidget(self.spin_dyn_range)
        rfx.addWidget(QLabel("Gamma:"))
        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.1, 5.0); self.spin_gamma.setValue(1.0); self.spin_gamma.setSingleStep(0.1)
        rfx.addWidget(self.spin_gamma)
        rfx.addWidget(QLabel("Blur:"))
        self.spin_blur = QDoubleSpinBox()
        self.spin_blur.setRange(0.0, 10.0); self.spin_blur.setValue(0.0); self.spin_blur.setSingleStep(0.5)
        rfx.addWidget(self.spin_blur)
        rfx.addStretch()
        lsp.addLayout(rfx)
        main_layout.addWidget(grp_spec)

        # ── 4. Audio Processing ──
        grp_aud = QGroupBox("Audio Processing")
        laud = QHBoxLayout(grp_aud)
        self.chk_normalize = QCheckBox("Normalize"); self.chk_normalize.setChecked(True)
        laud.addWidget(self.chk_normalize)
        self.chk_auto_hp = QCheckBox("Auto Clean Sub-Bass"); self.chk_auto_hp.setChecked(True)
        self.chk_auto_hp.setToolTip("Automatically detects and removes inaudible sub-bass rumble/noise.")
        self.chk_auto_hp.stateChanged.connect(self._toggle_auto_highpass)
        laud.addWidget(self.chk_auto_hp)
        laud.addWidget(QLabel("High-Pass:"))
        self.spin_highpass = QSpinBox(); self.spin_highpass.setRange(0, 200); self.spin_highpass.setValue(20); self.spin_highpass.setSuffix(" Hz")
        laud.addWidget(self.spin_highpass)
        laud.addWidget(QLabel("Bass:"))
        self.spin_bass = QDoubleSpinBox(); self.spin_bass.setRange(-12, 12); self.spin_bass.setSuffix(" dB")
        laud.addWidget(self.spin_bass)
        laud.addWidget(QLabel("Treble:"))
        self.spin_treble = QDoubleSpinBox(); self.spin_treble.setRange(-12, 12); self.spin_treble.setSuffix(" dB")
        laud.addWidget(self.spin_treble)
        laud.addStretch()
        main_layout.addWidget(grp_aud)

        # ── 5. Center Visualizer ──
        grp_vis = QGroupBox("Center Visualizer (Spectrum Analyzer)")
        lvis = QVBoxLayout(grp_vis)

        rv1 = QHBoxLayout()
        self.chk_center_line = QCheckBox("Enable Visualizer"); self.chk_center_line.setChecked(True)
        rv1.addWidget(self.chk_center_line)
        self.chk_draw_center_line = QCheckBox("Draw Center Line"); self.chk_draw_center_line.setChecked(True)
        rv1.addWidget(self.chk_draw_center_line)
        rv1.addWidget(QLabel("Position:"))
        self.combo_analyzer_pos = QComboBox()
        self.combo_analyzer_pos.addItems(['Center', 'Edges'])
        rv1.addWidget(self.combo_analyzer_pos)
        rv1.addWidget(QLabel("Color Style:"))
        self.combo_analyzer_color = QComboBox()
        self.combo_analyzer_color.addItems(['Match Video', 'Neon Gradient', 'Sunset Gradient', 'White', 'Rainbow'])
        rv1.addWidget(self.combo_analyzer_color)
        rv1.addStretch()
        lvis.addLayout(rv1)

        rv2 = QHBoxLayout()
        rv2.addWidget(QLabel("Bars:"))
        self.spin_bar_count = QSpinBox(); self.spin_bar_count.setRange(2, 256); self.spin_bar_count.setValue(64)
        rv2.addWidget(self.spin_bar_count)
        rv2.addWidget(QLabel("Width:"))
        self.spin_line_width = QSpinBox(); self.spin_line_width.setRange(1, 200); self.spin_line_width.setValue(6)
        rv2.addWidget(self.spin_line_width)
        rv2.addWidget(QLabel("Sensitivity:"))
        self.spin_line_sensitivity = QDoubleSpinBox()
        self.spin_line_sensitivity.setRange(0, 2000); self.spin_line_sensitivity.setValue(800); self.spin_line_sensitivity.setSingleStep(50)
        rv2.addWidget(self.spin_line_sensitivity)
        rv2.addWidget(QLabel("Glow:"))
        self.spin_line_glow = QDoubleSpinBox()
        self.spin_line_glow.setRange(0, 50); self.spin_line_glow.setValue(15); self.spin_line_glow.setSingleStep(1)
        rv2.addWidget(self.spin_line_glow)
        rv2.addWidget(QLabel("Bloom:"))
        self.spin_line_bloom = QDoubleSpinBox()
        self.spin_line_bloom.setRange(0, 50); self.spin_line_bloom.setValue(8); self.spin_line_bloom.setSingleStep(1)
        rv2.addWidget(self.spin_line_bloom)
        rv2.addStretch()
        lvis.addLayout(rv2)
        main_layout.addWidget(grp_vis)

        # ── Controls ──
        rl = QHBoxLayout()
        self.btn_start = QPushButton("Start Render")
        self.btn_start.setStyleSheet("background-color:#2ecc71; color:white; font-weight:bold; padding:15px;")
        self.btn_start.clicked.connect(self._start_processing)
        self.btn_preview = QPushButton("Preview (15s)")
        self.btn_preview.setStyleSheet("background-color:#f1c40f; color:black; font-weight:bold; padding:15px;")
        self.btn_preview.clicked.connect(self._start_preview)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet("background-color:#e74c3c; color:white; padding:15px;")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_processing)
        rl.addWidget(self.btn_start)
        rl.addWidget(self.btn_preview)
        rl.addWidget(self.btn_cancel)
        main_layout.addLayout(rl)

        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(110)
        main_layout.addWidget(self.log_widget)

        self._update_palette_preview()
        self._toggle_auto_freq()
        self._toggle_auto_highpass()

    def _init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Reset Settings", self._reset_settings)
        file_menu.addAction("Set Palette Folder...", self._change_palette_folder)
        file_menu.addAction("Open Palette Folder",
                            lambda: open_file_location(self._palette_folder))
        file_menu.addSeparator()
        file_menu.addAction("Open Output Folder",
                            lambda: open_file_location(self.output_target))
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction("YouTube Audio Downloader...", self._open_youtube_dialog)
        tools_menu.addAction("Bulk File Renamer...", self._open_renamer_dialog)

    # ── Toggles ──
    def _toggle_auto_freq(self):
        a = self.chk_auto_freq.isChecked()
        self.spin_freq_min.setEnabled(not a)
        self.spin_freq_max.setEnabled(not a)

    def _toggle_auto_highpass(self):
        self.spin_highpass.setEnabled(not self.chk_auto_hp.isChecked())

    def _toggle_output_mode(self):
        is_merge = self.chk_merge.isChecked()
        self.output_target = self.output_target_merge if is_merge else self.output_target_batch
        self.lbl_output.setText(self.output_target)

    # ── Palette preview ──
    def _update_palette_preview(self):
        name = self.combo_cmap.currentText()
        lut = self.available_luts.get(name)
        if lut is None:
            self.palette_preview.setText("(no palette)")
            return
        try:
            grad = np.arange(256, dtype=np.uint8).reshape(1, 256)
            colored_rgb = cv2.cvtColor(cv2.applyColorMap(grad, lut), cv2.COLOR_BGR2RGB)
            display = cv2.resize(colored_rgb, (256, 20), interpolation=cv2.INTER_LINEAR)
            h, w, ch = display.shape
            self._preview_data = display.copy()
            qimg = QImage(self._preview_data.data, w, h, ch * w, QImage.Format_RGB888)
            self.palette_preview.setPixmap(
                QPixmap.fromImage(qimg).scaled(
                    max(1, self.palette_preview.width()), 20,
                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        except Exception:
            self.palette_preview.setText("(error)")

    # ── Settings persistence ──
    def _setup_autosave(self):
        widgets = [
            self.combo_res, self.spin_fps, self.chk_gpu, self.chk_merge, self.txt_suffix,
            self.combo_spec_type, self.combo_vis_mode, self.combo_cmap, self.combo_color_mode,
            self.combo_nfft, self.spin_hop, self.combo_interp, self.chk_sharpen, self.chk_overlay,
            self.spin_window, self.chk_auto_freq, self.spin_freq_min, self.spin_freq_max,
            self.spin_dyn_range, self.spin_gamma, self.spin_blur,
            self.chk_normalize, self.chk_auto_hp, self.spin_highpass, self.spin_bass, self.spin_treble,
            self.chk_center_line, self.combo_analyzer_color, self.spin_line_width,
            self.spin_line_sensitivity, self.spin_line_glow, self.spin_line_bloom,
            self.spin_bar_count, self.combo_analyzer_pos, self.chk_draw_center_line
        ]
        for w in widgets:
            try:
                if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                    w.valueChanged.connect(self._save_settings)
                elif isinstance(w, QComboBox):
                    w.currentIndexChanged.connect(self._save_settings)
                elif isinstance(w, QCheckBox):
                    w.stateChanged.connect(self._save_settings)
                elif isinstance(w, QLineEdit):
                    w.textChanged.connect(self._save_settings)
            except Exception:
                pass

    def _log(self, text, level="INFO"):
        ts = time.strftime("%H:%M:%S")
        colors = {"INFO": "white", "WARN": "#F39C12",
                  "ERROR": "#E74C3C", "SUCCESS": "#2ECC71"}
        self.log_widget.append(
            f'<span style="color:{colors.get(level, "white")};">[{ts}] {text}</span>')
        self.log_widget.moveCursor(QTextCursor.End)

    def _save_settings(self):
        s = {
            'resolution': self.combo_res.currentIndex(),
            'fps': self.spin_fps.value(),
            'gpu': self.chk_gpu.isChecked(),
            'merge': self.chk_merge.isChecked(),
            'suffix': self.txt_suffix.text(),
            'spec_type': self.combo_spec_type.currentText(),
            'visual_mode': self.combo_vis_mode.currentText(),
            'colormap': self.combo_cmap.currentText(),
            'color_mode': self.combo_color_mode.currentText(),
            'n_fft': int(self.combo_nfft.currentText()),
            'hop_length': self.spin_hop.value(),
            'interpolation': self.combo_interp.currentText().lower(),
            'sharpen': self.chk_sharpen.isChecked(),
            'overlay': self.chk_overlay.isChecked(),
            'window': self.spin_window.value(),
            'auto_freq': self.chk_auto_freq.isChecked(),
            'min_freq': self.spin_freq_min.value(),
            'max_freq': self.spin_freq_max.value(),
            'dyn_range': self.spin_dyn_range.value(),
            'gamma': self.spin_gamma.value(),
            'blur': self.spin_blur.value(),
            'normalize': self.chk_normalize.isChecked(),
            'auto_highpass': self.chk_auto_hp.isChecked(),
            'highpass_freq': self.spin_highpass.value(),
            'bass': self.spin_bass.value(),
            'treble': self.spin_treble.value(),
            'center_line': self.chk_center_line.isChecked(),
            'analyzer_color_mode': self.combo_analyzer_color.currentText(),
            'line_base_width': self.spin_line_width.value(),
            'line_magnitude_sensitivity': self.spin_line_sensitivity.value(),
            'line_glow_intensity': self.spin_line_glow.value(),
            'line_bloom': self.spin_line_bloom.value(),
            'analyzer_bar_count': self.spin_bar_count.value(),
            'analyzer_position': self.combo_analyzer_pos.currentText(),
            'analyzer_show_center_line': self.chk_draw_center_line.isChecked(),
            'last_input_dir': self.last_input_dir,
            'default_load_folder': self.default_load_folder,
            'palette_folder': self._palette_folder,
        }
        self.config_manager.save({
            'output_batch': self.output_target_batch,
            'output_merge': self.output_target_merge,
            'settings': s
        })

    def _load_settings(self):
        config = self.config_manager.load()
        if config.get('output_batch'):
            self.output_target_batch = config['output_batch']
        if config.get('output_merge'):
            self.output_target_merge = config['output_merge']
        self._toggle_output_mode()

        s = config.get('settings', DEFAULT_SETTINGS)
        self.combo_res.setCurrentIndex(s.get('resolution', 1))
        self.spin_fps.setValue(s.get('fps', 30))
        self.chk_gpu.setChecked(s.get('gpu', True))
        self.chk_merge.setChecked(s.get('merge', False))
        self.txt_suffix.setText(s.get('suffix', ''))
        self._set_combo(self.combo_spec_type, s.get('spec_type', 'Standard (STFT)'))
        self._set_combo(self.combo_vis_mode, s.get('visual_mode', 'Standard'))
        self._set_combo(self.combo_cmap, s.get('colormap', 'Vibrant (Turbo)'))
        self._set_combo(self.combo_color_mode, s.get('color_mode', 'Static'))
        self._set_combo(self.combo_nfft, str(s.get('n_fft', 4096)))
        self._set_combo(self.combo_interp, s.get('interpolation', 'Lanczos'))
        self.spin_hop.setValue(s.get('hop_length', 512))
        self.chk_sharpen.setChecked(s.get('sharpen', True))
        self.chk_overlay.setChecked(s.get('overlay', True))
        self.spin_window.setValue(s.get('window', 5.0))
        self.chk_auto_freq.setChecked(s.get('auto_freq', True))
        self._toggle_auto_freq()
        self.spin_freq_min.setValue(s.get('min_freq', 20))
        self.spin_freq_max.setValue(s.get('max_freq', 20000))
        self.spin_dyn_range.setValue(s.get('dyn_range', 80))
        self.spin_gamma.setValue(s.get('gamma', 1.0))
        self.spin_blur.setValue(s.get('blur', 0.0))
        self.chk_normalize.setChecked(s.get('normalize', True))
        self.chk_auto_hp.setChecked(s.get('auto_highpass', True))
        self._toggle_auto_highpass()
        self.spin_highpass.setValue(s.get('highpass_freq', 20))
        self.spin_bass.setValue(s.get('bass', 0.0))
        self.spin_treble.setValue(s.get('treble', 0.0))
        self.chk_center_line.setChecked(s.get('center_line', True))
        self.chk_draw_center_line.setChecked(s.get('analyzer_show_center_line', True))
        self._set_combo(self.combo_analyzer_color, s.get('analyzer_color_mode', 'Match Video'))
        self._set_combo(self.combo_analyzer_pos, s.get('analyzer_position', 'Center'))
        self.spin_line_width.setValue(s.get('line_base_width', 6))
        self.spin_line_sensitivity.setValue(s.get('line_magnitude_sensitivity', 800.0))
        self.spin_line_glow.setValue(s.get('line_glow_intensity', 15.0))
        self.spin_line_bloom.setValue(s.get('line_bloom', 8.0))
        self.spin_bar_count.setValue(s.get('analyzer_bar_count', 64))

        saved_default = s.get('default_load_folder', '').strip()
        self.default_load_folder = (
            saved_default if saved_default and os.path.isdir(saved_default)
            else _get_default_load_folder())
        self.lbl_default_folder.setText(self.default_load_folder)

        saved_last = s.get('last_input_dir', '').strip()
        self.last_input_dir = (
            saved_last if saved_last and os.path.isdir(saved_last)
            else self.default_load_folder)

        if s.get('palette_folder', '').strip():
            self._palette_folder = s['palette_folder']
        self._update_palette_preview()

    @staticmethod
    def _set_combo(combo, text):
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _reset_settings(self):
        self.config_manager.save({'settings': DEFAULT_SETTINGS.copy()})
        self._load_settings()
        self._log("Settings reset to defaults.", "WARN")

    # ── Palette folder ──
    def _change_palette_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Palette Folder", self._palette_folder)
        if folder:
            self._palette_folder = folder
            self.available_luts = get_all_luts(folder)
            prev = self.combo_cmap.currentText()
            self.combo_cmap.blockSignals(True)
            self.combo_cmap.clear()
            self.combo_cmap.addItems(sorted(self.available_luts.keys()))
            idx = self.combo_cmap.findText(prev)
            self.combo_cmap.setCurrentIndex(idx if idx >= 0 else 0)
            self.combo_cmap.blockSignals(False)
            self._update_palette_preview()
            self._save_settings()
            self._log(f"Palette folder set to: {folder}", "SUCCESS")

    # ── Default load folder ──
    def _change_default_load_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Set Default Load Folder", self.default_load_folder)
        if folder:
            self.default_load_folder = folder
            self.last_input_dir = folder
            self.lbl_default_folder.setText(folder)
            self._save_settings()
            self.table_widget.setRowCount(0)
            self._log(f"Auto-loading from: {folder}", "SUCCESS")
            self._add_files_from_paths([folder])

    # ── File management ──
    def _load_all_from_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", self.last_input_dir)
        if folder:
            self._add_files_from_paths([folder])
            self.last_input_dir = folder
            self._save_settings()

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio", self.last_input_dir,
            "Audio (*.wav *.mp3 *.ogg *.flac *.aac *.m4a)")
        if files:
            self._add_files_from_paths(files)
            self.last_input_dir = os.path.dirname(files[0])
            self._save_settings()

    def _add_files_from_paths(self, paths):
        count = 0
        for path in paths:
            if os.path.isdir(path):
                walker = (os.walk(path) if self.chk_recursive.isChecked()
                          else [(path, [], os.listdir(path))])
                for root, _dirs, files in walker:
                    for f in files:
                        if f.lower().endswith(AUDIO_EXTS):
                            count += self._add_valid_item(os.path.join(root, f))
                    if not self.chk_recursive.isChecked():
                        break
            elif os.path.isfile(path) and path.lower().endswith(AUDIO_EXTS):
                count += self._add_valid_item(path)
        if count > 0:
            self._log(f"Added {count} file(s).")

    def _add_valid_item(self, filepath):
        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 0)
            if item and item.data(Qt.UserRole) == filepath:
                return 0
        try:
            info = sf.info(filepath)
            row = self.table_widget.rowCount()
            self.table_widget.insertRow(row)
            name_item = QTableWidgetItem(os.path.basename(filepath))
            name_item.setData(Qt.UserRole, filepath)
            name_item.setToolTip(filepath)
            self.table_widget.setItem(row, 0, name_item)
            dur_str = f"{int(info.duration // 60):02}:{int(info.duration % 60):02}"
            self.table_widget.setItem(row, 1, QTableWidgetItem(dur_str))
            self.table_widget.setItem(row, 2, QTableWidgetItem("Pending"))
            return 1
        except Exception as e:
            self._log(f"Cannot read {os.path.basename(filepath)}: {e}", "WARN")
            return 0

    def _update_table_status(self, filepath, text, color):
        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 0)
            if item and item.data(Qt.UserRole) == filepath:
                si = self.table_widget.item(row, 2)
                if si:
                    si.setText(text)
                    si.setForeground(QBrush(QColor(color)))
                break

    # ── Output ──
    def _select_output(self):
        if self.chk_merge.isChecked():
            f, _ = QFileDialog.getSaveFileName(
                self, "Save Video", self.output_target, "MP4 (*.mp4)")
            if f:
                if not f.lower().endswith('.mp4'):
                    f += '.mp4'
                self.output_target_merge = f
                self.output_target = f
                self.lbl_output.setText(f)
        else:
            d = QFileDialog.getExistingDirectory(
                self, "Select Folder", self.output_target)
            if d:
                self.output_target_batch = d
                self.output_target = d
                self.lbl_output.setText(d)
        self._save_settings()

    # ── Dialogs ──
    def _open_youtube_dialog(self):
        if not YT_DLP_AVAILABLE:
            QMessageBox.warning(self, "Missing Dependency",
                                "yt-dlp is not installed.\n\n"
                                "Run: pip install yt-dlp\n\n"
                                "Also ensure FFmpeg is in your PATH.")
            return
        dlg = DownloadDialog(self.default_load_folder, self)
        dlg.files_downloaded.connect(self._on_youtube_files_downloaded)
        dlg.exec()

    def _on_youtube_files_downloaded(self, files):
        if files:
            self._log(f"YouTube download: {len(files)} file(s) received.", "SUCCESS")
            self._add_files_from_paths(files)

    def _open_renamer_dialog(self):
        dlg = FileRenamerDialog(self.default_load_folder, self)
        dlg.exec()

    # ── Common settings ──
    def _get_common_settings(self, preview=False):
        spec_map = {'Standard (STFT)': 'stft', 'Mel-Scaled': 'mel', 'CQT': 'cqt'}
        vis_map = {'Standard': 'standard', 'Classic': 'classic', 'Radial': 'radial'}
        return {
            'merge_mode': self.chk_merge.isChecked(),
            'colormap': self.combo_cmap.currentText(),
            'luts': self.available_luts,
            'color_mode': self.combo_color_mode.currentText().lower().replace(' ', '_'),
            'visual_mode': vis_map.get(self.combo_vis_mode.currentText(), 'standard'),
            'spec_type': spec_map.get(self.combo_spec_type.currentText(), 'stft'),
            'use_gpu': self.chk_gpu.isChecked(),
            'width': [1280, 1920, 2560, 3840][self.combo_res.currentIndex()],
            'height': [720, 1080, 1440, 2160][self.combo_res.currentIndex()],
            'fps': self.spin_fps.value(),
            'preview_mode': preview,
            'draw_overlay': self.chk_overlay.isChecked(),
            'n_fft': int(self.combo_nfft.currentText()),
            'hop_length': self.spin_hop.value(),
            'sharpen': self.chk_sharpen.isChecked(),
            'interpolation': self.combo_interp.currentText().lower(),
            'window_size': self.spin_window.value(),
            'auto_freq': self.chk_auto_freq.isChecked(),
            'min_freq': self.spin_freq_min.value(),
            'max_freq': self.spin_freq_max.value(),
            'dynamic_range_db': self.spin_dyn_range.value(),
            'gamma': self.spin_gamma.value(),
            'blur': self.spin_blur.value(),
            'normalize': self.chk_normalize.isChecked(),
            'auto_highpass': self.chk_auto_hp.isChecked(),
            'highpass_freq': self.spin_highpass.value(),
            'bass_gain': self.spin_bass.value(),
            'treble_gain': self.spin_treble.value(),
            'filename_suffix': self.txt_suffix.text(),
            'center_line': self.chk_center_line.isChecked(),
            'analyzer_color_mode': self.combo_analyzer_color.currentText(),
            'line_base_width': self.spin_line_width.value(),
            'line_magnitude_sensitivity': self.spin_line_sensitivity.value(),
            'line_glow_intensity': self.spin_line_glow.value(),
            'line_bloom': self.spin_line_bloom.value(),
            'analyzer_bar_count': self.spin_bar_count.value(),
            'analyzer_position': self.combo_analyzer_pos.currentText(),
            'analyzer_show_center_line': self.chk_draw_center_line.isChecked(),
        }

    # ── Processing ──
    def _start_processing(self):
        files = self._get_files_to_process()
        if not files:
            self._log("No files to process.", "WARN")
            return
        settings = self._get_common_settings()
        self._save_settings()
        self._preview_path = None
        self._run_worker(files, self.output_target, settings)

    def _start_preview(self):
        files = self._get_files_to_process()
        if not files:
            self._log("No files to preview.", "WARN")
            return
        preview_out = os.path.join(tempfile.gettempdir(), "spectrogram_preview.mp4")
        self._preview_path = preview_out
        settings = self._get_common_settings(preview=True)
        self._run_worker(files, preview_out, settings)

    def _get_files_to_process(self):
        sel = []
        for i in self.table_widget.selectedItems():
            item = self.table_widget.item(i.row(), 0)
            if item:
                sel.append(item.data(Qt.UserRole))
        sel = list(set(sel))
        if sel:
            return sel
        all_files = []
        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 0)
            if item:
                all_files.append(item.data(Qt.UserRole))
        return all_files

    def _run_worker(self, files, output_target, settings):
        self.btn_start.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.error_count = 0
        self.progress_bar.setValue(0)

        self.worker = RenderWorker(files, output_target, settings)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(lambda s: self._log(s))
        self.worker.log.connect(lambda m: self._log(m))
        self.worker.file_started.connect(
            lambda f: self._update_table_status(f, "Rendering", "yellow"))
        self.worker.file_finished.connect(
            lambda f: self._update_table_status(f, "Done", "#2ECC71"))
        self.worker.batch_error.connect(self._on_batch_error)
        self.worker.error.connect(self._on_fatal_error)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_batch_error(self, filepath, msg):
        self._update_table_status(filepath, "Error", "red")
        self.error_count += 1
        self._log(f"Error: {os.path.basename(filepath)} — {msg}", "ERROR")

    def _on_fatal_error(self, msg):
        self._log(f"Fatal: {msg}", "ERROR")

    def _cancel_processing(self):
        if self.worker:
            self.worker.cancel()
            self._log("Cancelling...", "WARN")

    def _on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        if self.worker and self.worker.is_cancelled:
            self.progress_bar.setValue(0)
            self._log("Render cancelled.", "WARN")
        else:
            self.progress_bar.setValue(100)
            if self.error_count > 0:
                self._log(f"Batch complete with {self.error_count} error(s).", "WARN")
            else:
                self._log("Batch complete!", "SUCCESS")

            if self._preview_path and os.path.exists(self._preview_path):
                self._log(f"Preview saved: {self._preview_path}", "SUCCESS")
                open_file_location(self._preview_path)
                self._preview_path = None

        self.worker = None


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setOrganizationName("SpectrogramStudio")
    app.setApplicationName("Spectrogram Studio Pro")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())