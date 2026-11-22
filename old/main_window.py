import os
import sys
import shutil
import tempfile
import time
import soundfile as sf
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QProgressBar, QGroupBox, QComboBox, QDoubleSpinBox, QCheckBox,
    QSpinBox, QLineEdit, QMenuBar, QTextEdit, QMessageBox, QTableWidgetItem
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QBrush, QTextCursor, QImage, QPixmap

from config import ConfigManager, AUDIO_EXTS, DEFAULT_SETTINGS, get_default_load_folder
from utils import open_file_location
from palettes import get_all_luts, get_default_palette_dir
from worker import RenderWorker  # UPDATED IMPORT
from widgets import AudioTableWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrogram Studio Pro v9.6")
        self.resize(1100, 980) 

        if not shutil.which("ffmpeg"):
            QMessageBox.critical(self, "Error", "FFmpeg not found! Please install FFmpeg and add it to your PATH.")
            sys.exit(1)

        self.worker = None
        self.config_manager = ConfigManager()

        saved_cfg = self.config_manager.load()
        saved_settings = saved_cfg.get('settings', {})
        self._palette_folder = saved_settings.get('palette_folder', '').strip()
        if not self._palette_folder:
            self._palette_folder = get_default_palette_dir()

        self.available_luts = get_all_luts(self._palette_folder)

        saved_load = saved_settings.get('default_load_folder', '').strip()
        self.default_load_folder = (
            saved_load if saved_load and os.path.isdir(saved_load)
            else get_default_load_folder())

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

        self.init_ui()
        self.init_menu()
        self.load_settings()
        self.setup_autosave()

        if self.default_load_folder and os.path.isdir(self.default_load_folder):
            self.add_files_from_paths([self.default_load_folder])

    def init_ui(self):
        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #e0e0e0; font-family: 'Segoe UI', Arial; }
            QGroupBox { border: 1px solid #3a3a3a; border-radius: 5px; margin-top: 20px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #3c3c3c; border: 1px solid #555; border-radius: 4px; padding: 8px; }
            QPushButton:hover { background-color: #4d4d4d; }
            QPushButton:disabled { background-color: #2a2a2a; color: #666; }
            QProgressBar::chunk { background-color: #2ecc71; border-radius: 3px; }
            QComboBox { padding: 4px; }
            QSpinBox, QDoubleSpinBox, QLineEdit { padding: 3px; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)

        # ---- 1. Source Files ----
        group_source = QGroupBox("Source Files")
        layout_source = QVBoxLayout(group_source)

        r_default = QHBoxLayout()
        r_default.addWidget(QLabel("Default Load Folder:"))
        self.lbl_default_folder = QLabel(self.default_load_folder)
        self.lbl_default_folder.setStyleSheet("color: #5dade2; font-weight: bold;")
        self.lbl_default_folder.setWordWrap(True)
        self.btn_change_default_folder = QPushButton("Change...")
        self.btn_change_default_folder.setFixedWidth(80)
        self.btn_change_default_folder.clicked.connect(self.change_default_load_folder)
        r_default.addWidget(self.lbl_default_folder, 1)
        r_default.addWidget(self.btn_change_default_folder)
        layout_source.addLayout(r_default)

        ctrl_row = QHBoxLayout()
        self.btn_add_files = QPushButton("Add Files")
        self.btn_add_files.clicked.connect(self.add_files)
        self.btn_load_folder = QPushButton("Load Folder")
        self.btn_load_folder.clicked.connect(self.load_all_from_folder)
        self.chk_recursive = QCheckBox("Recursive")
        self.chk_recursive.setChecked(True)
        ctrl_row.addWidget(self.btn_add_files)
        ctrl_row.addWidget(self.btn_load_folder)
        ctrl_row.addWidget(self.chk_recursive)
        ctrl_row.addStretch()
        layout_source.addLayout(ctrl_row)

        self.table_widget = AudioTableWidget()
        self.table_widget.files_dropped.connect(self.add_files_from_paths)
        layout_source.addWidget(self.table_widget)
        main_layout.addWidget(group_source)

        # ---- 2. Output & Video ----
        group_video = QGroupBox("Output Settings")
        layout_vid = QVBoxLayout(group_video)

        r_out = QHBoxLayout()
        self.btn_output = QPushButton("Save Location...")
        self.btn_output.clicked.connect(self.select_output)
        self.lbl_output = QLabel(self.output_target)
        self.lbl_output.setStyleSheet("font-weight: bold; color: #5dade2;")
        self.lbl_output.setWordWrap(True)
        r_out.addWidget(self.btn_output)
        r_out.addWidget(self.lbl_output, 1)
        layout_vid.addLayout(r_out)

        r_vid1 = QHBoxLayout()
        r_vid1.addWidget(QLabel("Resolution:"))
        self.combo_res = QComboBox()
        self.combo_res.addItems(['720p', '1080p', '1440p', '2160p'])
        self.combo_res.setCurrentIndex(1)
        r_vid1.addWidget(self.combo_res)
        r_vid1.addWidget(QLabel("FPS:"))
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(10, 60)
        self.spin_fps.setValue(30)
        r_vid1.addWidget(self.spin_fps)
        self.chk_gpu = QCheckBox("GPU (NVENC)")
        self.chk_gpu.setChecked(True)
        r_vid1.addWidget(self.chk_gpu)
        r_vid1.addStretch()
        layout_vid.addLayout(r_vid1)

        r_vid2 = QHBoxLayout()
        self.chk_merge = QCheckBox("Merge into One Video")
        self.chk_merge.stateChanged.connect(self.toggle_output_mode)
        r_vid2.addWidget(self.chk_merge)
        r_vid2.addWidget(QLabel("Filename Suffix:"))
        self.txt_suffix = QLineEdit()
        self.txt_suffix.setMaximumWidth(150)
        self.txt_suffix.setPlaceholderText("_spectrogram")
        r_vid2.addWidget(self.txt_suffix)
        r_vid2.addStretch()
        layout_vid.addLayout(r_vid2)
        main_layout.addWidget(group_video)

        # ---- 3. Spectrogram Engine ----
        group_spec = QGroupBox("Spectrogram Engine")
        layout_spec = QVBoxLayout(group_spec)

        r_algo = QHBoxLayout()
        r_algo.addWidget(QLabel("Algorithm:"))
        self.combo_spec_type = QComboBox()
        self.combo_spec_type.addItems(['Standard (STFT)', 'Mel-Scaled', 'CQT'])
        r_algo.addWidget(self.combo_spec_type)
        r_algo.addWidget(QLabel("Display:"))
        self.combo_vis_mode = QComboBox()
        self.combo_vis_mode.addItems(['Standard', 'Classic', 'Radial'])
        r_algo.addWidget(self.combo_vis_mode)
        r_algo.addStretch()
        layout_spec.addLayout(r_algo)

        r_color = QHBoxLayout()
        r_color.addWidget(QLabel("Palette:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(sorted(self.available_luts.keys()))
        self.combo_cmap.currentIndexChanged.connect(self.update_palette_preview)
        r_color.addWidget(self.combo_cmap)
        self.palette_preview = QLabel()
        self.palette_preview.setFixedHeight(22)
        self.palette_preview.setMinimumWidth(180)
        self.palette_preview.setStyleSheet("border: 1px solid #555; border-radius: 3px;")
        r_color.addWidget(self.palette_preview, 1)
        r_color.addWidget(QLabel("Color Cycle:"))
        self.combo_color_mode = QComboBox()
        self.combo_color_mode.addItems(['Static', 'Rainbow Cycle'])
        r_color.addWidget(self.combo_color_mode)
        r_color.addStretch()
        layout_spec.addLayout(r_color)

        r_gran = QHBoxLayout()
        r_gran.addWidget(QLabel("FFT:"))
        self.combo_nfft = QComboBox()
        self.combo_nfft.addItems(['2048', '4096', '8192', '16384'])
        self.combo_nfft.setCurrentIndex(1)
        r_gran.addWidget(self.combo_nfft)
        r_gran.addWidget(QLabel("Hop:"))
        self.spin_hop = QSpinBox()
        self.spin_hop.setRange(64, 4096)
        self.spin_hop.setValue(512)
        r_gran.addWidget(self.spin_hop)
        r_gran.addWidget(QLabel("Interp:"))
        self.combo_interp = QComboBox()
        self.combo_interp.addItems(['Linear', 'Cubic', 'Lanczos'])
        self.combo_interp.setCurrentIndex(2)
        r_gran.addWidget(self.combo_interp)
        self.chk_sharpen = QCheckBox("Sharpen")
        self.chk_sharpen.setChecked(True)
        r_gran.addWidget(self.chk_sharpen)
        r_gran.addStretch()
        layout_spec.addLayout(r_gran)

        r_freq = QHBoxLayout()
        self.chk_auto_freq = QCheckBox("Auto Hz")
        self.chk_auto_freq.setChecked(True)
        self.chk_auto_freq.stateChanged.connect(self.toggle_auto_freq)
        r_freq.addWidget(self.chk_auto_freq)
        r_freq.addWidget(QLabel("Min:"))
        self.spin_freq_min = QSpinBox()
        self.spin_freq_min.setRange(1, 20000)
        self.spin_freq_min.setValue(20)
        r_freq.addWidget(self.spin_freq_min)
        r_freq.addWidget(QLabel("Max:"))
        self.spin_freq_max = QSpinBox()
        self.spin_freq_max.setRange(20, 22000)
        self.spin_freq_max.setValue(20000)
        r_freq.addWidget(self.spin_freq_max)
        r_freq.addStretch()
        layout_spec.addLayout(r_freq)

        r_fx = QHBoxLayout()
        self.chk_overlay = QCheckBox("Time Overlay")
        self.chk_overlay.setChecked(True)
        r_fx.addWidget(self.chk_overlay)
        r_fx.addWidget(QLabel("Window:"))
        self.spin_window = QDoubleSpinBox()
        self.spin_window.setRange(1, 60)
        self.spin_window.setValue(5.0)
        r_fx.addWidget(self.spin_window)
        r_fx.addWidget(QLabel("Dyn:"))
        self.spin_dyn_range = QSpinBox()
        self.spin_dyn_range.setRange(20, 150)
        self.spin_dyn_range.setValue(80)
        r_fx.addWidget(self.spin_dyn_range)
        r_fx.addWidget(QLabel("Gamma:"))
        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.1, 5.0)
        self.spin_gamma.setValue(1.0)
        self.spin_gamma.setSingleStep(0.1)
        r_fx.addWidget(self.spin_gamma)
        r_fx.addStretch()
        layout_spec.addLayout(r_fx)

        main_layout.addWidget(group_spec)

        # ---- 4. Audio Processing ----
        group_audio = QGroupBox("Audio Processing")
        r_audio = QHBoxLayout(group_audio)
        
        self.chk_normalize = QCheckBox("Normalize")
        self.chk_normalize.setChecked(True)
        r_audio.addWidget(self.chk_normalize)
        
        self.chk_auto_hp = QCheckBox("Auto Clean Sub-Bass")
        self.chk_auto_hp.setChecked(True)
        self.chk_auto_hp.setToolTip("Automatically detects and removes inaudible sub-bass rumble/noise.")
        self.chk_auto_hp.stateChanged.connect(self.toggle_auto_highpass)
        r_audio.addWidget(self.chk_auto_hp)
        
        r_audio.addWidget(QLabel("High-Pass:"))
        self.spin_highpass = QSpinBox()
        self.spin_highpass.setRange(0, 200)
        self.spin_highpass.setValue(20)
        self.spin_highpass.setSuffix(" Hz")
        r_audio.addWidget(self.spin_highpass)
        
        r_audio.addWidget(QLabel("Bass:"))
        self.spin_bass = QDoubleSpinBox()
        self.spin_bass.setRange(-12, 12)
        self.spin_bass.setSuffix(" dB")
        r_audio.addWidget(self.spin_bass)
        
        r_audio.addWidget(QLabel("Treble:"))
        self.spin_treble = QDoubleSpinBox()
        self.spin_treble.setRange(-12, 12)
        self.spin_treble.setSuffix(" dB")
        r_audio.addWidget(self.spin_treble)
        
        r_audio.addStretch()
        main_layout.addWidget(group_audio)

        # ---- 5. Center Visualizer (Spectrum Analyzer) ----
        group_visualizer = QGroupBox("Center Visualizer (Spectrum Analyzer)")
        layout_vis = QVBoxLayout(group_visualizer)

        r_vis_en = QHBoxLayout()
        self.chk_center_line = QCheckBox("Enable Visualizer")
        self.chk_center_line.setChecked(True)
        r_vis_en.addWidget(self.chk_center_line)
        
        self.chk_draw_center_line = QCheckBox("Draw Center Line")
        self.chk_draw_center_line.setChecked(True)
        r_vis_en.addWidget(self.chk_draw_center_line)
        
        r_vis_en.addWidget(QLabel("Position:"))
        self.combo_analyzer_pos = QComboBox()
        self.combo_analyzer_pos.addItems(['Center', 'Edges'])
        r_vis_en.addWidget(self.combo_analyzer_pos)
        
        r_vis_en.addWidget(QLabel("Color Style:"))
        self.combo_analyzer_color = QComboBox()
        self.combo_analyzer_color.addItems(['Match Video', 'Neon Gradient', 'Sunset Gradient', 'White', 'Rainbow'])
        r_vis_en.addWidget(self.combo_analyzer_color)
        r_vis_en.addStretch()
        layout_vis.addLayout(r_vis_en)

        r_vis_param = QHBoxLayout()
        
        r_vis_param.addWidget(QLabel("Bars:"))
        self.spin_bar_count = QSpinBox()
        self.spin_bar_count.setRange(2, 256)
        self.spin_bar_count.setValue(64)
        r_vis_param.addWidget(self.spin_bar_count)
        
        r_vis_param.addWidget(QLabel("Width:"))
        self.spin_line_width = QSpinBox()
        self.spin_line_width.setRange(1, 200)
        self.spin_line_width.setValue(6)
        r_vis_param.addWidget(self.spin_line_width)
        r_vis_param.addWidget(QLabel("Sensitivity:"))
        self.spin_line_sensitivity = QDoubleSpinBox()
        self.spin_line_sensitivity.setRange(0, 2000)
        self.spin_line_sensitivity.setValue(800)
        self.spin_line_sensitivity.setSingleStep(50)
        r_vis_param.addWidget(self.spin_line_sensitivity)
        r_vis_param.addWidget(QLabel("Glow:"))
        self.spin_line_glow = QDoubleSpinBox()
        self.spin_line_glow.setRange(0, 50)
        self.spin_line_glow.setValue(15)
        self.spin_line_glow.setSingleStep(1)
        r_vis_param.addWidget(self.spin_line_glow)
        r_vis_param.addWidget(QLabel("Bloom:"))
        self.spin_line_bloom = QDoubleSpinBox()
        self.spin_line_bloom.setRange(0, 50)
        self.spin_line_bloom.setValue(8)
        self.spin_line_bloom.setSingleStep(1)
        r_vis_param.addWidget(self.spin_line_bloom)
        r_vis_param.addStretch()
        layout_vis.addLayout(r_vis_param)

        main_layout.addWidget(group_visualizer)

        # ---- Controls ----
        render_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Render")
        self.btn_start.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 15px;")
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_preview = QPushButton("Preview (15s)")
        self.btn_preview.setStyleSheet("background-color: #f1c40f; color: black; font-weight: bold; padding: 15px;")
        self.btn_preview.clicked.connect(self.start_preview)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet("background-color: #e74c3c; color: white; padding: 15px;")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_processing)
        render_layout.addWidget(self.btn_start)
        render_layout.addWidget(self.btn_preview)
        render_layout.addWidget(self.btn_cancel)
        main_layout.addLayout(render_layout)

        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(100)
        main_layout.addWidget(self.log_widget)

        self.update_palette_preview()
        self.toggle_auto_freq()
        self.toggle_auto_highpass()

    def init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Reset Settings", self.reset_settings)
        file_menu.addAction("Set Palette Folder...", self.change_palette_folder)
        file_menu.addAction("Open Palette Folder", lambda: open_file_location(self._palette_folder))
        file_menu.addSeparator()
        file_menu.addAction("Open Output Folder", lambda: open_file_location(self.output_target))
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

    def toggle_auto_freq(self):
        is_auto = self.chk_auto_freq.isChecked()
        self.spin_freq_min.setEnabled(not is_auto)
        self.spin_freq_max.setEnabled(not is_auto)

    def toggle_auto_highpass(self):
        is_auto = self.chk_auto_hp.isChecked()
        self.spin_highpass.setEnabled(not is_auto)

    def update_palette_preview(self):
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
            self.palette_preview.setPixmap(QPixmap.fromImage(qimg).scaled(max(1, self.palette_preview.width()), 20, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        except Exception:
            self.palette_preview.setText("(error)")

    def change_palette_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Palette Folder", self._palette_folder)
        if folder:
            self._palette_folder = folder
            self.available_luts = get_all_luts(folder)
            prev_text = self.combo_cmap.currentText()
            self.combo_cmap.blockSignals(True)
            self.combo_cmap.clear()
            self.combo_cmap.addItems(sorted(self.available_luts.keys()))
            idx = self.combo_cmap.findText(prev_text)
            self.combo_cmap.setCurrentIndex(idx if idx >= 0 else 0)
            self.combo_cmap.blockSignals(False)
            self.update_palette_preview()
            self.save_settings()
            self.log_message(f"Palette folder set to: {folder}", "SUCCESS")

    def change_default_load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Set Default Load Folder", self.default_load_folder)
        if folder:
            self.default_load_folder = folder
            self.last_input_dir = folder
            self.lbl_default_folder.setText(folder)
            self.save_settings()
            self.table_widget.setRowCount(0)
            self.log_message(f"Auto-loading from: {folder}", "SUCCESS")
            self.add_files_from_paths([folder])

    def setup_autosave(self):
        widgets = [
            self.combo_res, self.spin_fps, self.chk_gpu, self.chk_merge, self.txt_suffix,
            self.combo_spec_type, self.combo_vis_mode, self.combo_cmap, self.combo_color_mode,
            self.combo_nfft, self.spin_hop, self.combo_interp, self.chk_sharpen, self.chk_overlay,
            self.spin_window, self.chk_auto_freq, self.spin_freq_min, self.spin_freq_max,
            self.spin_dyn_range, self.spin_gamma, self.chk_normalize, self.chk_auto_hp, self.spin_highpass, self.spin_bass, self.spin_treble,
            self.chk_center_line, self.combo_analyzer_color, self.spin_line_width, 
            self.spin_line_sensitivity, self.spin_line_glow, self.spin_line_bloom,
            self.spin_bar_count, self.combo_analyzer_pos, self.chk_draw_center_line
        ]
        for w in widgets:
            try:
                if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                    w.valueChanged.connect(self.save_settings)
                elif isinstance(w, QComboBox):
                    w.currentIndexChanged.connect(self.save_settings)
                elif isinstance(w, QCheckBox):
                    w.stateChanged.connect(self.save_settings)
                elif isinstance(w, QLineEdit):
                    w.textChanged.connect(self.save_settings)
            except Exception:
                pass

    def log_message(self, text, level="INFO"):
        timestamp = time.strftime("%H:%M:%S")
        color_map = {"INFO": "white", "WARN": "#F39C12", "ERROR": "#E74C3C", "SUCCESS": "#2ECC71"}
        self.log_widget.append(f'<span style="color:{color_map.get(level, "white")};">[{timestamp}] {text}</span>')
        self.log_widget.moveCursor(QTextCursor.End)

    def save_settings(self):
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
        config = {'output_batch': self.output_target_batch, 'output_merge': self.output_target_merge, 'settings': s}
        self.config_manager.save(config)

    def load_settings(self):
        config = self.config_manager.load()
        if config.get('output_batch'): self.output_target_batch = config['output_batch']
        if config.get('output_merge'): self.output_target_merge = config['output_merge']
        self.toggle_output_mode()

        s = config.get('settings', DEFAULT_SETTINGS)
        
        self.combo_res.setCurrentIndex(s.get('resolution', 1))
        self.spin_fps.setValue(s.get('fps', 30))
        self.chk_gpu.setChecked(s.get('gpu', True))
        self.chk_merge.setChecked(s.get('merge', False))
        self.txt_suffix.setText(s.get('suffix', ''))
        
        self._set_combo_text(self.combo_spec_type, s.get('spec_type', 'Standard (STFT)'))
        self._set_combo_text(self.combo_vis_mode, s.get('visual_mode', 'Standard'))
        self._set_combo_text(self.combo_cmap, s.get('colormap', 'Vibrant (Turbo)'))
        self._set_combo_text(self.combo_color_mode, s.get('color_mode', 'Static'))
        self._set_combo_text(self.combo_nfft, str(s.get('n_fft', 4096)))
        self._set_combo_text(self.combo_interp, s.get('interpolation', 'Lanczos'))
        
        self.spin_hop.setValue(s.get('hop_length', 512))
        self.chk_sharpen.setChecked(s.get('sharpen', True))
        self.chk_overlay.setChecked(s.get('overlay', True))
        self.spin_window.setValue(s.get('window', 5.0))
        self.chk_auto_freq.setChecked(s.get('auto_freq', True))
        self.toggle_auto_freq()
        self.spin_freq_min.setValue(s.get('min_freq', 20))
        self.spin_freq_max.setValue(s.get('max_freq', 20000))
        self.spin_dyn_range.setValue(s.get('dyn_range', 80))
        self.spin_gamma.setValue(s.get('gamma', 1.0))
        
        self.chk_normalize.setChecked(s.get('normalize', True))
        self.chk_auto_hp.setChecked(s.get('auto_highpass', True))
        self.toggle_auto_highpass()
        self.spin_highpass.setValue(s.get('highpass_freq', 20))
        self.spin_bass.setValue(s.get('bass', 0.0))
        self.spin_treble.setValue(s.get('treble', 0.0))
        
        self.chk_center_line.setChecked(s.get('center_line', True))
        self.chk_draw_center_line.setChecked(s.get('analyzer_show_center_line', True))
        self._set_combo_text(self.combo_analyzer_color, s.get('analyzer_color_mode', 'Match Video'))
        self._set_combo_text(self.combo_analyzer_pos, s.get('analyzer_position', 'Center'))
        self.spin_line_width.setValue(s.get('line_base_width', 6))
        self.spin_line_sensitivity.setValue(s.get('line_magnitude_sensitivity', 800.0))
        self.spin_line_glow.setValue(s.get('line_glow_intensity', 15.0))
        self.spin_line_bloom.setValue(s.get('line_bloom', 8.0))
        self.spin_bar_count.setValue(s.get('analyzer_bar_count', 64))

        saved_default = s.get('default_load_folder', '').strip()
        self.default_load_folder = saved_default if saved_default and os.path.isdir(saved_default) else get_default_load_folder()
        self.lbl_default_folder.setText(self.default_load_folder)

        saved_last = s.get('last_input_dir', '').strip()
        self.last_input_dir = saved_last if saved_last and os.path.isdir(saved_last) else self.default_load_folder

        if s.get('palette_folder', '').strip(): self._palette_folder = s['palette_folder']
        self.update_palette_preview()

    @staticmethod
    def _set_combo_text(combo, text):
        idx = combo.findText(text)
        if idx >= 0: combo.setCurrentIndex(idx)

    def reset_settings(self):
        self.config_manager.save({'settings': DEFAULT_SETTINGS})
        self.load_settings()
        self.log_message("Settings reset to defaults.", "WARN")

    def load_all_from_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", self.last_input_dir)
        if folder:
            self.add_files_from_paths([folder])
            self.last_input_dir = folder
            self.save_settings()

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Audio", self.last_input_dir, "Audio (*.wav *.mp3 *.ogg *.flac *.aac *.m4a)")
        if files:
            self.add_files_from_paths(files)
            self.last_input_dir = os.path.dirname(files[0])
            self.save_settings()

    def add_files_from_paths(self, paths):
        count = 0
        for path in paths:
            if os.path.isdir(path):
                walker = (os.walk(path) if self.chk_recursive.isChecked() else [(path, [], os.listdir(path))])
                for root, _dirs, files in walker:
                    for f in files:
                        if f.lower().endswith(AUDIO_EXTS):
                            count += self.add_valid_item(os.path.join(root, f))
                    if not self.chk_recursive.isChecked(): break
            elif os.path.isfile(path) and path.lower().endswith(AUDIO_EXTS):
                count += self.add_valid_item(path)
        if count > 0: self.log_message(f"Added {count} file(s).")

    def add_valid_item(self, filepath):
        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 0)
            if item and item.data(Qt.UserRole) == filepath: return 0
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
            self.log_message(f"Cannot read {os.path.basename(filepath)}: {e}", "WARN")
            return 0

    def update_table_status(self, filepath, text, color):
        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 0)
            if item and item.data(Qt.UserRole) == filepath:
                status_item = self.table_widget.item(row, 2)
                if status_item:
                    status_item.setText(text)
                    status_item.setForeground(QBrush(QColor(color)))
                break

    def toggle_output_mode(self):
        is_merge = self.chk_merge.isChecked()
        self.output_target = self.output_target_merge if is_merge else self.output_target_batch
        self.lbl_output.setText(self.output_target)

    def select_output(self):
        if self.chk_merge.isChecked():
            f, _ = QFileDialog.getSaveFileName(self, "Save Video", self.output_target, "MP4 (*.mp4)")
            if f:
                if not f.lower().endswith('.mp4'): f += '.mp4'
                self.output_target_merge = f
                self.output_target = f
                self.lbl_output.setText(f)
        else:
            d = QFileDialog.getExistingDirectory(self, "Select Folder", self.output_target)
            if d:
                self.output_target_batch = d
                self.output_target = d
                self.lbl_output.setText(d)
        self.save_settings()

    def get_common_settings(self, preview=False):
        spec_type_map = {'Standard (STFT)': 'stft', 'Mel-Scaled': 'mel', 'CQT': 'cqt'}
        vis_mode_map = {'Standard': 'standard', 'Classic': 'classic', 'Radial': 'radial'}
        return {
            'merge_mode': self.chk_merge.isChecked(),
            'colormap': self.combo_cmap.currentText(),
            'luts': self.available_luts,
            'color_mode': self.combo_color_mode.currentText().lower().replace(' ', '_'),
            'visual_mode': vis_mode_map.get(self.combo_vis_mode.currentText(), 'standard'),
            'spec_type': spec_type_map.get(self.combo_spec_type.currentText(), 'stft'),
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

    def start_processing(self):
        files = self.get_files_to_process()
        if not files:
            self.log_message("No files to process.", "WARN")
            return
        settings = self.get_common_settings()
        self.save_settings()
        self._preview_path = None
        self.run_worker(files, self.output_target, settings)

    def start_preview(self):
        files = self.get_files_to_process()
        if not files:
            self.log_message("No files to preview.", "WARN")
            return
        preview_output = os.path.join(tempfile.gettempdir(), "spectrogram_preview.mp4")
        self._preview_path = preview_output
        settings = self.get_common_settings(preview=True)
        self.run_worker(files, preview_output, settings)

    def get_files_to_process(self):
        sel = []
        for i in self.table_widget.selectedItems():
            item = self.table_widget.item(i.row(), 0)
            if item: sel.append(item.data(Qt.UserRole))
        sel = list(set(sel))
        if sel: return sel
        all_files = []
        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 0)
            if item: all_files.append(item.data(Qt.UserRole))
        return all_files

    def run_worker(self, files, output_target, settings):
        self.btn_start.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.error_count = 0
        self.progress_bar.setValue(0)

        self.worker = RenderWorker(files, output_target, settings)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(lambda m: self.log_message(m))
        self.worker.file_started.connect(lambda f: self.update_table_status(f, "Rendering", "yellow"))
        self.worker.file_finished.connect(lambda f: self.update_table_status(f, "Done", "#2ECC71"))
        self.worker.batch_error.connect(self._on_batch_error)
        self.worker.error.connect(self._on_fatal_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def _on_batch_error(self, filepath, msg):
        self.update_table_status(filepath, "Error", "red")
        self.error_count += 1
        self.log_message(f"Error: {os.path.basename(filepath)} — {msg}", "ERROR")

    def _on_fatal_error(self, msg):
        self.log_message(f"Fatal: {msg}", "ERROR")

    def cancel_processing(self):
        if self.worker:
            self.worker.cancel()
            self.log_message("Cancelling...", "WARN")

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.btn_cancel.setEnabled(False)

        if self.worker and self.worker.is_cancelled:
            self.progress_bar.setValue(0)
            self.log_message("Render cancelled.", "WARN")
        else:
            self.progress_bar.setValue(100)
            if self.error_count > 0:
                self.log_message(f"Batch complete with {self.error_count} error(s).", "WARN")
            else:
                self.log_message("Batch complete!", "SUCCESS")

            if self._preview_path and os.path.exists(self._preview_path):
                self.log_message(f"Preview saved: {self._preview_path}", "SUCCESS")
                open_file_location(self._preview_path)
                self._preview_path = None

        self.worker = None