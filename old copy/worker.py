# worker.py
import os
import tempfile
import subprocess
import traceback
import soundfile as sf
import numpy as np
from PySide6.QtCore import QThread, Signal
from typing import List, Dict

# Import the consolidated engine
from renderer import SpectrogramProcessor, VideoRenderer

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

        # Determine temp directory
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
                self.process_concatenated()
            else:
                self.process_batch()
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
        """Build output path with optional suffix and collision avoidance."""
        suffix = self.settings.get('filename_suffix', '').strip()
        name = base_name
        if suffix:
            name = f"{name}{suffix}"
        out = os.path.join(out_dir, f"{name}.mp4")
        # Avoid overwriting existing files
        if os.path.exists(out):
            out = os.path.join(out_dir, f"{name}_{index}.mp4")
        # Further collision avoidance
        counter = 1
        final_out = out
        while os.path.exists(final_out):
            stem, ext = os.path.splitext(out)
            final_out = f"{stem}_v{counter}{ext}"
            counter += 1
        return final_out

    def _cleanup_temp(self, *paths):
        """Safely remove temporary files."""
        for tmp in paths:
            try:
                if tmp and os.path.exists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass

    def process_concatenated(self):
        self.log.emit("Merging files...")
        valid_files, total_dur, seg_info = [], 0.0, []
        for fpath in self.file_list:
            if self.is_cancelled:
                return
            try:
                info = sf.info(fpath)
                if info.duration > 0:
                    seg_info.append({
                        'start': total_dur,
                        'end': total_dur + info.duration,
                        'name': os.path.basename(fpath)
                    })
                    valid_files.append(fpath)
                    total_dur += info.duration
            except Exception:
                self.batch_error.emit(fpath, "Read Error")

        if not valid_files:
            self.log.emit("No valid files to merge.")
            return

        temp_base = self._get_temp_dir()
        list_path = os.path.join(temp_base, "concat.txt")
        temp_wav = os.path.join(temp_base, "merged.wav")
        temp_proc_wav = os.path.join(temp_base, "merged_processed.wav")

        try:
            # Write concat list with forward slashes for FFmpeg compatibility
            with open(list_path, 'w', encoding='utf-8') as wf:
                for p in valid_files:
                    # Escape single quotes in paths for FFmpeg concat format
                    escaped = p.replace('\\', '/').replace("'", "'\\''")
                    wf.write(f"file '{escaped}'\n")

            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_path,
                '-ac', '1', '-ar', '44100', '-c:a', 'pcm_f32le', temp_wav
            ]
            result = subprocess.run(cmd, stderr=subprocess.PIPE)
            if result.returncode != 0:
                err_msg = result.stderr.decode('utf-8', errors='replace')[-500:]
                raise Exception(f"FFmpeg concat failed: {err_msg}")

            y, sr = sf.read(temp_wav, dtype='float32')
            dur = len(y) / sr
            if dur == 0:
                raise Exception("Merged audio is empty")

            if self.settings.get('preview_mode', False):
                y = y[:int(min(15, dur) * sr)]
                dur = len(y) / sr

            # Process audio (normalize, EQ)
            y_proc = self.processor.process_audio(y, sr)

            # Write processed audio so the video gets the correct audio track
            sf.write(temp_proc_wav, y_proc, sr)

            spec, _ = self.processor.generate_spectrogram_image(y_proc, sr, dur)

            out = (self.output_target if os.path.isfile(self.output_target)
                   else os.path.join(self.output_target, "Merged.mp4"))

            self.log.emit(f"Rendering merged video ({dur:.1f}s)...")

            self.renderer.render_video_ffmpeg(
                temp_proc_wav, out, spec, y_proc, sr, dur, seg_info,
                lambda p, s: (self.progress.emit(p), self.status.emit(s)),
                lambda: self.is_cancelled
            )

            if not self.is_cancelled:
                self.log.emit(f"Merged output saved: {out}")

        finally:
            self._cleanup_temp(list_path, temp_wav, temp_proc_wav)

    def process_batch(self):
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

                # Process audio (normalize, EQ)
                y_proc = self.processor.process_audio(y, sr)

                if self.settings.get('preview_mode', False):
                    preview_dur = min(15, dur)
                    y_proc = y_proc[:int(preview_dur * sr)]
                    dur = len(y_proc) / sr

                # Write processed audio to temp WAV for FFmpeg
                temp_proc_wav = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False,
                    dir=self._get_temp_dir()).name
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

            # Update overall batch progress
            if not self.is_cancelled and total > 0:
                batch_pct = int(((i + 1) / total) * 100)
                self.progress.emit(batch_pct)

        # Clean up temp dir if empty
        try:
            if self.local_temp_dir and os.path.isdir(self.local_temp_dir):
                remaining = os.listdir(self.local_temp_dir)
                if not remaining:
                    os.rmdir(self.local_temp_dir)
        except OSError:
            pass