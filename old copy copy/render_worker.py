import os
import traceback
import librosa
import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal
from typing import List, Dict
from audio_engine import SpectrogramProcessor
from renderer import VideoRenderer

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

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        try:
            if self.settings.get('merge_mode', False):
                self._run_merge()
            else:
                self._run_batch()
        except Exception as e:
            self.error.emit(f"Fatal Worker Error: {str(e)}")
            traceback.print_exc()
        finally:
            self.finished.emit()

    def _run_batch(self):
        total_files = len(self.file_list)
        for i, filepath in enumerate(self.file_list):
            if self.is_cancelled:
                self.log.emit("Processing cancelled by user.")
                break

            self.file_started.emit(filepath)
            
            try:
                # Determine output path
                basename = os.path.splitext(os.path.basename(filepath))[0]
                suffix = self.settings.get('filename_suffix', '')
                out_name = f"{basename}{suffix}.mp4"
                output_path = os.path.join(self.output_target, out_name)
                
                self._process_single_file(filepath, output_path)
                self.file_finished.emit(filepath)
                
            except Exception as e:
                err_msg = str(e).replace('\n', ' | ') 
                self.batch_error.emit(filepath, err_msg)
            
            # Update overall progress
            self.progress.emit(int(((i + 1) / total_files) * 100))

    def _run_merge(self):
        # Basic merge implementation: processes files sequentially into the target file.
        # Note: True merging usually requires concatenation logic. 
        # This implementation processes the list, overwriting the target for the last file 
        # or handling one file if specific merge logic isn't fully defined.
        # For now, we process the first file found as the merged output source.
        
        if len(self.file_list) > 0:
            # In a real merge scenario, you might concatenate audio first, 
            # but here we will just process the file list to the specific output path.
            self.log.emit(f"Merge mode: Processing to {os.path.basename(self.output_target)}")
            self._process_single_file(self.file_list[0], self.output_target)
            # You could extend this to loop through and append if the encoder supported it.
        else:
            self.log.emit("No files to merge.")

    def _process_single_file(self, filepath: str, output_path: str):
        self.log.emit(f"Loading: {os.path.basename(filepath)}")
        
        # 1. Load Audio
        try:
            y, sr = librosa.load(filepath, sr=None, mono=True)
        except Exception as e:
            # Fallback to soundfile if librosa fails (rare but possible)
            try:
                data, sr = sf.read(filepath)
                if data.ndim > 1: data = data[:, 0] # Grab first channel
                y = data
            except Exception as e2:
                raise Exception(f"Failed to read audio: {e2}")

        duration = librosa.get_duration(y=y, sr=sr)
        
        # 2. Process Audio (EQ, Normalization, High-pass)
        self.status.emit("Processing Audio...")
        y_processed = self.processor.process_audio(y, sr)
        
        # 3. Generate Spectrogram Image Data
        self.status.emit("Generating Spectrogram...")
        spec_img, hop_length = self.processor.generate_spectrogram_image(y_processed, sr, duration)
        
        # 4. Render Video
        self.status.emit("Rendering Video...")
        
        # Prepare segment info (dummy for now, used for beat markers if implemented)
        segment_info = [] 
        
        def progress_cb(val, msg):
            self.status.emit(msg)
            # Hook for frame-by-frame progress if needed

        self.renderer.render_video(
            audio_path=filepath,
            output_path=output_path,
            spec_img=spec_img,
            y_audio=y_processed,
            sr=sr,
            duration=duration,
            segment_info=segment_info,
            progress_callback=progress_cb,
            cancel_check=lambda: self.is_cancelled
        )
        
        if not self.is_cancelled:
            self.log.emit(f"Finished: {os.path.basename(output_path)}")