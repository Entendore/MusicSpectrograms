import time
import numpy as np
import cv2
from typing import List, Dict, Callable
import librosa
import os

from video_encoder import VideoEncoder

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class VideoRenderer:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.luts = settings.get('luts', {})
        
        # --- SMART GPU INITIALIZATION ---
        self.device = torch.device("cpu")
        self.vram_capacity = 0 
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device("cuda")
            try:
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if total_vram >= 12: self.vram_capacity = 2 # High VRAM
                elif total_vram >= 6: self.vram_capacity = 1 # Med VRAM
                else: self.vram_capacity = 0 # Low VRAM
                print(f"[Renderer] GPU Acceleration Active: {torch.cuda.get_device_name(0)} ({total_vram:.1f} GB)")
            except Exception: 
                self.device = torch.device("cpu")

    def _get_optimal_batch_size(self, width, height):
        pixels_per_frame = width * height
        # Higher VRAM allows larger batches to minimize CPU/GPU sync overhead
        if self.vram_capacity == 2: target_frames = 120
        elif self.vram_capacity == 1: target_frames = 60
        else: target_frames = 20
        
        reference_pixels = 1920 * 1080
        scale_factor = reference_pixels / max(1, pixels_per_frame)
        return max(4, min(int(target_frames * scale_factor), 200))

    def _get_lut(self, name: str) -> np.ndarray:
        lut = self.luts.get(name)
        if lut is not None: return lut
        for fallback_name in ['Vibrant (Turbo)', 'Jet (Classic)']:
            lut = self.luts.get(fallback_name)
            if lut is not None: return lut
        if self.luts: return next(iter(self.luts.values()))
        gray = np.arange(256, dtype=np.uint8).reshape(256, 1, 1)
        return np.repeat(gray, 3, axis=2)

    def render_video(self, audio_path: str, output_path: str, spec_img: np.ndarray,
                            y_audio: np.ndarray, sr: int, duration: float,
                            segment_info: List[Dict],
                            progress_callback: Callable[[int, str], None],
                            cancel_check: Callable[[], bool]):
        
        fps = self.settings['fps']
        window_sec = self.settings['window_size']
        use_gpu_encoder = self.settings.get('use_gpu', False)
        draw_overlay = self.settings.get('draw_overlay', True)
        draw_center_line = self.settings.get('center_line', True)
        
        total_frames = int(duration * fps)
        if total_frames == 0: raise ValueError("Zero duration")

        img_h, img_w = spec_img.shape[:2]
        px_per_sec = img_w / duration
        target_w, target_h = self.settings['width'], self.settings['height']
        if target_w % 2 != 0: target_w -= 1
        if target_h % 2 != 0: target_h -= 1

        use_gpu_proc = self.device.type == 'cuda'
        batch_size = self._get_optimal_batch_size(target_w, target_h)
        if not use_gpu_proc: batch_size = 15

        lut_to_use = self._get_lut(self.settings.get('colormap', 'Vibrant (Turbo)'))
        # LUT is RGB format (256, 3)
        lut_rgb = cv2.cvtColor(lut_to_use, cv2.COLOR_BGR2RGB).reshape(256, 3)
        win_px = (window_sec / 2.0) * px_per_sec

        # --- GPU ASSET PRE-LOADING ---
        spec_tensor_gpu = None
        lut_tensor_gpu = None
        
        if use_gpu_proc:
            # Upload Spectrogram (H, W) -> (1, 1, H, W) for PyTorch
            spec_tensor_gpu = torch.from_numpy(spec_img).to(self.device).float().unsqueeze(0).unsqueeze(0)
            # Upload LUT
            lut_tensor_gpu = torch.from_numpy(lut_rgb).to(self.device).float() / 255.0
        
        # --- Encoder Setup ---
        encoder = VideoEncoder(target_w, target_h, fps, audio_path, output_path, use_gpu_encoder)
        encoder.start()
        start_time = time.time()
        
        try:
            for batch_start in range(0, total_frames, batch_size):
                if cancel_check(): 
                    return

                batch_end = min(batch_start + batch_size, total_frames)
                
                # --- RENDER BATCH ---
                if use_gpu_proc:
                    batch_frames = self._render_batch_gpu_core(
                        batch_start, batch_end, spec_tensor_gpu, lut_tensor_gpu, 
                        target_w, target_h, img_w, duration, px_per_sec, win_px
                    )
                    batch_frames = (batch_frames.cpu().numpy() * 255).astype(np.uint8)
                else:
                    batch_frames = self._render_batch_cpu(
                        batch_start, batch_end, spec_img, lut_rgb,
                        target_w, target_h, img_w, img_h, 
                        duration, px_per_sec, win_px
                    )

                # --- POST-PROCESSING (Overlays) ---
                for i, frame_idx in enumerate(range(batch_start, batch_end)):
                    frame = batch_frames[i]
                    
                    if draw_overlay:
                        self._draw_overlay(frame, frame_idx, fps, duration)
                    
                    if draw_center_line:
                        # Calculate current position in spectrogram pixels
                        time_sec = frame_idx / fps
                        center_x_spec = int(time_sec * px_per_sec)
                        
                        # Clamp to spectrogram width
                        cx = max(0, min(spec_img.shape[1] - 1, center_x_spec))
                        
                        # Extract the vertical column of data (the "Now" slice)
                        # This column represents frequencies (Y) at current time (X)
                        spec_column = spec_img[:, cx]
                        
                        self._draw_vertical_analyzer(frame, spec_column)

                # --- WRITE BATCH ---
                if not encoder.write(batch_frames.tobytes()):
                    raise Exception("FFmpeg process died unexpectedly.")

                # Progress
                if batch_start % (batch_size * 2) == 0:
                    elapsed = time.time() - start_time
                    speed = (batch_end) / elapsed if elapsed > 0 else 0
                    eta = int((total_frames - batch_end) / speed) if speed > 0 else 0
                    m, s = divmod(eta, 60)
                    enc = 'NVENC' if use_gpu_encoder else 'x264'
                    proc = 'GPU' if use_gpu_proc else 'CPU'
                    progress_callback(int((batch_end / total_frames) * 100), f"{proc}->{enc} — ETA: {m:02d}:{s:02d}")

        finally:
            err = encoder.finish()
            if cancel_check():
                raise Exception("Cancelled")
            if encoder.process.returncode != 0:
                raise Exception(f"FFmpeg Error:\n{err[-500:]}")

    def _render_batch_gpu_core(self, start_idx, end_idx, spec_t, lut_t, tw, th, sw, dur, pps, win_px):
        count = end_idx - start_idx
        indices = np.arange(start_idx, end_idx)
        times = indices / self.settings['fps']
        centers = (times * pps).astype(int)
        
        batch_list = []
        for i in range(count):
            cx = centers[i]
            x1 = int(max(0, cx - win_px))
            x2 = int(min(sw, cx + win_px))
            
            slice_t = spec_t[:, :, :, x1:x2] 
            resized = F.interpolate(slice_t, size=(th, tw), mode='bilinear', align_corners=False)
            batch_list.append(resized)

        batch_stack = torch.cat(batch_list, dim=0).squeeze(1)
        indices_map = batch_stack.long()
        colored_batch = lut_t[indices_map]
        
        return colored_batch

    def _render_batch_cpu(self, start_idx, end_idx, spec_img, lut_rgb, tw, th, sw, sh_src, dur, pps, win_px):
        count = end_idx - start_idx
        results = np.zeros((count, th, tw, 3), dtype=np.uint8)
        
        for i, frame_idx in enumerate(range(start_idx, end_idx)):
            time_sec = frame_idx / self.settings['fps']
            center_x = int(time_sec * pps)
            
            x1 = int(max(0, center_x - win_px))
            x2 = int(min(sw, center_x + win_px))
            
            slice_w = x2 - x1
            if slice_w <= 0:
                x1 = max(0, min(sw-1, center_x))
                x2 = x1 + 1

            img_slice = spec_img[:, x1:x2]
            colored = lut_rgb[img_slice].astype(np.uint8)
            
            try:
                frame = cv2.resize(colored, (tw, th), interpolation=cv2.INTER_LANCZOS4)
            except Exception:
                frame = np.zeros((th, tw, 3), dtype=np.uint8)
            
            results[i] = frame
            
        return results

    def _draw_overlay(self, frame, frame_idx, fps, duration):
        current_time = frame_idx / fps
        m, s = divmod(int(current_time), 60)
        ms = int((current_time - int(current_time)) * 100)
        text = f"{m:02}:{s:02}.{ms:02}"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_vertical_analyzer(self, frame, spec_column):
        """
        Draws a vertical frequency analyzer with borders and an optional center line.
        spec_column: A 1D numpy array (Height,) containing magnitude values 0-255.
        """
        h, w = frame.shape[:2]
        
        # Settings
        position = self.settings.get('analyzer_position', 'Center')
        bar_count = self.settings.get('analyzer_bar_count', 64)
        sensitivity = self.settings.get('line_magnitude_sensitivity', 800.0)
        color_mode = self.settings.get('analyzer_color_mode', 'Match Video')
        show_center_line = self.settings.get('analyzer_show_center_line', True)
        
        # Map the spectrogram column (height H_spec) to the number of bars
        # Using linear interpolation
        x_old = np.linspace(0, 1, len(spec_column))
        x_new = np.linspace(0, 1, bar_count)
        bar_magnitudes = np.interp(x_new, x_old, spec_column)
        
        bar_h = h / bar_count
        
        # Determine Color
        if color_mode == 'Match Video':
            # Sample color from center of frame
            base_color = frame[h//2, w//2].tolist()
        elif color_mode == 'White':
            base_color = (255, 255, 255)
        else:
            base_color = (0, 255, 255) # Cyan fallback

        # 1. Draw Center Line (if enabled)
        if show_center_line:
            cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)

        # 2. Draw Bars
        for i in range(bar_count):
            mag = bar_magnitudes[i]
            
            # Calculate horizontal length based on magnitude (0-255)
            bar_len = int((mag / 255.0) * sensitivity)
            
            # Y position (Top to Bottom)
            y1 = int(i * bar_h)
            y2 = int((i + 1) * bar_h) - 1
            
            # Determine color for this specific bar (Rainbow mode)
            if color_mode == 'Rainbow':
                hue = int((i / bar_count) * 180) % 180
                hsv = np.uint8([[[hue, 255, 255]]])
                color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
            else:
                color = base_color
            
            # Draw based on Position setting
            if position == 'Center':
                # Anchor is center of screen. Bars extend symmetrically.
                x_center = w // 2
                x1 = x_center - (bar_len // 2)
                x2 = x_center + (bar_len // 2)
                
                # Draw filled bar
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                # Draw border (white, thickness 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                
            elif position == 'Edges':
                # Anchor is Left (0) and Right (w). Bars extend inwards.
                
                # Draw Left Bar (Filled + Border)
                cv2.rectangle(frame, (0, y1), (bar_len, y2), color, -1)
                cv2.rectangle(frame, (0, y1), (bar_len, y2), (255, 255, 255), 1)
                
                # Draw Right Bar (Filled + Border)
                cv2.rectangle(frame, (w - bar_len, y1), (w, y2), color, -1)
                cv2.rectangle(frame, (w - bar_len, y1), (w, y2), (255, 255, 255), 1)