import numpy as np
import librosa
import cv2
from typing import Dict, Tuple, Optional
import hashlib

try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None

class SpectrogramProcessor:
    def __init__(self, settings: Dict):
        self.settings = settings
        self._filter_cache = {}
        self._remap_cache = {}
        self._gamma_lut = None
        self._gamma_cache_key = None

    def process_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Apply normalization, EQ, and Smart High-Pass filters."""
        if len(y) == 0:
            return y
        
        # --- 1. SMART NOISE DETECTION (New Improvement) ---
        hp_freq = 0
        if self.settings.get('auto_highpass', True):
            hp_freq = self._detect_sub_bass_noise(y, sr)
        else:
            hp_freq = self.settings.get('highpass_freq', 20)
            
        # Apply High-Pass Filter
        if scipy_signal and hp_freq > 0 and hp_freq < sr / 2:
            try:
                b, a = scipy_signal.butter(5, hp_freq / (sr / 2), btype='high', analog=False)
                y = scipy_signal.filtfilt(b, a, y)
            except Exception:
                pass

        # 2. Normalization
        if self.settings.get('normalize', True):
            peak = np.max(np.abs(y))
            if peak > 1e-5:
                y = librosa.util.normalize(y)

        # 3. EQ Shelf Filters
        if scipy_signal:
            if self.settings.get('bass_gain', 0) != 0:
                b, a = self._get_filter_coefficients(sr, 200, self.settings['bass_gain'], 'low')
                y = scipy_signal.lfilter(b, a, y)
            
            if self.settings.get('treble_gain', 0) != 0:
                b, a = self._get_filter_coefficients(sr, 4000, self.settings['treble_gain'], 'high')
                y = scipy_signal.lfilter(b, a, y)
                
        return y

    def _detect_sub_bass_noise(self, y: np.ndarray, sr: int) -> float:
        """
        Detects if low frequencies contain 'noise' (rumble/DC offset) vs 'music' (bass).
        Returns the recommended High-Pass cutoff frequency in Hz.
        """
        try:
            # Analyze a short segment (first 5 seconds or whole file if short)
            analysis_len = int(min(len(y), sr * 5))
            y_analysis = y[:analysis_len]
            
            # Compute STFT for low frequencies only
            n_fft = 4096
            hop = 1024
            
            # Get frequency matrix
            D = np.abs(librosa.stft(y_analysis, n_fft=n_fft, hop_length=hop))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # Isolate Sub-Bass Band (0Hz to 60Hz)
            sub_band_indices = np.where((freqs >= 0) & (freqs <= 60))[0]
            if len(sub_band_indices) == 0: return 20.0
            
            sub_band_energy = D[sub_band_indices, :]
            
            # Calculate Spectral Flatness for this band
            # Flatness close to 1 = Noise (Hiss/Rumble)
            # Flatness close to 0 = Tone (Bass/Synth)
            mean_energy = np.mean(sub_band_energy, axis=0) + 1e-10
            geo_mean_energy = np.exp(np.mean(np.log(sub_band_energy + 1e-10), axis=0))
            flatness = np.mean(geo_mean_energy / mean_energy)
            
            # Thresholding
            # If flatness > 0.5, it's likely noise/rumble. 
            # If flatness < 0.3, it's likely musical content.
            
            if flatness > 0.6:
                # High noise floor. Check where the energy drops off
                # Start cut at 40Hz to clean up heavily
                return 40.0 
            elif flatness > 0.35:
                # Moderate rumble. 
                # Check 20Hz specifically
                idx_20hz = np.argmin(np.abs(freqs - 20))
                idx_40hz = np.argmin(np.abs(freqs - 40))
                
                e_20 = np.mean(D[idx_20hz, :])
                e_40 = np.mean(D[idx_40hz, :])
                
                # If 20Hz is almost as loud as 40Hz, but flatness suggests noise, cut it.
                if e_20 > (e_40 * 0.8):
                    return 30.0
                else:
                    return 20.0
            else:
                # Likely musical bass. Use standard gentle cut.
                return 15.0

        except Exception:
            return 20.0 # Default safe fallback

    # ... (Keep rest of methods identical: _get_filter_coefficients, design_shelf_filter, etc.)
    def _get_filter_coefficients(self, sr: int, freq: float, gain_db: float, filter_type: str):
        key = (sr, freq, gain_db, filter_type)
        if key not in self._filter_cache:
            self._filter_cache[key] = self.design_shelf_filter(sr, freq, gain_db, filter_type)
        return self._filter_cache[key]

    def design_shelf_filter(self, sr: int, freq: float, gain_db: float, filter_type: str) -> Tuple[list, list]:
        if sr <= 0 or freq <= 0 or freq >= sr / 2:
            return [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]
        A = 10 ** (gain_db / 40.0)
        w0 = 2 * np.pi * freq / sr
        beta = np.sqrt(A + (1 / A))
        if filter_type == 'low':
            b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + beta * np.sin(w0))
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - beta * np.sin(w0))
            a0 = (A + 1) + (A - 1) * np.cos(w0) + beta * np.sin(w0)
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
            a2 = (A + 1) + (A - 1) * np.cos(w0) - beta * np.sin(w0)
        else:  # high
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + beta * np.sin(w0))
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - beta * np.sin(w0))
            a0 = (A + 1) - (A - 1) * np.cos(w0) + beta * np.sin(w0)
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - beta * np.sin(w0)
        return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]

    def _get_gamma_lut(self, gamma: float):
        key = gamma
        if self._gamma_cache_key != key or self._gamma_lut is None:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
            self._gamma_lut = table
            self._gamma_cache_key = key
        return self._gamma_lut

    def generate_spectrogram_image(self, y: np.ndarray, sr: int, duration: float) -> Tuple[np.ndarray, int]:
        vis_mode = self.settings.get('visual_mode', 'standard')
        spec_type = self.settings.get('spec_type', 'stft')
        n_fft = self.settings.get('n_fft', 4096)
        hop_length = self.settings.get('hop_length', 512)
        dyn_range = self.settings.get('dynamic_range_db', 80)
        min_freq = self.settings.get('min_freq', 20.0)
        max_freq = self.settings.get('max_freq', 20000.0)
        
        S_db = None

        if spec_type == 'mel':
            n_mels = min(256, self.settings.get('height', 1080))
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=min_freq, fmax=max_freq)
            S_db = librosa.power_to_db(S, ref=np.max)
            if self.settings.get('auto_freq', True):
                min_freq, max_freq = self._detect_bounds_from_mel(S_db, sr, n_mels, min_freq, max_freq)
                max_freq = min(max_freq, float(sr / 2))
                if min_freq >= max_freq: min_freq = 20.0
        elif spec_type == 'cqt':
            hop_length_cqt = hop_length
            S = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length_cqt, fmin=librosa.note_to_hz('C1'), n_bins=84, bins_per_octave=12))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            hop_length = hop_length_cqt
            if self.settings.get('auto_freq', True):
                min_freq, max_freq = self._detect_bounds_from_cqt(S_db, sr)
                max_freq = min(max_freq, float(sr / 2))
                if min_freq >= max_freq: min_freq = 20.0
        else:  # stft
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            if self.settings.get('auto_freq', True):
                min_freq, max_freq = self._detect_bounds_from_stft(S_db, sr, n_fft)
                max_freq = min(max_freq, float(sr / 2))
                if min_freq >= max_freq: min_freq = 20.0

        S_norm_float = (S_db + dyn_range) * (255.0 / dyn_range)
        gray_img = np.clip(S_norm_float, 0, 255).astype(np.uint8)

        gamma = self.settings.get('gamma', 1.0)
        if gamma != 1.0:
            lut = self._get_gamma_lut(gamma)
            gray_img = cv2.LUT(gray_img, lut)

        if vis_mode == 'radial':
            log_img = self._create_radial(gray_img, sr, n_fft, spec_type, min_freq, max_freq)
        elif vis_mode == 'classic':
            log_img = self._create_linear(gray_img)
        else:
            if spec_type == 'stft':
                log_img = self._create_log_freq(gray_img, sr, n_fft, min_freq, max_freq)
            else:
                log_img = self._create_linear(gray_img)

        if self.settings.get('sharpen', True):
            gaussian = cv2.GaussianBlur(log_img, (0, 0), 3.0)
            log_img = cv2.addWeighted(log_img, 1.5, gaussian, -0.5, 0)

        return log_img, hop_length

    # ... (Keep _detect_bounds and _create methods)
    def _detect_bounds_from_stft(self, S_db, sr, n_fft):
        threshold_db = -60.0
        max_db = np.max(S_db)
        if not np.isfinite(max_db): return 20.0, float(sr / 2)
        freq_maxes = np.max(S_db, axis=1)
        freq_mask = freq_maxes > (max_db + threshold_db)
        if not np.any(freq_mask): return 20.0, float(sr / 2)
        indices = np.where(freq_mask)[0]
        bin_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        return float(bin_freqs[max(0, indices[0] - 2)]), float(bin_freqs[min(len(freq_mask) - 1, indices[-1] + 2)])

    def _detect_bounds_from_mel(self, S_db, sr, n_mels, fmin, fmax):
        threshold_db = -60.0
        max_db = np.max(S_db)
        if not np.isfinite(max_db): return fmin, fmax
        freq_maxes = np.max(S_db, axis=1)
        freq_mask = freq_maxes > (max_db + threshold_db)
        if not np.any(freq_mask): return fmin, fmax
        indices = np.where(freq_mask)[0]
        min_mel = librosa.hz_to_mel(fmin)
        max_mel = librosa.hz_to_mel(fmax)
        mels = np.linspace(min_mel, max_mel, n_mels + 2)
        min_idx, max_idx = max(0, indices[0]), min(n_mels - 1, indices[-1])
        return max(20.0, librosa.mel_to_hz(mels[min_idx])), min(float(sr/2), librosa.mel_to_hz(mels[max_idx + 2]))

    def _detect_bounds_from_cqt(self, S_db, sr):
        threshold_db = -60.0
        max_db = np.max(S_db)
        if not np.isfinite(max_db): return 20.0, float(sr / 2)
        freq_maxes = np.max(S_db, axis=1)
        freq_mask = freq_maxes > (max_db + threshold_db)
        if not np.any(freq_mask): return 20.0, float(sr / 2)
        indices = np.where(freq_mask)[0]
        freqs = librosa.cqt_frequencies(S_db.shape[0], fmin=librosa.note_to_hz('C1'))
        return float(freqs[max(0, indices[0])]), float(freqs[min(len(freqs)-1, indices[-1])])

    def _create_linear(self, img):
        target_h = self.settings.get('height', 1080)
        h, w = img.shape[:2]
        if w > 15000:
            img = cv2.resize(img, (15000, h), interpolation=cv2.INTER_AREA)
        return cv2.resize(img, (img.shape[1], target_h), interpolation=cv2.INTER_LANCZOS4)

    def _get_cache_key(self, **kwargs):
        return hash(tuple(sorted(kwargs.items())))

    def _create_log_freq(self, img, sr, n_fft, min_freq, max_freq):
        target_h = self.settings.get('height', 1080)
        cache_key = self._get_cache_key(sr=sr, n_fft=n_fft, min_freq=min_freq, max_freq=max_freq, target_h=target_h, img_w=img.shape[1])
        
        if cache_key in self._remap_cache:
            map_x_grid, map_y_grid = self._remap_cache[cache_key]
            if map_x_grid.shape[1] == img.shape[1]:
                return cv2.remap(img, map_x_grid, map_y_grid, interpolation=cv2.INTER_LANCZOS4)

        if img.shape[1] > 20000:
            img = cv2.resize(img, (20000, img.shape[0]), interpolation=cv2.INTER_AREA)

        out_freqs = np.logspace(np.log10(max(1, min_freq)), np.log10(max_freq), target_h)
        bin_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        y_indices = np.interp(out_freqs, bin_freqs, np.arange(len(bin_freqs))).astype(np.float32)
        map_x = np.arange(img.shape[1], dtype=np.float32)
        map_x_grid, map_y_grid = np.meshgrid(map_x, y_indices)
        self._remap_cache[cache_key] = (map_x_grid, map_y_grid)
        
        log_img = cv2.remap(img, map_x_grid, map_y_grid, interpolation=cv2.INTER_LANCZOS4)
        if log_img.shape[1] > 16000:
            log_img = cv2.resize(log_img, (16000, target_h), interpolation=cv2.INTER_AREA)
        return log_img

    def _create_radial(self, img, sr, n_fft, spec_type, min_freq, max_freq):
        dim = min(self.settings.get('width', 1920), self.settings.get('height', 1080))
        if dim < 100: dim = 800
        nyquist = sr / 2

        if spec_type == 'stft':
            bin_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        elif spec_type == 'mel':
            bin_freqs = np.linspace(0, nyquist, img.shape[0])
        else:
            bin_freqs = librosa.cqt_frequencies(n_bins=84, fmin=librosa.note_to_hz('C1'))

        out_freqs = np.logspace(np.log10(max(1, min_freq)), np.log10(max_freq), dim // 2)
        y_indices = np.interp(out_freqs, bin_freqs, np.arange(len(bin_freqs))).astype(np.float32)
        
        if img.shape[1] > 20000:
            img = cv2.resize(img, (20000, img.shape[0]), interpolation=cv2.INTER_AREA)

        map_x = np.arange(img.shape[1], dtype=np.float32)
        map_x_grid, map_y_grid = np.meshgrid(map_x, y_indices)
        src_log_freq = cv2.remap(img, map_x_grid, map_y_grid, interpolation=cv2.INTER_LANCZOS4)
        src_polar = cv2.resize(src_log_freq, (3600, dim // 2), interpolation=cv2.INTER_LANCZOS4)
        
        center, max_radius = (dim // 2, dim // 2), dim // 2
        flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP + cv2.INTER_LANCZOS4
        return cv2.warpPolar(src_polar, (dim, dim), center, max_radius, flags)