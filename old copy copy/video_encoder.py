import subprocess
from utils import StreamReader

class VideoEncoder:
    def __init__(self, target_w, target_h, fps, audio_path, output_path, use_gpu):
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
            extra_flags = ['-rc', 'vbr', '-cq', '18', '-b:v', '15M', '-maxrate', '20M', '-profile:v', 'high', '-movflags', '+faststart']
        else:
            codec, preset = 'libx264', 'medium' 
            extra_flags = ['-crf', '20', '-profile:v', 'high', '-level', '4.2', '-movflags', '+faststart']
        
        return [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', 
            '-s', f'{self.target_w}x{self.target_h}', '-pix_fmt', 'rgb24', 
            '-r', str(self.fps), '-i', '-', '-i', self.audio_path, 
            '-map', '0:v:0', '-map', '1:a:0', '-c:v', codec, '-preset', preset, 
            *extra_flags, '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '320k', 
            '-shortest', self.output_path
        ]

    def start(self):
        cmd = self._build_cmd()
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        self.stderr_reader = StreamReader(self.process.stderr)
        self.stderr_reader.start()

    def write(self, data):
        try:
            self.process.stdin.write(data)
            return True
        except BrokenPipeError:
            return False

    def finish(self):
        try:
            if self.process.stdin: self.process.stdin.close()
        except Exception: pass
        
        try: 
            self.process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            print("[Encoder] Error: FFmpeg finalization timed out. Terminating.")
            self.process.terminate()
            self.process.wait()
        
        if self.stderr_reader:
            self.stderr_reader.join(timeout=5)
            return self.stderr_reader.get_output()
        return ""

    def is_alive(self):
        return self.process and self.process.poll() is None