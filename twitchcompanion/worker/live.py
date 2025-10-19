import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from twitchcompanion.logger import logger as MainLogger

# create a child logger
logger = MainLogger.getChild(__name__)

# disable for this file
logger.disabled = False


class TwitchStreamAudio:
    def __init__(self, twitch_url: str):
        self.twitch_url = twitch_url
        self.buffer_queue = queue.Queue(maxsize=10)
        self.running = False
        self.ffmpeg_proc = None

    def start(self):
        streamlink_cmd = ["streamlink", "--stdout", self.twitch_url, "audio_only"]
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-vn",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "pipe:1",
        ]
        
        streamlink_proc = subprocess.Popen(
            streamlink_cmd,
            stdout=subprocess.PIPE,
            # stderr=subprocess.DEVNULL,
        )
        self.ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=streamlink_proc.stdout,
            stdout=subprocess.PIPE,
            # stderr=subprocess.DEVNULL,
            bufsize=4096,
        )

        print("Launching Streamlink and FFmpeg...")
        print("Streamlink PID:", streamlink_proc.pid)
        print("FFmpeg PID:", self.ffmpeg_proc.pid)

        self.running = True
        threading.Thread(target=self._reader_loop, daemon=True).start()

    def _reader_loop(self):
        chunk_size = 16000 * 2  # 1 second of 16kHz mono PCM16
        while self.running:
            data = self.ffmpeg_proc.stdout.read(chunk_size)
            if self.ffmpeg_proc.poll() is not None:
                break
            if not data:
                break
            try:
                self.buffer_queue.put_nowait(data)
            except queue.Full:
                _ = self.buffer_queue.get_nowait()  # drop oldest
                self.buffer_queue.put_nowait(data)

    def read_audio(self, duration_sec=3):
        total_bytes = duration_sec * 16000 * 2
        buf = bytearray()
        while len(buf) < total_bytes:
            try:
                buf.extend(self.buffer_queue.get(timeout=1))
            except queue.Empty:
                break
        audio = np.frombuffer(buf, np.int16).astype(np.float32) / 32768.0
        return audio

    def stop(self):
        self.running = False
        if self.ffmpeg_proc:
            try:
                self.ffmpeg_proc.stdout.close()
                self.ffmpeg_proc.terminate()
                self.ffmpeg_proc.wait(timeout=2)
            except Exception:
                pass

class LiveWhisperTranscriber:
    def __init__(self, streamer: TwitchStreamAudio, out_dir: str, model_size="small", device="cuda", window_sec=6, step_sec=3,):
        self.streamer = streamer
        self.model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
        self.running = False
        self.transcript = []
        self.sample_rate = 16000
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.buffer = np.zeros(self.window_sec * self.sample_rate, dtype=np.float32)
        self._lock = threading.Lock()
        self.out_file = Path(out_dir) / "transcript.txt"

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def get_latest_transcription(self, n=30) -> str:
        """Get the latest transcription from the transcript file."""
        try:
            with(open(self.out_file, "r", encoding="utf-8")) as f:
                lines = f.readlines()

            if not lines:
                logger.warning("Transcript file is empty.")
                return ""

            transcriptions = lines[-n:]
            transcriptions = [line.strip() for line in transcriptions if line.strip()]
            transcriptions = [line.split("  ")[-1] for line in transcriptions]
            return transcriptions
        
        except FileNotFoundError:
            return ""

    def _loop(self):
        while self.running:
            # Read new audio (3 s chunk)
            new_audio = self.streamer.read_audio(duration_sec=self.step_sec)
            if len(new_audio) == 0:
                time.sleep(0.5)
                continue

            # Slide the window: drop oldest, append newest
            with self._lock:
                self.buffer = np.roll(self.buffer, -len(new_audio))
                self.buffer[-len(new_audio):] = new_audio.copy()

            # Transcribe the full window
            segments, _ = self.model.transcribe(self.buffer, beam_size=1)
            text = " ".join([seg.text.strip() for seg in segments])
            if text.strip():
                with open(self.out_file, 'a', encoding='utf-8') as f:
                    f.write(f"{text}\n")
                self.transcript.append(text)

            time.sleep(self.step_sec * 0.9)  # smooth pacing

    def stop(self):
        self.running = False