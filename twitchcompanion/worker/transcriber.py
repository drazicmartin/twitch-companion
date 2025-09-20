import contextlib
import msvcrt
import os
import time
import wave
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import webrtcvad
import whisper

from twitchcompanion.logger import logger as MainLogger

# create a child logger
logger = MainLogger.getChild(__name__)

# disable for this file
logger.disabled = False


def has_speech(wav_path: str, aggressiveness: int = 1) -> bool:
    """
    Return True if speech is detected in the WAV file.
    
    aggressiveness: 0 (least) -> 3 (most aggressive)
    """
    return True
    vad = webrtcvad.Vad(aggressiveness)

    with contextlib.closing(wave.open(str(wav_path), 'rb')) as wf:
        sample_rate = wf.getframerate()
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("webrtcvad only supports 8,16,32,48kHz audio")
        n_channels = wf.getnchannels()
        if n_channels != 1:
            raise ValueError("webrtcvad only supports mono audio")
        
        frame_duration = 30  # ms
        frame_size = int(sample_rate * frame_duration / 1000) * 2  # 16-bit samples
        
        has_voice = False
        while True:
            frame = wf.readframes(frame_size // 2)
            if len(frame) < frame_size:
                break
            if vad.is_speech(frame, sample_rate):
                has_voice = True
                break
        return has_voice

def is_valid_wav(wav_path: str) -> bool:
    """
    Check if a WAV file is valid.
    Returns True if it can be opened and has >0 frames, mono/stereo.
    """
    try:
        with contextlib.closing(wave.open(str(wav_path), 'rb')) as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            if n_frames == 0:
                return False
            if n_channels not in (1, 2):
                return False
            if sample_rate not in (8000, 16000, 32000, 44100, 48000):
                return False
    except (wave.Error, EOFError, FileNotFoundError):
        return False
    return True

def get_create_time(file_path: str) -> float:
    return os.path.getctime(file_path)


class TwitchTranscriber:
    def __init__(self, audio_dir: str, segment_time: int, whisper_model_size: str = "medium",  n_models: int = 1):
        self.audio_dir = Path(audio_dir)
        self.out_file = self.audio_dir.parent / "transcript.txt"
        self.running = True
        self.do_remove = False
        self.segment_time = segment_time

        # Load multiple models
        self.n_models = n_models
        self.worker_threads = []
        logger.info(f"Loading {n_models} Whisper models of size '{whisper_model_size}'...")
        self.models = [whisper.load_model(whisper_model_size, device="cuda") for _ in range(n_models)]

        # Queue of files ready to transcribe
        self.file_queue = Queue()

        logger.info("Transcriber initialized.")

    def start(self):
        # Start worker threads
        logger.info("Starting transcription workers...")
        for idx in range(self.n_models):
            t = Thread(target=self._model_worker, args=(idx,), daemon=True)
            t.start()
            self.worker_threads.append(t)

        # Start scanner thread
        self.scanner_thread = Thread(target=self._scanner_loop, daemon=True)
        self.scanner_thread.start()

    def remove_file(self, path: Path):
        if self.do_remove:
            for _ in range(10):
                try:
                    os.remove(path)
                    break
                except PermissionError:
                    logger.debug(f"Retrying delete of {path}")
                    time.sleep(0.5)

    def _scanner_loop(self):
        """Continuously scan the folder for ready files and put them in queue."""
        seen = set()
        logger.info("Starting file scanner...")
        while self.running:
            for wav_file in self.audio_dir.glob("*.wav"):
                created_time = get_create_time(wav_file)
                if wav_file not in seen and created_time + (self.segment_time + 2) < time.time():
                    if has_speech(wav_file):
                        logger.info(f"Detected new file: {wav_file}")
                        self.file_queue.put(wav_file)
                    else:
                        logger.info(f"Skipping silent file: {wav_file}")
                        self.remove_file(wav_file)
                    seen.add(wav_file)
            time.sleep(1)

    def _is_file_ready(self, path: Path) -> bool:
        """Check if file is stable (not growing)."""
        try:
            size1 = path.stat().st_size
            time.sleep(1)
            size2 = path.stat().st_size
            return size1 == size2
        except FileNotFoundError:
            return False

    def _model_worker(self, model_idx: int):
        while self.running:
            try:
                wav_file = self.file_queue.get(timeout=1)
            except Empty:
                continue

            try:
                start_size = os.path.getsize(wav_file)
                logger.debug(f"Model {model_idx} picked up {Path(wav_file).name} (size: {start_size})")
                logger.info(f"Model {model_idx} transcribing {wav_file}...")
                ctime = time.ctime()
                result = self.models[model_idx].transcribe(str(wav_file))
                logger.info(f"Model {model_idx} finished {wav_file}")
                with open(self.out_file, "a", encoding="utf-8") as f:
                    f.write(f"{ctime} {result['text']}\n")

                self.file_queue.task_done()
                self.remove_file(wav_file)
            except Exception as e:
                logger.error(f"Error transcribing {wav_file}: {e}")

    def stop(self):
        self.running = False
        
        # Add a reasonable timeout (e.g., 5 seconds)
        timeout = 5.0
        
        # Wait for threads with timeout
        for t in self.worker_threads:
            t.join(timeout=timeout)
        self.scanner_thread.join(timeout=timeout)
        
        # Check if threads are still alive
        if any(t.is_alive() for t in self.worker_threads) or self.scanner_thread.is_alive():
            raise RuntimeError("Threads failed to stop within timeout")
