import concurrent.futures
import os
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import whisper

from twitchcompanion.logger import logger as MainLogger
from twitchcompanion.utils import get_create_time, has_speech

# create a child logger
logger = MainLogger.getChild(__name__)

# disable for this file
logger.disabled = False


class TwitchTranscriber:
    def __init__(self, audio_dir: str, segment_time: int, whisper_model_size: str = "medium",  n_models: int = 1, words_file: str = None):
        self.audio_dir = Path(audio_dir)
        self.out_file = self.audio_dir.parent / "transcription.txt"
        self.running = True
        self.do_remove = False
        self.segment_time = segment_time

        # Load multiple models
        self.n_models = n_models
        self.worker_threads = []
        logger.info(f"Loading {n_models} Whisper models of size '{whisper_model_size}'...")
        self.models = [whisper.load_model(whisper_model_size, device="cuda" if self.n_models == 1 else "cpu") for _ in range(n_models)]

        self.flagged = False
        self.flag_words = []
        if words_file is not None:
            with open(words_file, 'r', encoding="utf-8") as f:
                lines = f.readlines()
            self.flag_words = lines

        # Queue of files ready to transcribe
        self.file_queue = Queue()
        self.seen = set()

        logger.info("Transcriber initialized.")

    def solo_main(self):
        """Single-threaded transcription loop."""
        if self.running:
            try:
                wav_file = self.file_queue.get(timeout=1)
            except Empty:
                self._scanner_main()
                return

            self._model_main(self.models[0], wav_file)

    def start(self):
        # Start worker threads
        if self.n_models == 1:
            return # managed by parent class
        else:
            logger.info("Starting transcription workers...")
            for idx in range(self.n_models):
                t = Thread(target=self._model_worker_loop, args=(idx,), daemon=True)
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

    def flag_check(self, transcription: str):
        transcription_lower = transcription.lower()
        for word in self.flag_words:
            if word in transcription_lower:
                return True
        return False

    def _scanner_main(self):
        for wav_file in self.audio_dir.glob("*.wav"):
            created_time = get_create_time(wav_file)
            if wav_file not in self.seen and created_time + (self.segment_time + 1.5) < time.time():
                if has_speech(wav_file):
                    logger.info(f"Detected new file: {wav_file}")
                    self.file_queue.put(wav_file)
                else:
                    logger.info(f"Skipping silent file: {wav_file}")
                    self.remove_file(wav_file)
                self.seen.add(wav_file)

    def _scanner_loop(self):
        """Continuously scan the folder for ready files and put them in queue."""
        logger.info("Starting file scanner...")
        while self.running:
            self._scanner_main()
            time.sleep(1)

    def get_latest_transcription(self, n=30) -> str:
        """Get the latest transcription from the transcript file."""
        try:
            with(open(self.out_file, "r", encoding="utf-8")) as f:
                lines = f.readlines()

            if not lines:
                logger.warning("Transcript file is empty.")
                return ""

            # Get last 30 lines, 5min for 10s segments
            transcriptions = lines[-n:]
            transcriptions = [line.strip() for line in transcriptions if line.strip()]
            transcriptions = [line.split("  ")[-1] for line in transcriptions]
            return transcriptions
        
        except FileNotFoundError:
            return ""

    def _model_main(self, model, wav_file: Path):
        try:
            start_size = os.path.getsize(wav_file)
            logger.debug(f"Model picked up {Path(wav_file).name} (size: {start_size})")
            logger.info(f"Model transcribing {wav_file}...")
            ctime = time.ctime()
            result = model.transcribe(str(wav_file))
            logger.info(f"Model finished {wav_file}")
            transcription = result['text']
            self.flag_check(transcription)
            with open(self.out_file, "a", encoding="utf-8") as f:
                f.write(f"{ctime} {transcription}\n")

            self.file_queue.task_done()
            self.remove_file(wav_file)
        except Exception as e:
            logger.error(f"Error transcribing {wav_file}: {e}")

    def _model_worker_loop(self, model_idx: int):
        while self.running:
            try:
                wav_file = self.file_queue.get(timeout=1)
            except Empty:
                continue

            self._model_main(self.models[model_idx], wav_file)

    def stop(self):
        self.running = False
        
        # Add a reasonable timeout (e.g., 5 seconds)
        timeout = 5.0
        
        if self.n_models == 1:
            return  # no threads to stop
        # Wait for threads with timeout
        for t in self.worker_threads:
            t.join(timeout=timeout)
        self.scanner_thread.join(timeout=timeout)
        
        # Check if threads are still alive
        if any(t.is_alive() for t in self.worker_threads) or self.scanner_thread.is_alive():
            raise RuntimeError("Threads failed to stop within timeout")
