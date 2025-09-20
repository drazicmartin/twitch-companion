import os
import subprocess
import threading
from pathlib import Path

from twitchcompanion.logger import logger as MainLogger

# create a child logger
logger = MainLogger.getChild(__name__)

# disable for this file
logger.disabled = True


class TwitchRecorder:
    def __init__(self, twitch_url: str, audio_dir: str, segment_time: int):
        self.twitch_url = twitch_url
        self.audio_dir = audio_dir
        self.segment_time = segment_time
        self._running = False
        self._streamlink_proc = None
        self._ffmpeg_proc = None
        self._thread = None

    def _record_loop(self):
        os.makedirs(self.audio_dir, exist_ok=True)

        streamlink_cmd = ["streamlink", "--stdout", self.twitch_url, "audio_only"]
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i", "pipe:0",
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-f", "segment",
            "-segment_time", str(self.segment_time),
            "-reset_timestamps", "1",
            os.path.join(self.audio_dir, "chunk_%05d.wav"),
        ]

        # Open a file to save both stdout and stderr
        ffmpeg_file = open(Path(self.audio_dir).parent / "ffmpeg.log", "a")
        streamlink_file = open(Path(self.audio_dir).parent / "streamlink.log", "a")

        self._streamlink_proc = subprocess.Popen(streamlink_cmd, stdout=subprocess.PIPE, stderr=streamlink_file)
        self._ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=self._streamlink_proc.stdout, stdout=ffmpeg_file, stderr=ffmpeg_file)

        self._running = True
        logger.info("Started recording Twitch stream.")

    def start(self):
        """Starts recording in a separate thread so Ray call returns immediately."""
        if self._thread and self._thread.is_alive():
            logger.warning("Recording is already running.")
            return

        self._record_loop()

    def stop(self):
        """Stops recording gracefully."""
        logger.info("Stopping recording...")
        self._running = False
        if self._streamlink_proc:
            self._streamlink_proc.terminate()
            self._streamlink_proc.wait()
            self._streamlink_proc = None
        if self._ffmpeg_proc:
            self._ffmpeg_proc.terminate()
            self._ffmpeg_proc.wait()
            self._ffmpeg_proc = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Recording stopped.")
