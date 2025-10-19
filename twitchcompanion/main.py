import json
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv
from mistralai import Mistral
import torch

from twitchcompanion.logger import logger as MainLogger
from twitchcompanion.twitch import TwitchClient
from twitchcompanion.worker import TwitchRecorder, TwitchTranscriber
from twitchcompanion.worker.live import TwitchStreamAudio, LiveWhisperTranscriber

load_dotenv()

# create a child logger
logger = MainLogger.getChild(__name__)
logger.disabled = False

class TwitchWatcher:
    def __init__(self, channel: str, check_interval: int = 30, response_interval: int = 60, words_file = None, live_mode: bool = True, **kwargs):
        self.channel = channel
        self.check_interval = check_interval
        self.response_interval = response_interval
        self.running = False
        self.live_mode = live_mode
        self.output_dir = Path("output") / self.channel
        self.twitch_url = f"https://twitch.tv/{self.channel}"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components based on mode
        self.stream_audio = None
        self.recorder = None
        self.transcriber = None
        os.makedirs(self.output_dir, exist_ok=True)
        self.api_key = os.environ["MISTRAL_API_KEY"]
        self.model = "mistral-small-latest"
        self.client = Mistral(api_key=self.api_key)
        self.title = "Unknown Title"
        self.category = "Unknown Game"
        self.broadcaster_id = None
        self.response_history = []
        self.segment_time = 10
        self.num_response = 0
        self.on_work_response = words_file is not None
        self.words_file = words_file
        self.no_send = kwargs.get('no_send', False)
        self.no_ai = kwargs.get('no_ai', False)

        self.twitch_client = TwitchClient(self.channel)

        self.kwargs = kwargs

        self.response_file = Path(self.output_dir) / "response.txt"
        self.last_response = time.time() + response_interval
        logger.info(f"Watching Twitch channel: {self.twitch_url}")

    def is_stream_online(self) -> bool:
        try:
            result = subprocess.run(
                ["streamlink", "--json", f"twitch.tv/{self.channel}", "audio_only"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                self.title = data['metadata'].get("title", "Unknown Title")
                self.category = data['metadata'].get("category", "Unknown Game")
                self.broadcaster_id = data['metadata'].get("id", None)
                with open(self.output_dir / "stream_info.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                return True
            return False
        except Exception as e:
            print("Error checking stream:", e)
            return False

    def start_workers(self, whisper_model_size):
        if self.live_mode:
            # Live mode: use real-time audio processing
            self.stream_audio = TwitchStreamAudio(
                self.twitch_url
            )
            self.transcriber = LiveWhisperTranscriber(
                self.stream_audio,
                out_dir=self.output_dir,
                model_size=whisper_model_size,
                device=self.device,
            )
            self.stream_audio.start()
            self.transcriber.start()
        else:
            # Recorded mode: use chunked audio files
            audio_dir = self.output_dir / "audio_chunks"
            self.transcriber = TwitchTranscriber(
                audio_dir=audio_dir, 
                segment_time=self.segment_time,
                whisper_model_size=whisper_model_size,
                words_file=self.words_file,
            )
            self.recorder = TwitchRecorder(
                self.twitch_url, 
                audio_dir=audio_dir,
                segment_time=self.segment_time,
            )
            self.recorder.start()
            self.transcriber.start()

    def _loop(self):
        tic = time.time()
        while self.running:
            self.transcriber.solo_main()
            time.sleep(1)
            if time.time() - tic > self.response_interval:
                self._response_main()
                tic = time.time() 

    def start(self, whisper_model_size: str = "small"):
        while True:
            online = self.is_stream_online()
            if online and not self.running:
                print("✅ Stream is live! Starting transcription...")
                self.running = True

                self.start_workers(whisper_model_size=whisper_model_size)

            elif not online and self.running:
                print("❌ Stream went offline. Stopping...")
                self.stop()
            
            if self.running:
                self._response_main()

            time.sleep(self.check_interval)

    def stop(self):
        self.running = False
        if self.live_mode:
            if self.stream_audio:
                self.stream_audio.stop()
            if self.transcriber:
                self.transcriber.stop()
        else:
            if self.recorder:
                self.recorder.stop()
            if self.transcriber:
                self.transcriber.stop()

    def should_respond(self):
        if self.response_interval is not None:
            if self.last_response + self.response_interval < time.time():
                return True
        if self.on_work_response and self.word_flagged:
            return True
            
        return False

    def should_send(self, message: str):
        if self.no_send:
            return False

        if self.no_ai:
            return False
        
        if message in self.response_history:
            return False

        return True

    def handle_send(self, message: str):
        if self.should_send(message):
            self.response_history.append(message)
            try:
                self.twitch_client.send_message(
                    message=message
                )
                self.num_response += 1
            except Exception as e:
                logger.error(f"Error sending message to Twitch chat: {e}")

        logger.debug(f"Response generated: {message}")
        with open(self.response_file, 'a', encoding="utf-8") as f:
            ctime = time.ctime()
            f.write(f"{ctime} : {message}\n")
        self.last_response = time.time()

    def _response_main(self):
        """Generate a response based on the latest transcription."""
        if not self.should_respond():
            return  # too soon to respond again

        logger.info("Generating response...")

        latest_transcription = self.transcriber.get_latest_transcription()
        
        logger.debug(f"Latest transcription: {latest_transcription}")

        role_message = (
            "You are a stream companion. "
            "Your role is to interact with the streamer and the viewers in a fun and engaging way. "
            "You must write the answer in French. "
            "Your name is NIOX; the streamer can ask you specific things by calling you NIOX, and you must answer accordingly. "
            f"The title of the stream is {self.title}. "
            f"The streamer name is {self.channel}. He is currently playing {self.category}. "
            "You must interact with the streamer based on what he is saying. Keep your responses short and engaging. "
            "You can make jokes and be funny or even silly. Do not mention you are an AI model or chatbot. "
            "Caution: some of the things in the transcription may be inaccurate or come from the background music. "
            "Here are some recent things that have been said coming from the automatic transcription:\n"
            + "\n".join(latest_transcription)
        )

        if not self.no_ai:
            chat_response = self.client.chat.complete(
                model = self.model,
                messages = [
                    {
                        "role": "system",
                        "content": role_message
                    },
                    {
                        "role": "user",
                        "content": "Write a short message to the chat based on the above context without apostrophes."
                    }
                ],
                temperature=0.4,
                max_tokens=30,
                safe_prompt=True,
                random_seed=self.num_response,
            )

            # TODO add chat messages to context if possible

            message = chat_response.choices[0].message.content
        else:
            message = "Je suis désolé, je ne peux pas répondre pour le moment."

        self.handle_send(message)