import json
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv
from mistralai import Mistral

from twitchcompanion.logger import logger as MainLogger
from twitchcompanion.twitch import TwitchClient
from twitchcompanion.worker import TwitchRecorder, TwitchTranscriber

load_dotenv()

# create a child logger
logger = MainLogger.getChild(__name__)
logger.disabled = False

class TwitchWatcher:
    def __init__(self, channel: str, check_interval: int = 30, response_interval: int = 60, words_file = None, **kwargs):
        self.channel = channel
        self.check_interval = check_interval
        self.response_interval = response_interval
        self.running = False
        self.recorder = None
        self.transcriber = None
        self.output_dir = Path("output") / self.channel
        self.twitch_url = f"https://twitch.tv/{self.channel}"
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
                with open(self.output_dir / "stream_info.json", "w") as f:
                    json.dump(data, f, indent=4)
                return True
            return False
        except Exception as e:
            print("Error checking stream:", e)
            return False

    def start(self, whisper_model_size: str = "small"):
        while True:
            online = self.is_stream_online()
            if online and not self.running:
                print("✅ Stream is live! Starting recording + transcription...")
                self.running = True

                audio_dir = self.output_dir / "audio_chunks"

                # spawn recorder + transcriber
                self.transcriber = TwitchTranscriber(
                    audio_dir=audio_dir, 
                    segment_time=self.segment_time,
                    whisper_model_size=whisper_model_size,
                    words_file = self.words_file,
                )
                self.recorder = TwitchRecorder(
                    self.twitch_url, 
                    audio_dir=audio_dir,
                    segment_time=self.segment_time,
                )

                self.recorder.start()
                self.transcriber.start()

            elif not online and self.running:
                print("❌ Stream went offline. Stopping...")
                self.stop()
            
            if self.running:
                self.generate_response()

            time.sleep(self.check_interval)

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
        
        if message in self.response_history:
            return False

        return True

    def generate_response(self):
        """Generate a response based on the latest transcription."""
        if not self.should_respond():
            return  # too soon to respond again

        logger.info("Generating response...")

        latest_transcription = self.transcriber.get_latest_transcription()
        logger.debug(f"Latest transcription: {latest_transcription}")

        role_message = ("" +
            "You are a stream companion," +
            "Your role is to interact with the streamer and the viewers in a fun and engaging way." +
            "You must write the answer in french." +
            "Your name is NIOX, The streamer can ask you specific thing by calling you NIOX, you must anwser his questions and though",
            f"The title of the stream is {self.title}. " +
            f"The streamer name is {self.channel}. He is actually playing {self.category}." +
            "You must interact with the streamer based on what he his saying. Keep your responses short and engaging." +
            "You can make jokes and be funny or even silly. Do not mention you are an AI model or chatbot." +
            "Caution some of the things in the transcription may be inaccurate. and some may come from the background music do not take these into account" +
            "Here are some recent things that have been said comming from the automatic transcription:\n" + "\n".join(latest_transcription)
        )

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
            max_tokens=15,
            safe_prompt=True,
            random_seed=self.num_response,
        )

        # TODO add chat messages to context if possible

        message = chat_response.choices[0].message.content

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
        with open(self.response_file, 'a') as f:
            ctime = time.ctime()
            f.write(f"{ctime} : {message}\n")
        self.last_response = time.time()

    def stop(self):
        self.running = False
        if self.recorder:
            self.recorder.stop()
            self.recorder = None
        if self.transcriber:
            self.transcriber.stop()
            self.transcriber = None
        print("TwitchWatcher stopped.")

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Twitch Stream Watcher")
    parser.add_argument("--channel", type=str, required=True, help="Twitch channel name to watch")
    parser.add_argument("--check-interval", type=int, default=30, help="Interval (in seconds) to check if stream is live")
    parser.add_argument("--response-interval", type=int, default=60, help="Interval (in seconds) between chat responses")
    parser.add_argument("--whisper-model-size", type=str, default="medium", choices=["tiny", "base", "small", "medium", "large"], help="Size of the Whisper model to use for transcription")
    parser.add_argument("--delete-audio", action="store_true", default=False, help="Delete audio file after processing")
    parser.add_argument('--word-file', type=str, default="twitchcompanion/words.txt", help="File of words to activate response on call")
    parser.add_argument('--no-send', action="store_true", default=False, help="Do not send message to streamer")
    return parser

def main():
    args = get_args().parse_args()
    args_dict = vars(args)
    
    watcher = TwitchWatcher(**args_dict)

    try:
        # Start watching (this will manage recorder + transcriber automatically)
        watcher.start(whisper_model_size=args.whisper_model_size)

        print("Watcher running. Press Ctrl+C to stop.")
        import time
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping TwitchWatcher...")
        watcher.stop()  # optional: if you implement stop in TwitchWatcher
        print("Watcher stopped.")


if __name__ == "__main__":
    main()