from twitchcompanion.main import TwitchWatcher
from twitchcompanion.agent import TwitchAgent

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
    parser.add_argument('--no-ai', action="store_true", default=False, help="Do not call mistral")
    parser.add_argument('--live-mode', action="store_true", default=False, help="Use live audio processing instead of recorded chunks")
    return parser

def main():
    args = get_args().parse_args()
    args_dict = vars(args)
    
    # watcher = TwitchWatcher(**args_dict)
    agent = TwitchAgent(**args_dict)  # Initialize agent (if needed)
    

    try:
        # Start watching (this will manage recorder + transcriber automatically)
        agent.start(whisper_model_size=args.whisper_model_size)

        print("Watcher running. Press Ctrl+C to stop.")
        import time
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping TwitchWatcher...")
        agent.stop()  # optional: if you implement stop in TwitchWatcher
        print("Watcher stopped.")


if __name__ == "__main__":
    main()