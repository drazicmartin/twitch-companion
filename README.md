# Interactive Twitch Stream Companion

- Using Whisper for live transcription of Twitch streams.
    - possibility to have multiple models loaded for parallel transcription.
- Using Mistal API for generating chat responses based on the transcription.
- Using Twitch API for interacting with Twitch chat.

## Start the bot
```bash
# inside .env
MISTRAL_API_KEY=TODO
TWITCH_ACCESS_TOKEN=TODO

python twitchcompanion/main.py --channel <channel_name> --whisper_model_size <tiny|base|small|medium|large>

## example
python twitchcompanion/main.py --channel etaneex --whisper-model-size medium
```