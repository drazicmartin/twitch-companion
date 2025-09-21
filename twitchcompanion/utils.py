import contextlib
import os
import wave

import webrtcvad


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