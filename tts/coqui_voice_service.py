import os
import time
import simpleaudio as sa
from TTS.api import TTS

tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)


def play_text_to_speech(text, emotion="neutral", pitch=1.0, speed=1.0):
    start_time = time.time()

    # Generate TTS to a temporary file
    tts.tts_to_file(text=text, file_path="temp_audio.wav", speed=speed)

    # Play the audio
    wave_obj = sa.WaveObject.from_wave_file("temp_audio.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()

    # Remove temp file
    os.remove("temp_audio.wav")

    end_time = time.time()
    print(f"TTS Execution Time: {end_time - start_time:.2f} seconds")
