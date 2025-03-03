import os
import time
import simpleaudio as sa
from TTS.api import TTS

def play_text_to_speech(text, emotion="neutral", pitch=1.0, speed=1.0):
    # Start time measurement
    start_time = time.time()

    # Initialize TTS (Coqui TTS)
    tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)

    # Customize the voice by modifying parameters
    tts.tts_to_file(text=text, file_path="temp_audio.wav", speed=speed)

    # Play the audio
    wave_obj = sa.WaveObject.from_wave_file("temp_audio.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()

    # Remove the temp audio file after playing
    os.remove("temp_audio.wav")

    # End time measurement
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"TTS Execution Time: {execution_time:.2f} seconds")

