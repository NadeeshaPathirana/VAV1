import simpleaudio as sa
from pathlib import Path
from openai import OpenAI

# Initialize client
client = OpenAI(api_key="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")

# Output file path
speech_file_path = Path("output.wav")

# Request speech synthesis
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",   # OpenAI TTS model
    voice="alloy",             # voice options: alloy, verse, etc.
    input="Hello! This is OpenAI speaking."
) as response:
    response.stream_to_file(speech_file_path)

# Play the audio
wave_obj = sa.WaveObject.from_wave_file(str(speech_file_path))
play_obj = wave_obj.play()
play_obj.wait_done()
