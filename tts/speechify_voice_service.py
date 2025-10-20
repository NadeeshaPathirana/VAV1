from speechify import Speechify
from speechify.core.api_error import ApiError
import simpleaudio as sa
import base64

client = Speechify(
    token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NTk0OTYyMDAsImlzcyI6InNwZWVjaGlmeS1hcGkiLCJzY29wZSI6ImF1ZGlvOmFsbCB2b2ljZXM6cmVhZCIsInN1YiI6IjR3aFJTT1cybEhOMTJTQjdxYkQ1OWhUelJ2ZjEifQ.qReW5VureZ9QPo5EeOm8P0dyWPgFAdv-_LZ1O3lJbQg",
)

def play_text_to_speech(text):
    try:
        response = client.tts.audio.speech(
            input=text,
            voice_id="lisa",  # Replace with a real Speechify voice ID
            model="simba-english",
            # emotion='sad',
        )

        audio_bytes = base64.b64decode(response.audio_data)

        # Save audio file
        with open("output.wav", "wb") as f:
            f.write(audio_bytes)

        # Play the WAV file
        wave_obj = sa.WaveObject.from_wave_file("output.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()

    except ApiError as e:
        print(e.status_code)
        print(e.body)

play_text_to_speech("<speak><speechify:style emotion=\"angry\">Hi there </speechify:style></speak>")