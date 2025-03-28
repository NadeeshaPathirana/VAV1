import time
import pyaudio
import numpy as np
from faster_whisper import WhisperModel

from tts import google_voice_service as vs
from rag.AIVA import AIVA
from rag.AIVA_Chroma import AIVA_Chroma

DEFAULT_MODEL_SIZE = "medium"
DEFAULT_CHUNK_LENGTH = 4

# ai_assistant = AIVoiceAssistant() # first version
# ai_assistant = AIVA() # second version
ai_assistant = AIVA_Chroma()

# trying to optimise the recording process

def is_silence(data, threshold=500):
    """Check if audio data contains silence."""
    return np.max(np.abs(data)) < threshold
def record_audio_chunk(stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    # Convert to numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Check for silence
    if is_silence(audio_data):
        return None  # Indicate silence
    else:
        return audio_data  # Return audio chunk


def transcribe_audio(model, audio_data):
    """Transcribe audio directly from numpy array."""
    segments, _ = model.transcribe(audio_data, beam_size=7)
    return ' '.join(segment.text for segment in segments)


def detect_pause(start_time, pause_duration=2.0):
    """Check if silence has lasted long enough to trigger a stop."""
    return (time.time() - start_time) >= pause_duration


def main():
    model_size = DEFAULT_MODEL_SIZE + ".en"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    try:
        while True:
            print("_")
            concat_transcription = ''
            start_silence_time = None

            while True:
                start_time = time.time()
                audio_data = record_audio_chunk(stream)

                if audio_data is None:
                    # Start counting silence time
                    if start_silence_time is None:
                        start_silence_time = time.time()

                    if detect_pause(start_silence_time):
                        print("Silence detected. Stopping recording...")
                        end_time = time.time()  # End time measurement
                        execution_time = end_time - start_time
                        print(
                            f"Recording Audio Execution Time: {execution_time:.2f} seconds")  # Print the total execution time
                        break  # Stop recording
                else:
                    start_silence_time = None  # Reset silence timer

                    # Transcribe on the fly
                    transcription = transcribe_audio(model, audio_data)
                    # print(f"Transcription: {transcription}")
                    concat_transcription += " " + transcription

            if concat_transcription.strip():
                print(f"User: {concat_transcription}")
                response = ai_assistant.interact_with_llm(concat_transcription)

                if response:
                    response = response.lstrip()
                    print(f"AI Assistant: {response}")

                    # Stop mic input to avoid feedback
                    stream.stop_stream()

                    # Play the AI response
                    vs.play_text_to_speech(response)

                    # Restart mic input
                    stream.start_stream()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()