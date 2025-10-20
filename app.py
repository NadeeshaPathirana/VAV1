import time
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
from scipy.io import wavfile

from emotion_recognition.SpeechEmotionRecognizer import SpeechEmotionRecognizer
# from emotion_recognition.SpeechEmotionRecognizerV2 import SpeechEmotionRecognizerV2
# from tts import google_voice_service as vs
# from tts import pyttx_tts_voice_service as vs
from tts import coqui_voice_service as vs
# from tts import speechify_voice_service as vs
from rag.AIVA import AIVA
from rag.AIVA_Chroma import AIVA_Chroma

DEFAULT_MODEL_SIZE = "small"  # set from medium to small to improve speed of the transcription
DEFAULT_CHUNK_LENGTH = 0.5  # smaller this value -> audio recording is efficient, but can only record very small chunks

# ai_assistant = AIVoiceAssistant() # first version
# ai_assistant = AIVA() # second version
ai_assistant = AIVA_Chroma()
recognizer = SpeechEmotionRecognizer("C:/Users/220425722/Desktop/Python/Emotion Recognition/saved_model/Model_18.1/")
# recognizer = SpeechEmotionRecognizerV2()

# V3 - trying to optimise the recording process

def is_silence(data, threshold=200): # threshold = max amplitude
    """Check if audio data contains silence."""
    return np.max(np.abs(data)) < threshold


def record_audio_chunk(stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    start_time = time.time()  # Start time measurement
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

    # Convert to numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Check for silence
    if is_silence(audio_data):
        end_time = time.time()  # End time measurement
        execution_time = end_time - start_time
        print(f"Record Audio Execution Time - is silence: {execution_time:.2f} seconds")  # Print the total execution time
        return None  # Indicate silence
    else:
        end_time = time.time()  # End time measurement
        execution_time = end_time - start_time
        print(f"Record Audio Execution Time - with audio: {execution_time:.2f} seconds")  # Print the total execution time
        return audio_data  # Return audio chunk


def transcribe_audio(model, audio_data):
    start_time = time.time()
    segments, _ = model.transcribe(audio_data, beam_size=3)  # beam_size was set to 3 to improve speed
    transcribe = ' '.join(segment.text for segment in segments)
    end_time = time.time()  # End time measurement
    execution_time = end_time - start_time
    print(f"Transcribe Audio Execution Time: {execution_time:.2f} seconds")  # Print the total execution time
    return transcribe


def detect_pause(start_time, pause_duration=0.5): # this will be checked after a silence to see if the silence continues. then the total pause time = 0.5 + DEFAULT_CHUNK_LENGTH
    """Check if silence has lasted long enough to trigger a stop."""
    return (time.time() - start_time) >= pause_duration


def main():
    model_size = DEFAULT_MODEL_SIZE + ".en"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    try:

        while True:
            audio_chunks = []
            print("_")
            start_silence_time = None

            while True:
                audio_data = record_audio_chunk(stream)

                if audio_data is not None:
                    # Start counting silence time
                    start_silence_time = None  # Reset silence timer
                    audio_chunks.append(audio_data)
                else:
                    if start_silence_time is None:
                        start_silence_time = time.time()  # Start silence timer

                    if detect_pause(start_silence_time):
                        print("Silence detected. Stopping recording...")
                        break  # Stop recording

            if len(audio_chunks) > 0:
                # Save all chunks together as one file before transcription
                full_audio_data = np.concatenate(audio_chunks)
                wavfile.write("full_audio.wav", 16000, full_audio_data)

                emotion = recognizer.predict_emotion("full_audio.wav")
                print(f"Predicted Emotion: {emotion}")

                # Transcribe the entire concatenated audio
                transcription = transcribe_audio(model, "full_audio.wav")

                # Reset for the next recording session
                audio_chunks = []
                if transcription.strip():
                    print(f"User: {transcription}")
                    response = ai_assistant.interact_with_llm(transcription, emotion)

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
