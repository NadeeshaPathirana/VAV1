import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import numpy as np


class SpeechEmotionRecognizer:
    def __init__(self, model_path):
        """
        Initialize the Speech Emotion Recognition model.
        :param model_path: Path to the trained model directory.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HubertForSequenceClassification.from_pretrained(model_path).to(self.device)
        # self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.emotion_labels = {0: 'happy', 1: 'sad', 2: 'angry'}
        # self.emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'ps', 6: 'sad'}

        self.max_length = 32000

    def predict_emotion(self, file_path):
        """
        Predict the emotion from a given audio file.
        :param file_path: Path to the audio file (.wav format, mono, 16kHz).
        :return: Predicted emotion label.
        """
        # Load and preprocess audio
        speech, sr = librosa.load(file_path, sr=16000)  # Ensure 16kHz sampling rate

        # pad or truncate the speech to the required length
        if len(speech) > self.max_length:
            speech = speech[:self.max_length]
        else:
            speech = np.pad(speech, (0, self.max_length - len(speech)), 'constant')
        # preprocess the audio file
        inputs = self.feature_extractor(speech, sampling_rate=16000, return_tensors='pt', padding=True, trucate=True,
                                        max_length=self.max_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            logits = self.model(**inputs).logits
            predicted_class = torch.argmax(logits, dim=-1).item()

        return self.emotion_labels.get(predicted_class, "Unknown")
