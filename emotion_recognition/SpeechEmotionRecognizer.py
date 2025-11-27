import torch
import librosa
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import numpy as np
import time


class SpeechEmotionRecognizer:
    def __init__(self):
        """
        Initialize the Speech Emotion Recognition model.
        :param model_path: Path to the trained model directory.
        """
        self.model_path = ("C:/Users/220425722/Desktop/Python/Emotion Recognition/Improved Models/s3prl "
                           "logs/Hubert/s3prl_hubert_class_bal_model/cls_bal_model_12/s3prl_hubert_class_balanced_12/")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HubertForSequenceClassification.from_pretrained(self.model_path).to(self.device)
        # self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_path).to(self.device)
        self.model.eval()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_path)
        self.emotion_labels = {0: "Anger", 1: "Happiness", 2: "Sadness", 3: "Neutral"}

        self.max_length = 32000

    def predict_emotion(self, file_path):
        """
        Predict the emotion from a given audio file.
        :param file_path: Path to the audio file (.wav format, mono, 16kHz).
        :return: Predicted emotion label.
        """
        start_time = time.time()
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
            predicted_class = torch.argmax(logits, dim=-1).item() # todo: check why 'Anger' is coming all the time. ex: do we need to preprocess audio doifferently
        end_time = time.time()
        print(f"Emotion Recogniser Time: {end_time - start_time:.2f} seconds")

        return self.emotion_labels.get(predicted_class, "Unknown")
