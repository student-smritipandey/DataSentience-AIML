from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

class CryptoSentimentAnalyzer:
    def __init__(self):
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def predict_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        scores = softmax(outputs.logits[0].numpy())
        labels = ["negative", "neutral", "positive"]
        return dict(zip(labels, scores))
