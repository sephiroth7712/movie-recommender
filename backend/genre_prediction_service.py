import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.classification.genre_predictor import (
    GenrePredictor,
)


class GenrePredictionService:
    def __init__(self):
        self.classifier = GenrePredictor()

    def predict_genres(self, plot_summary: str):
        genres = self.classifier.predict_genres(plot_summary)
        return genres
