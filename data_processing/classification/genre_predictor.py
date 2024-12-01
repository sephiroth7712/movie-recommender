from .enhanced_genre_classifier import EnhancedMovieGenreClassifier


class GenrePredictor:
    def __init__(self):
        self.emgc = EnhancedMovieGenreClassifier()
        self.cls = self.emgc.load_model(
            "C:\VCU\CMSC591\Project\movie-recommender\data_processing\classification\models\enhanced_genre_classifier_model.pkl"
        )

    def predict_genres(self, plot_summary: str):
        print(plot_summary)
        _, scores = self.cls.predict(plot_summary)
        return scores
