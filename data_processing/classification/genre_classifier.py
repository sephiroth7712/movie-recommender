import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, f1_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle
from typing import List, Tuple, Dict
from config import *


class MovieGenreClassifier:
    def __init__(self, max_features: int = 5000):
        """
        Initialize the movie genre classifier with specified parameters.

        Args:
            max_features: Maximum number of features for TF-IDF vectorization
        """
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), stop_words="english"
        )
        self.mlb = MultiLabelBinarizer()
        self.svm = OneVsRestClassifier(
            LinearSVC(random_state=42), verbose=10, n_jobs=-1
        )
        self.rf = OneVsRestClassifier(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4),
            n_jobs=4,
            verbose=10,
        )
        self.lemmatizer = WordNetLemmatizer()
        # Download required NLTK data
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the text data by cleaning, tokenizing, and lemmatizing.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text string
        """
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]

        return " ".join(tokens)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare the data for training by preprocessing text and encoding genres.

        Args:
            df: DataFrame containing 'plot_summary' and 'genres' columns

        Returns:
            Tuple of features matrix and encoded genre labels
        """
        # Preprocess plot summaries
        X = df["plot_summary"].apply(self.preprocess_text)
        X = self.tfidf.fit_transform(X)

        # Convert genre strings to lists and encode
        y = df["genres"].apply(
            eval
        )  # Convert string representation of list to actual list
        y = self.mlb.fit_transform(y)

        return X, y

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the ensemble model on the provided data.

        Args:
            df: DataFrame containing training data

        Returns:
            Dictionary containing evaluation metrics
        """
        # Prepare data
        X, y = self.prepare_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train models
        self.svm.fit(X_train, y_train)
        self.rf.fit(X_train, y_train)

        # Make predictions
        svm_pred = self.svm.predict(X_test)
        rf_pred = self.rf.predict(X_test)

        # Combine predictions with weighted voting
        ensemble_pred = (0.6 * svm_pred + 0.4 * rf_pred > 0.5).astype(int)

        # Calculate metrics
        metrics = {
            "hamming_loss": hamming_loss(y_test, ensemble_pred),
            "micro_f1": f1_score(y_test, ensemble_pred, average="micro"),
            "macro_f1": f1_score(y_test, ensemble_pred, average="macro"),
        }

        return metrics

    def predict(self, plot_summary: str) -> List[str]:
        """
        Predict genres for a given plot summary.

        Args:
            plot_summary: Plot summary text

        Returns:
            List of predicted genres
        """
        # Preprocess the input text
        processed_text = self.preprocess_text(plot_summary)
        X = self.tfidf.transform([processed_text])

        # Make predictions with both models
        svm_pred = self.svm.predict(X)
        rf_pred = self.rf.predict(X)

        # Combine predictions with weighted voting
        ensemble_pred = (0.6 * svm_pred + 0.4 * rf_pred > 0.5).astype(int)

        # Convert binary predictions back to genre labels
        predicted_genres = self.mlb.inverse_transform(ensemble_pred)[0]

        return list(predicted_genres)

    def save_model(self, path: str):
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model
        """
        model_components = {
            "tfidf": self.tfidf,
            "mlb": self.mlb,
            "svm": self.svm,
            "rf": self.rf,
        }
        with open(path, "wb") as f:
            pickle.dump(model_components, f)

    @classmethod
    def load_model(cls, path: str) -> "MovieGenreClassifier":
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded MovieGenreClassifier instance
        """
        classifier = cls()
        with open(path, "rb") as f:
            model_components = pickle.load(f)

        classifier.tfidf = model_components["tfidf"]
        classifier.mlb = model_components["mlb"]
        classifier.svm = model_components["svm"]
        classifier.rf = model_components["rf"]

        return classifier


# Example usage
if __name__ == "__main__":
    # Read the dataset
    df = pd.read_csv(DATASETS_PROCESSED + "/aligned_genre_data.csv")

    # Initialize and train the classifier
    classifier = MovieGenreClassifier(max_features=5000)
    metrics = classifier.train(df)

    print("Training metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Save the model
    classifier.save_model("./models/genre_classifier_model.pkl")

    # Example prediction
    sample_plot = (
        "A police officer must save hostages from a building taken over by terrorists."
    )
    predicted_genres = classifier.predict(sample_plot)
    print(f"\nPredicted genres: {predicted_genres}")
