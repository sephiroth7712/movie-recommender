from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, f1_score, precision_recall_curve
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import re
import pickle
from typing import List, Tuple, Dict
from sklearn.calibration import CalibratedClassifierCV
from .config import *


class EnhancedMovieGenreClassifier:
    def __init__(self, max_features: int = 5000):
        """
        Initialize the enhanced movie genre classifier.
        """
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), stop_words="english"
        )
        self.mlb = MultiLabelBinarizer()

        # Use calibrated SVM for probability estimates
        base_svm = LinearSVC(random_state=42)
        self.svm = OneVsRestClassifier(
            CalibratedClassifierCV(base_svm), n_jobs=-1, verbose=10
        )

        self.rf = OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced", n_jobs=4
            ),
            n_jobs=4,
            verbose=10,
        )

        self.lemmatizer = WordNetLemmatizer()
        self.thresholds = None
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

        # Download required NLTK data
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")

    def _preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Process a batch of texts using SpaCy's pipe
        """
        processed = []
        for doc in self.nlp.pipe(texts, batch_size=100):
            tokens = []
            for token in doc:
                if not token.ent_type_ and token.pos_ not in ["PROPN", "PRON"]:
                    tokens.append(token.lemma_)
            text = " ".join(tokens)

            # Tokenize and lemmatize
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

            # Remove stopwords
            stop_words = set(stopwords.words("english"))
            tokens = [token for token in tokens if token not in stop_words]

            processed.append(" ".join(tokens))
        return processed

    def _parallel_batch_process(
        self, content_strings: pd.Series, n_jobs: int = None, batch_size: int = 1000
    ) -> List[str]:
        """
        Process text using both batching and parallel processing
        """
        # Convert series to list and clean the texts first
        texts = [
            re.sub(
                r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
            ).strip()
            for text in content_strings
        ]

        # Split into batches
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(self._preprocess_batch, batches))

        # Flatten results
        return [item for sublist in results for item in sublist]

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the text data with enhanced cleaning.
        """
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        tokens = []
        doc = self.nlp(text)

        for token in doc:
            if not token.ent_type_ and token.pos_ not in ["PROPN", "PRON"]:
                tokens.append(token.lemma_)

        text = " ".join(tokens)

        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]

        return " ".join(tokens)

    def optimize_thresholds(self, X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        """
        Find optimal decision thresholds for each genre using F1 score optimization.
        """
        # Get prediction probabilities
        svm_probs = self.svm.predict_proba(X_val)
        rf_probs = self.rf.predict_proba(X_val)

        # Combine predictions with ensemble weights
        ensemble_probs = 0.6 * svm_probs + 0.4 * rf_probs

        optimal_thresholds = []
        f1_scores = []

        # Optimize threshold for each genre
        for i in range(y_val.shape[1]):
            precision, recall, thresholds = precision_recall_curve(
                y_val[:, i], ensemble_probs[:, i]
            )

            # Add an extra threshold point for precision_recall_curve
            thresholds = np.append(thresholds, 1.0)

            # Calculate F1 scores for each threshold
            f1_scores_i = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores_i)

            optimal_thresholds.append(thresholds[optimal_idx])
            f1_scores.append(f1_scores_i[optimal_idx])

        return np.array(optimal_thresholds)

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the enhanced classifier with threshold optimization.
        """
        # Prepare data
        X = self._parallel_batch_process(df["plot_summary"], n_jobs=8, batch_size=1000)
        X = self.tfidf.fit_transform(X)

        # Convert genre strings to lists and encode
        y = df["genres"].apply(eval)
        y = self.mlb.fit_transform(y)

        # Split data into train, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42
        )

        # Train models
        print("Training SVM classifier...")
        self.svm.fit(X_train, y_train)

        print("Training Random Forest classifier...")
        self.rf.fit(X_train, y_train)

        # Optimize thresholds on validation set
        print("Optimizing decision thresholds...")
        self.thresholds = self.optimize_thresholds(X_val, y_val)

        # Make predictions on test set
        svm_probs = self.svm.predict_proba(X_test)
        rf_probs = self.rf.predict_proba(X_test)
        ensemble_probs = 0.6 * svm_probs + 0.4 * rf_probs

        # Apply optimized thresholds
        ensemble_pred = (ensemble_probs > self.thresholds).astype(int)

        # Calculate metrics
        metrics = {
            "hamming_loss": hamming_loss(y_test, ensemble_pred),
            "micro_f1": f1_score(y_test, ensemble_pred, average="micro"),
            "macro_f1": f1_score(y_test, ensemble_pred, average="macro"),
        }

        # Add per-genre F1 scores
        genre_names = self.mlb.classes_
        genre_f1_scores = f1_score(y_test, ensemble_pred, average=None)
        for genre, f1 in zip(genre_names, genre_f1_scores):
            metrics[f"f1_{genre}"] = f1

        return metrics

    def predict(self, plot_summary: str) -> Tuple[List[str], Dict[str, float]]:
        """
        Predict genres for a given plot summary with confidence scores.
        """
        # Preprocess the input text
        processed_text = self.preprocess_text(plot_summary)
        X = self.tfidf.transform([processed_text])

        # Get probability predictions
        svm_probs = self.svm.predict_proba(X)
        rf_probs = self.rf.predict_proba(X)
        ensemble_probs = 0.6 * svm_probs + 0.4 * rf_probs

        # Apply optimized thresholds
        ensemble_pred = (ensemble_probs > self.thresholds).astype(int)

        # Convert predictions to genre labels
        predicted_genres = self.mlb.inverse_transform(ensemble_pred)[0]

        # Calculate confidence scores
        genre_names = self.mlb.classes_
        confidence_scores = {
            genre: float(ensemble_probs[0, i])
            for i, genre in enumerate(genre_names)
            if ensemble_pred[0, i] == 1
        }

        return list(predicted_genres), confidence_scores

    def save_model(self, path: str):
        """Save the trained model to disk."""
        model_components = {
            "tfidf": self.tfidf,
            "mlb": self.mlb,
            "svm": self.svm,
            "rf": self.rf,
            "thresholds": self.thresholds,
        }
        with open(path, "wb") as f:
            pickle.dump(model_components, f)

    @classmethod
    def load_model(cls, path: str) -> "EnhancedMovieGenreClassifier":
        """Load a trained model from disk."""
        classifier = cls()
        with open(path, "rb") as f:
            model_components = pickle.load(f)

        classifier.tfidf = model_components["tfidf"]
        classifier.mlb = model_components["mlb"]
        classifier.svm = model_components["svm"]
        classifier.rf = model_components["rf"]
        classifier.thresholds = model_components["thresholds"]

        return classifier


# Example usage
if __name__ == "__main__":
    # Read the dataset
    df = pd.read_csv(DATASETS_PROCESSED + "/aligned_genre_data.csv")

    # Initialize and train the classifier
    classifier = EnhancedMovieGenreClassifier(max_features=5000)
    metrics = classifier.train(df)

    print("\nTraining metrics:")
    for metric_name, value in metrics.items():
        if not metric_name.startswith("f1_"):
            print(f"{metric_name}: {value:.4f}")

    print("\nPer-genre F1 scores:")
    for metric_name, value in metrics.items():
        if metric_name.startswith("f1_"):
            genre = metric_name[3:]
            print(f"{genre}: {value:.4f}")

    # Save the model
    classifier.save_model("./models/enhanced_genre_classifier_model.pkl")

    # Example prediction
    sample_plot = (
        "A police officer must save hostages from a building taken over by terrorists."
    )
    predicted_genres, confidence_scores = classifier.predict(sample_plot)
    print("\nPredicted genres with confidence scores:")
    for genre in predicted_genres:
        print(f"{genre}: {confidence_scores[genre]:.4f}")
