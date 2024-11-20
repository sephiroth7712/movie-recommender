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
import re
import pickle
from typing import List, Tuple, Dict
from sklearn.calibration import CalibratedClassifierCV
from config import *

class EnhancedMovieGenreClassifier:
    def __init__(self, max_features: int = 5000):
        """
        Initialize the enhanced movie genre classifier.
        """
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.mlb = MultiLabelBinarizer()
        
        # Use calibrated SVM for probability estimates
        base_svm = LinearSVC(random_state=42)
        self.svm = OneVsRestClassifier(CalibratedClassifierCV(base_svm))
        
        self.rf = OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        )
        
        self.lemmatizer = WordNetLemmatizer()
        self.thresholds = None
        
        # Define genre relationships
        self.genre_relationships = {
            'romance film': ['romance'],
            'comedy film': ['comedy'],
            'romantic drama': ['romance', 'drama'],
            'psychological thriller': ['thriller'],
            'film-noir': ['drama', 'crime'],
            'biographical': ['drama'],
            'family film': ['children'],
            'science fiction': ['fantasy'],
            'bollywood': ['world cinema'],
            'japanese movies': ['world cinema']
        }
        
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the text data with enhanced cleaning.
        """
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    
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
    
    def post_process_predictions(self, predictions: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """
        Adjust predictions based on genre relationships and confidence scores.
        """
        genre_names = self.mlb.classes_
        
        for i in range(len(predictions)):
            pred = predictions[i]
            probs = probabilities[i]
            
            # Apply genre relationships
            for main_genre, related_genres in self.genre_relationships.items():
                main_idx = np.where(genre_names == main_genre)[0]
                if len(main_idx) == 0:  # Skip if genre not in training set
                    continue
                main_idx = main_idx[0]
                
                related_idx = []
                for g in related_genres:
                    idx = np.where(genre_names == g)[0]
                    if len(idx) > 0:
                        related_idx.append(idx[0])
                
                # If main genre is predicted with high confidence, include related genres
                if pred[main_idx] and probs[main_idx] > 0.7:
                    for idx in related_idx:
                        pred[idx] = 1
                
                # If all related genres are predicted with good confidence, include main genre
                if related_idx and all(pred[idx] and probs[idx] > 0.6 for idx in related_idx):
                    pred[main_idx] = 1
        
        return predictions
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the enhanced classifier with threshold optimization.
        """
        # Prepare data
        X = df['plot_summary'].apply(self.preprocess_text)
        X = self.tfidf.fit_transform(X)
        
        # Convert genre strings to lists and encode
        y = df['genres'].apply(eval)
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
        
        # Post-process predictions
        ensemble_pred = self.post_process_predictions(ensemble_pred, ensemble_probs)
        
        # Calculate metrics
        metrics = {
            'hamming_loss': hamming_loss(y_test, ensemble_pred),
            'micro_f1': f1_score(y_test, ensemble_pred, average='micro'),
            'macro_f1': f1_score(y_test, ensemble_pred, average='macro')
        }
        
        # Add per-genre F1 scores
        genre_names = self.mlb.classes_
        genre_f1_scores = f1_score(y_test, ensemble_pred, average=None)
        for genre, f1 in zip(genre_names, genre_f1_scores):
            metrics[f'f1_{genre}'] = f1
        
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
        
        # Post-process predictions
        ensemble_pred = self.post_process_predictions(ensemble_pred, ensemble_probs)
        
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
            'tfidf': self.tfidf,
            'mlb': self.mlb,
            'svm': self.svm,
            'rf': self.rf,
            'thresholds': self.thresholds,
            'genre_relationships': self.genre_relationships
        }
        with open(path, 'wb') as f:
            pickle.dump(model_components, f)
    
    @classmethod
    def load_model(cls, path: str) -> 'EnhancedMovieGenreClassifier':
        """Load a trained model from disk."""
        classifier = cls()
        with open(path, 'rb') as f:
            model_components = pickle.load(f)
        
        classifier.tfidf = model_components['tfidf']
        classifier.mlb = model_components['mlb']
        classifier.svm = model_components['svm']
        classifier.rf = model_components['rf']
        classifier.thresholds = model_components['thresholds']
        classifier.genre_relationships = model_components['genre_relationships']
        
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
        if not metric_name.startswith('f1_'):
            print(f"{metric_name}: {value:.4f}")
    
    print("\nPer-genre F1 scores:")
    for metric_name, value in metrics.items():
        if metric_name.startswith('f1_'):
            genre = metric_name[3:]
            print(f"{genre}: {value:.4f}")
    
    # Save the model
    classifier.save_model("./models/enhanced_genre_classifier_model.pkl")
    
    # Example prediction
    sample_plot = "A police officer must save hostages from a building taken over by terrorists."
    predicted_genres, confidence_scores = classifier.predict(sample_plot)
    print("\nPredicted genres with confidence scores:")
    for genre in predicted_genres:
        print(f"{genre}: {confidence_scores[genre]:.4f}")