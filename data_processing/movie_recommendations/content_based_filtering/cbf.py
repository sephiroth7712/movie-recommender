from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import re
from sqlalchemy import Row, Sequence
from backend.models import Movie, Rating


class ContentBasedRecommender:
    def __init__(self, movies_df: pd.DataFrame):
        """
        Initialize the memory-efficient content-based recommender system.

        Args:
            movies_df: DataFrame containing movie information
        """
        # Clean the DataFrame first
        self.movies_df = self._clean_dataframe(movies_df)

        self.tfidf = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(1, 2)
        )
        self.content_matrix = None

        # Prepare the content features
        self._prepare_content_features()

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input DataFrame and handle missing values
        """
        df = df.copy()

        # Handle missing values
        df["genres"] = df["genres"].fillna("")
        df["plot_summary"] = df["plot_summary"].fillna("")
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
        df["release_year"] = df["release_year"].fillna(0).astype(int)

        # Ensure genres is string type
        df["genres"] = df["genres"].astype(str)

        return df

    def _preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing
        """
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _create_content_string(self, row: pd.Series) -> str:
        """
        Create a combined content string from movie features with revised weights.
        Plot summary has increased weight, and genres are integrated into main recommendations.
        """
        # Process genres
        # genres = str(row['genres']).replace('|', ' ')
        genres = " ".join(row["genres"])

        # Process plot summary (increased weight)
        plot = str(row["plot_summary"])

        # Process year
        year = str(int(row["release_year"])) if pd.notna(row["release_year"]) else ""

        # Combine features with adjusted weights:
        # - Plot summary: 5x weight
        # - Genres: 2x weight
        # - Year: 1x weight
        content_parts = [
            genres * 2,  # Genre weight
            plot * 5,  # Increased plot weight
            year,  # Year weight
        ]

        return " ".join(content_parts)

    def _prepare_content_features(self):
        """
        Prepare the content features matrix
        """
        content_strings = self.movies_df.apply(self._create_content_string, axis=1)
        processed_contents = content_strings.apply(self._preprocess_text)
        self.content_matrix = self.tfidf.fit_transform(processed_contents)

    def _get_similar_movies(
        self,
        movie_idx: int,
        n_recommendations: int = 4,
        min_year: int = None,
        max_year: int = None,
        genres: List[str] = None,
    ) -> pd.DataFrame:
        """
        Get similar movies for a given movie index
        """
        # Get the movie's feature vector
        movie_vector = self.content_matrix[movie_idx : movie_idx + 1]

        # Calculate similarities in batches
        batch_size = 1000
        n_movies = self.content_matrix.shape[0]
        similarity_scores = []

        for i in range(0, n_movies, batch_size):
            end_idx = min(i + batch_size, n_movies)
            batch_similarities = cosine_similarity(
                movie_vector, self.content_matrix[i:end_idx]
            )[0]
            similarity_scores.extend(batch_similarities)

        # Create DataFrame with results
        recommendations_df = pd.DataFrame(
            {"idx": range(len(similarity_scores)), "similarity": similarity_scores}
        )

        # Merge with movie information
        recommendations_df = recommendations_df.merge(
            self.movies_df.reset_index(), left_on="idx", right_on="index"
        )

        # Apply filters
        if min_year is not None and min_year > 0:
            recommendations_df = recommendations_df[
                recommendations_df["release_year"] >= min_year
            ]
        if max_year is not None:
            recommendations_df = recommendations_df[
                recommendations_df["release_year"] <= max_year
            ]
        if genres:
            recommendations_df = recommendations_df[
                recommendations_df["genres"].apply(
                    lambda x: any(genre.lower() in x.lower() for genre in genres)
                )
            ]

        return recommendations_df

    def get_movie_recommendations(
        self,
        movie_id: int,
        n_recommendations: int = 4,
        min_year: int = None,
        max_year: int = None,
        genres: List[str] = None,
        exclude_movie_ids: List[int] = None,
    ) -> List[Dict[str, any]]:
        """
        Get movie recommendations based on a given movie ID
        """
        try:
            # Get movie index
            movie_idx = self.movies_df[self.movies_df["movie_id"] == movie_id].index
            if len(movie_idx) == 0:
                raise ValueError(f"Movie ID {movie_id} not found in the dataset")
            movie_idx = movie_idx[0]

            # Get similar movies
            recommendations_df = self._get_similar_movies(
                movie_idx, n_recommendations, min_year, max_year, genres
            )

            # Remove the input movie and get top N
            recommendations_df = recommendations_df[
                recommendations_df["movie_id"] != movie_id
            ]

            if exclude_movie_ids is not None:
                recommendations_df = recommendations_df[
                    ~recommendations_df["movie_id"].isin(exclude_movie_ids)
                ]

            recommendations_df = recommendations_df.nlargest(
                n_recommendations, "similarity"
            )

            # Format results
            recommendations = []
            for _, row in recommendations_df.iterrows():
                recommendations.append(
                    {
                        "movie_id": row["movie_id"],
                        "title": row["title"],
                        "release_year": int(row["release_year"]),
                        "genres": row["genres"],
                        "similarity_score": float(row["similarity"]),
                    }
                )

            return recommendations

        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return []

    def get_user_recommendations(
        self,
        user_ratings: Sequence[Row[Tuple[Rating, Movie]]],
        n_recommendations: int = 4,
        min_rating: float = 3.5,
        min_year: int = None,
        max_year: int = None,
        genres: List[str] = None,
    ) -> List[Tuple]:
        """Get recommendations based on all movies a user has liked"""

        # Get recommendations from each rated movie
        all_recommendations = defaultdict(float)

        # Get set of movies user has already rated
        rated_movie_ids = {rating.Rating.movie_id for rating in user_ratings}

        for rating, movie in user_ratings:
            # Weight by user's rating
            rating_weight = (rating.rating - min_rating + 1) / (5 - min_rating + 1)

            # Get recommendations for this movie
            movie_recs = self.get_movie_recommendations(
                movie_id=movie.movie_id,
                n_recommendations=n_recommendations,
                min_year=min_year,
                max_year=max_year,
                genres=genres,
                exclude_movie_ids=rated_movie_ids,
            )

            # Aggregate recommendations with rating weight
            for rec in movie_recs:
                all_recommendations[rec["movie_id"]] += (
                    rec["similarity_score"] * rating_weight
                )

        # Get full details for recommended movies
        recommended_movie_ids = sorted(
            all_recommendations.items(),
            key=lambda x: all_recommendations[x],
            reverse=True,
        )[:n_recommendations]

        if not recommended_movie_ids:
            return []
        return recommended_movie_ids


# # Example usage
# if __name__ == "__main__":
#     try:
#         # Load movies DataFrame
#         movies_df = pd.read_csv("F:\Data Science Project\movie-recommender\data_processing\datasets\datasets\processed_data\merged_movies_dataset.csv")
#         print("Movie Recommendation System")
#         print("-" * 30)

#         # Initialize recommender
#         recommender = ContentBasedRecommender(movies_df)

#         # Get movie title from user
#         movie_title = input("\nEnter a movie title: ")

#         # Find the movie in the dataset
#         movie_mask = movies_df['title'].str.lower() == movie_title.lower()
#         if not any(movie_mask):
#             print(f"\nMovie '{movie_title}' not found!")
#             # Show similar titles
#             similar_titles = movies_df[movies_df['title'].str.lower().str.contains(movie_title.lower())]
#             if not similar_titles.empty:
#                 print("\nDid you mean one of these?")
#                 for title in similar_titles['title'].head():
#                     print(f"- {title}")
#             exit()

#         # Get the movie ID and details
#         sample_movie_id = movies_df[movie_mask]['movieId'].iloc[0]
#         sample_movie = movies_df[movie_mask].iloc[0]
#         print(f"\nFinding movies similar to: {sample_movie['title']} ({sample_movie['year']})")
#         print(f"Genres: {sample_movie['genres']}")
#         print("-" * 30)

#         # Get recommendations
#         recommendations = recommender.get_movie_recommendations(
#             movie_id=sample_movie_id,
#             n_recommendations=4,
#             min_year=None,
#             genres=None
#         )

#         # Print recommendations
#         print("\nRecommended Movies:")
#         print("-" * 30)
#         for rec in recommendations:
#             print(f"{rec['title']} ({rec['year']})")
#             print(f"Genres: {rec['genres']}")
#             print(f"Similarity: {rec['similarity_score']:.3f}")
#             print()

#     except Exception as e:
#         print(f"Error: {str(e)}")
