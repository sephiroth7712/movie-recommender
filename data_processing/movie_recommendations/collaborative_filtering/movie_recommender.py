import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import os
from sklearn.preprocessing import StandardScaler

class MovieRecommender:
    def __init__(self, n_neighbors=5):
        self.model = NearestNeighbors(metric='cosine', n_neighbors=n_neighbors, algorithm='brute')
        self.sparse_matrix = None
        self.movies_df = None
        self.user_mapper = None
        self.movie_mapper = None
        self.movie_ids = None
        
    def _create_mappings(self, ratings_df):
        """Create user and movie ID mappings to compressed indices"""
        unique_users = ratings_df['userId'].unique()
        unique_movies = ratings_df['movieId'].unique()
        
        self.user_mapper = {old: new for new, old in enumerate(unique_users)}
        self.movie_mapper = {old: new for new, old in enumerate(unique_movies)}
        self.reverse_movie_mapper = {new: old for old, new in self.movie_mapper.items()}
        
    def fit(self, ratings_df, movies_df, min_ratings=10):
        """
        Fit the recommender system with improved memory efficiency
        
        Parameters:
        - ratings_df: DataFrame with columns (userId, movieId, rating)
        - movies_df: DataFrame with movie information
        - min_ratings: Minimum number of ratings for a movie to be included
        """
        print("Processing data...")
        
        # Filter out movies with too few ratings
        movie_counts = ratings_df['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= min_ratings].index
        ratings_df = ratings_df[ratings_df['movieId'].isin(valid_movies)]
        
        # Create compressed mappings
        self._create_mappings(ratings_df)
        
        # Convert IDs to compressed indices
        rows = ratings_df['userId'].map(self.user_mapper)
        cols = ratings_df['movieId'].map(self.movie_mapper)
        
        # Create sparse matrix
        self.sparse_matrix = csr_matrix(
            (ratings_df['rating'], (rows, cols)),
            shape=(len(self.user_mapper), len(self.movie_mapper))
        )
        
        # Store movie information
        self.movies_df = movies_df
        self.movie_ids = list(self.movie_mapper.keys())
        
        # Fit the model
        print(f"Fitting model with matrix shape: {self.sparse_matrix.shape}")
        self.model.fit(self.sparse_matrix)
        print("Model fitted successfully!")
        
    def _create_user_vector(self, movie_ratings):
        """Create a sparse user vector from movie ratings"""
        vector = np.zeros(len(self.movie_mapper))
        for movie_id, rating in movie_ratings.items():
            if movie_id in self.movie_mapper:
                vector[self.movie_mapper[movie_id]] = rating
        return vector
    
    def get_movie_recommendations(self, movie_ratings, n_recommendations=5):
        """Generate recommendations based on movie ratings"""
        try:
            # Create user vector
            user_vector = self._create_user_vector(movie_ratings)
            
            # Find similar users
            distances, indices = self.model.kneighbors([user_vector])
            
            # Get recommendations using sparse matrix operations
            similar_users_matrix = self.sparse_matrix[indices[0]]
            weights = 1 - distances[0]
            weights = weights / weights.sum()  # Normalize weights
            
            # Calculate weighted average ratings
            weighted_ratings = similar_users_matrix.T.dot(weights)
            
            # Convert to series for easier processing
            predicted_ratings = pd.Series(
                weighted_ratings,
                index=[self.reverse_movie_mapper[i] for i in range(len(weighted_ratings))]
            )
            
            # Filter out already rated movies
            predicted_ratings = predicted_ratings[~predicted_ratings.index.isin(movie_ratings.keys())]
            
            # Get top recommendations
            top_movies = predicted_ratings.nlargest(n_recommendations)
            
            # Prepare results
            results = []
            for movie_id, pred_rating in top_movies.items():
                movie = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
                results.append({
                    'movieId': movie_id,
                    'title': movie['title'],
                    'genres': movie['genres'],
                    'predicted_rating': round(float(pred_rating), 2),
                    'year': movie.get('year'),
                    'average_rating': round(float(movie['average_rating']), 2) if 'average_rating' in movie else None
                })
            
            return results
            
        except Exception as e:
            print(f"Error in get_movie_recommendations: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []

def search_movies(movies_df, title_query):
    """Search for movies by title"""
    matches = movies_df[
        movies_df['title'].str.contains(title_query, case=False, na=False)
    ]
    return matches[['movieId', 'title', 'genres', 'year', 'average_rating']].head()
