import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Dict, Optional
import logging

class CollaborativeRecommender:
    def __init__(self, n_neighbors: int = 5, metric: str = 'cosine'):
        """
        Initialize the collaborative filtering recommender system.
        
        Args:
            n_neighbors (int): Number of neighbors to use for kNN
            metric (str): Distance metric for kNN ('cosine' or 'euclidean')
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = NearestNeighbors(metric=metric, n_neighbors=n_neighbors)
        self.movie_data = None
        self.user_movie_matrix = None
        self.movie_id_map = None
        self.user_id_map = None
        self.reverse_movie_id_map = None
        self.reverse_user_id_map = None
        self._update_counter = 0
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        """
        Fit the recommender system with rating and movie data.
        
        Args:
            ratings_df (pd.DataFrame): DataFrame with columns [user_id, movie_id, rating]
            movies_df (pd.DataFrame): DataFrame with movie information
        """
        self.movie_data = movies_df
        
        # Create ID mappings
        unique_users = ratings_df['user_id'].unique()
        unique_movies = ratings_df['movie_id'].unique()
        
        self.user_id_map = {id: idx for idx, id in enumerate(unique_users)}
        self.movie_id_map = {id: idx for idx, id in enumerate(unique_movies)}
        
        # Create reverse mappings
        self.reverse_user_id_map = {idx: id for id, idx in self.user_id_map.items()}
        self.reverse_movie_id_map = {idx: id for id, idx in self.movie_id_map.items()}
        
        # Map IDs to matrix indices
        user_idx = ratings_df['user_id'].map(self.user_id_map)
        movie_idx = ratings_df['movie_id'].map(self.movie_id_map)
        
        # Create sparse matrix
        self.user_movie_matrix = csr_matrix(
            (ratings_df['rating'].values, (user_idx, movie_idx)),
            shape=(len(unique_users), len(unique_movies))
        )
        
        # Fit the kNN model
        self.model.fit(self.user_movie_matrix)
        self.logger.info(f"Model fitted with {len(unique_users)} users and {len(unique_movies)} movies")

    def _get_user_vector(self, user_ratings: Dict[int, float]) -> csr_matrix:
        """
        Convert user ratings dictionary to a sparse vector.
        
        Args:
            user_ratings (Dict[int, float]): Dictionary of {movie_id: rating}
        
        Returns:
            csr_matrix: Sparse user ratings vector
        """
        data = []
        indices = []
        
        for movie_id, rating in user_ratings.items():
            if movie_id in self.movie_id_map:
                data.append(rating)
                indices.append(self.movie_id_map[movie_id])
        
        return csr_matrix(
            (data, (np.zeros(len(data)), indices)),
            shape=(1, len(self.movie_id_map))
        )

    def recommend_movies(
        self, 
        user_ratings: Dict[int, float], 
        n_recommendations: int = 4,
    ) -> List[Dict]:
        """
        Generate movie recommendations for a user based on their ratings.
        """
        if not user_ratings:
            self.logger.warning("No user ratings provided")
            return []

        try:
            # Convert user ratings to sparse vector
            user_vector = self._get_user_vector(user_ratings)
            
            # Log number of ratings
            self.logger.info(f"User has provided {len(user_ratings)} ratings")

            # Find similar users
            distances, indices = self.model.kneighbors(user_vector)
            self.logger.info(f"Found {len(indices[0])} similar users")
            
            # Get similar users' ratings
            similar_users_matrix = self.user_movie_matrix[indices[0]]
            
            # Calculate weighted average ratings
            weights = 1 - distances[0]
            weighted_ratings = np.average(similar_users_matrix.toarray(), axis=0, weights=weights)
            
            # Create recommendations array
            movie_scores = []
            for idx, score in enumerate(weighted_ratings):
                real_movie_id = self.reverse_movie_id_map[idx]
                if real_movie_id not in user_ratings:
                    movie_scores.append((real_movie_id, score))
            
            # Sort and get top N recommendations
            movie_scores.sort(key=lambda x: x[1], reverse=True)
            top_recommendations = movie_scores[:n_recommendations]
            
            # Get movie details
            recommended_movies = []
            for movie_id, score in top_recommendations:
                movie_info = self.movie_data[self.movie_data['movie_id'] == movie_id].iloc[0]
                recommended_movies.append({
                    'movie_id': movie_id,
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'year': movie_info.get('year'),
                    'predicted_rating': round(float(score), 2),
                    'average_rating': movie_info.get('average_rating'),
                    'number_of_ratings': movie_info.get('number_of_ratings')
                })
            
            return recommended_movies
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return []

    def update_ratings(self, new_ratings: List[Dict[str, any]]) -> bool:
        """
        Update the model with new ratings.
        
        Args:
            new_ratings (List[Dict]): List of dictionaries containing user_id, movie_id, and rating
        
        Returns:
            bool: True if update was successful
        """
        try:
            for rating in new_ratings:
                user_id = rating['user_id']
                movie_id = rating['movie_id']
                rating_value = rating['rating']
                
                # Add new user/movie if needed
                if user_id not in self.user_id_map:
                    new_idx = len(self.user_id_map)
                    self.user_id_map[user_id] = new_idx
                    self.reverse_user_id_map[new_idx] = user_id
                    
                if movie_id not in self.movie_id_map:
                    new_idx = len(self.movie_id_map)
                    self.movie_id_map[movie_id] = new_idx
                    self.reverse_movie_id_map[new_idx] = movie_id
                
                # Update matrix
                user_idx = self.user_id_map[user_id]
                movie_idx = self.movie_id_map[movie_id]
                
                # Expand matrix if needed
                if user_idx >= self.user_movie_matrix.shape[0] or movie_idx >= self.user_movie_matrix.shape[1]:
                    new_shape = (
                        max(user_idx + 1, self.user_movie_matrix.shape[0]),
                        max(movie_idx + 1, self.user_movie_matrix.shape[1])
                    )
                    self.user_movie_matrix = self._resize_sparse_matrix(self.user_movie_matrix, new_shape)
                
                # Update rating
                self.user_movie_matrix[user_idx, movie_idx] = rating_value
            
            self._update_counter += 1
            if self._update_counter >= 10:  # Refit every 10 updates
                self.model.fit(self.user_movie_matrix)
                self._update_counter = 0
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating ratings: {str(e)}")
            return False

    def _resize_sparse_matrix(self, matrix: csr_matrix, new_shape: Tuple[int, int]) -> csr_matrix:
        """
        Resize a sparse matrix to a new shape.
        """
        if new_shape[0] < matrix.shape[0] or new_shape[1] < matrix.shape[1]:
            raise ValueError("New shape must be larger than current shape")
        
        # Create new empty matrix
        new_matrix = csr_matrix(new_shape)
        
        # Copy old data
        new_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
        
        return new_matrix