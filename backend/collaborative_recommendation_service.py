import pandas as pd
import numpy as np
from typing import Dict, List
import logging
import os
import sys
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models import Movie, Rating
from database import engine

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.movie_recommendations.collaborative_filtering.cobf import (
    CollaborativeRecommender,
)


class CollaborativeRecommendationService:
    def __init__(self):
        self.recommender = CollaborativeRecommender(n_neighbors=10, metric="cosine")
        self.logger = logging.getLogger(__name__)

    async def setup(self):
        async with AsyncSession(engine) as session:
            # Fetch movies from database
            query = select(Movie)
            result = await session.execute(query)
            movies = result.scalars().all()
            self.logger.info("Fetched movies")

            # Convert movies to DataFrame
            movies_dict = [
                {
                    column.name: getattr(movie, column.name)
                    for column in Movie.__table__.columns
                }
                for movie in movies
            ]
            movies_df = pd.DataFrame(movies_dict)

            # Load ratings from CSV and database
            ratings_df = pd.read_csv(
                "../data_processing/datasets/processed_data/processed_ratings.csv"
            )
            ratings_df = ratings_df.rename(
                columns={"movieId": "movie_id", "userId": "user_id"}
            )

            # Fetch additional ratings from database
            query = select(Rating)
            result = await session.execute(query)
            ratings = result.scalars().all()
            self.logger.info("Fetched ratings")

            # Convert database ratings to DataFrame
            ratings_dict = [
                {
                    column.name: getattr(rating, column.name)
                    for column in Rating.__table__.columns
                }
                for rating in ratings
            ]

            self.logger.info("Converting to dataframe")
            if len(ratings_dict) > 0:
                db_ratings_df = pd.DataFrame(ratings_dict)
                ratings_df = pd.concat([ratings_df, db_ratings_df], ignore_index=True)

            try:
                # Calculate movie statistics
                movie_stats = (
                    ratings_df.groupby("movie_id")
                    .agg({"rating": ["mean", "std", "count"]})
                    .reset_index()
                )

                movie_stats.columns = [
                    "movie_id",
                    "average_rating",
                    "rating_std",
                    "number_of_ratings",
                ]

                # Merge movie statistics
                self.movies_df = movies_df.merge(movie_stats, on="movie_id", how="left")

                # Fit the recommender with sparse matrix approach
                self.recommender.fit(ratings_df, self.movies_df)
                self.logger.info("Data prepared and model fitted successfully")

            except Exception as e:
                self.logger.error(f"Error preparing data: {str(e)}")
                raise

    def _calculate_confidence_score(
        self,
        predicted_rating: float,
        num_ratings: int,
        rating_variance: float,
        similarity_score: float,
    ) -> float:
        """Calculate confidence score for recommendations"""
        ratings_weight = 2 / (1 + np.exp(-num_ratings / 1000)) - 1
        variance_penalty = 1 / (1 + rating_variance)

        confidence = (
            0.4 * (predicted_rating / 5)
            + 0.3 * ratings_weight
            + 0.2 * variance_penalty
            + 0.1 * similarity_score
        )

        return round(min(max(confidence, 0), 1), 3)

    async def get_recommendations_for_user(
        self,
        session: AsyncSession,
        user_id: int,
        n_recommendations: int = 4,
    ) -> List[Dict]:
        """Get movie recommendations for a specific user"""
        try:
            # Get user's matrix index
            if user_id not in self.recommender.user_id_map:
                self.logger.warning(f"User {user_id} not found in the model")
                return []

            user_idx = self.recommender.user_id_map[user_id]

            # Get user's ratings from sparse matrix
            user_row = self.recommender.user_movie_matrix[user_idx]

            # Add logging to check user's ratings
            self.logger.info(f"User has rated {user_row.getnnz()} movies")

            rated_movie_indices = user_row.indices
            ratings = user_row.data

            # Convert to dictionary using reverse mapping
            user_ratings = {
                self.recommender.reverse_movie_id_map[idx]: rating
                for idx, rating in zip(rated_movie_indices, ratings)
            }

            if not user_ratings:
                self.logger.warning(f"No ratings found for user {user_id}")
                return []

            # Get recommendations using sparse implementation
            recommendations = self.recommender.recommend_movies(
                user_ratings,
                n_recommendations=n_recommendations,
            )

            self.logger.info(
                f"Got {len(recommendations)} recommendations before processing"
            )

            # Calculate genre similarity
            user_rated_genres = set()
            for movie_id in user_ratings.keys():
                movie_genres = self.movies_df[self.movies_df["movie_id"] == movie_id][
                    "genres"
                ].iloc[0]
                if isinstance(movie_genres, str):
                    user_rated_genres.update(movie_genres)

            # Add confidence scores
            for movie in recommendations:
                movie_genres = set(movie["genres"])

                genre_similarity = (
                    len(movie_genres.intersection(user_rated_genres))
                    / len(movie_genres)
                    if movie_genres
                    else 0
                )

                movie["confidence_score"] = self._calculate_confidence_score(
                    predicted_rating=movie["predicted_rating"],
                    num_ratings=movie["number_of_ratings"],
                    rating_variance=self.movies_df[
                        self.movies_df["movie_id"] == movie["movie_id"]
                    ]["rating_std"].iloc[0],
                    similarity_score=genre_similarity,
                )

                del movie["predicted_rating"]

            recommended_movie_ids = []
            scores_dict = dict()
            for movie in recommendations:
                recommended_movie_ids.append(int(movie["movie_id"]))
                scores_dict[int(movie["movie_id"])] = float(movie["confidence_score"])

            recommended_movies_query = select(Movie).where(
                Movie.movie_id.in_(recommended_movie_ids)
            )
            result = await session.execute(recommended_movies_query)
            recommended_movie_details = result.scalars().all()

            final_recommendations = []
            for movie in recommended_movie_details:
                final_recommendations.append(
                    {
                        "movie_id": movie.movie_id,
                        "title": movie.title,
                        "release_year": movie.release_year,
                        "genres": movie.genres,
                        "runtime": movie.runtime,
                        "plot_summary": movie.plot_summary,
                        "similarity_score": scores_dict[movie.movie_id],
                    }
                )

            final_recommendations.sort(
                key=lambda x: x["similarity_score"], reverse=True
            )

            self.logger.info(
                f"Returning {len(final_recommendations)} final recommendations"
            )
            return final_recommendations

        except Exception as e:
            self.logger.error(f"Error getting recommendations: {str(e)}")
            return []

    async def update_user_rating(
        self,
    ) -> bool:
        try:
            await self.setup()
        except Exception as e:
            self.logger.error(
                f"Error updating rating in recommendation service: {str(e)}"
            )
            return False
