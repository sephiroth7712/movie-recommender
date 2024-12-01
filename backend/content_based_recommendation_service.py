from typing import List, Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from models import Movie, Rating
import sys
import os
import pandas as pd
from database import engine

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.movie_recommendations.content_based_filtering.cbf import (
    ContentBasedRecommender,
)

from fastapi import HTTPException


class ContentBasedRecommendationService:
    async def setup(self):
        async with AsyncSession(engine) as session:
            query = select(Movie)
            result = await session.execute(query)
            movies = result.scalars().all()
            movies_dict = [
                {
                    column.name: getattr(movie, column.name)
                    for column in Movie.__table__.columns
                }
                for movie in movies
            ]
            movies_df = pd.DataFrame(movies_dict)
            print(movies_df.head())
            self.recommender = ContentBasedRecommender(movies_df)

    async def search_movies(
        self, session: AsyncSession, search_term: str
    ) -> List[Dict]:
        """Search for movies by title"""
        query = (
            select(Movie).where(or_(Movie.title.ilike(f"%{search_term}%"))).limit(10)
        )

        result = await session.execute(query)
        movies = result.scalars().all()

        return [
            {
                "movie_id": movie.movie_id,
                "title": movie.title,
                "year": movie.release_year,
                "genres": movie.genres,
            }
            for movie in movies
        ]

    async def get_recommendations(
        self,
        session: AsyncSession,
        movie_id: int,
        n_recommendations: int = 4,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        genres: Optional[List[str]] = None,
    ) -> List[Dict]:
        try:
            recommended_movies = self.recommender.get_movie_recommendations(
                movie_id, n_recommendations, min_year, max_year, genres
            )

            recommended_movie_ids = [movie_id for movie_id, _ in recommended_movies]

            recommended_movies_query = select(Movie).where(
                Movie.movie_id.in_(recommended_movie_ids)
            )
            result = await session.execute(recommended_movies_query)
            recommended_movie_details = result.scalars().all()

            scores_dict = dict(recommended_movies)

            # Format final recommendations
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
                        "similarity_score": float(scores_dict[movie.movie_id]),
                    }
                )

            # Sort by aggregated similarity score
            final_recommendations.sort(
                key=lambda x: x["similarity_score"], reverse=True
            )
            return final_recommendations
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    async def get_user_recommendations(
        self,
        session: AsyncSession,
        user_id: int,
        n_recommendations: int = 10,
        min_rating: float = 3.5,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        genres: Optional[List[str]] = None,
    ) -> List[Dict]:
        try:
            rated_movies_query = (
                select(Rating, Movie)
                .join(Movie)
                .where(Rating.user_id == user_id, Rating.rating >= min_rating)
            )
            result = await session.execute(rated_movies_query)
            user_ratings = result.fetchall()

            if not user_ratings:
                raise ValueError(f"No rated movies found for user {user_id}")

            recommended_movies = self.recommender.get_user_recommendations(
                user_ratings,
                n_recommendations,
                min_rating,
                min_year,
                max_year,
                genres,
            )

            recommended_movie_ids = [movie_id for movie_id, _ in recommended_movies]

            recommended_movies_query = select(Movie).where(
                Movie.movie_id.in_(recommended_movie_ids)
            )
            result = await session.execute(recommended_movies_query)
            recommended_movie_details = result.scalars().all()

            scores_dict = dict(recommended_movies)

            # Format final recommendations
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
                        "similarity_score": float(scores_dict[movie.movie_id]),
                    }
                )

            # Sort by aggregated similarity score
            final_recommendations.sort(
                key=lambda x: x["similarity_score"], reverse=True
            )
            return final_recommendations
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
