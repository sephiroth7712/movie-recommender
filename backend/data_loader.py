import pandas as pd
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import numpy as np
from typing import List, Any
from database import engine
from models import Movie, Rating, User, Base


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


def validate_genres(genres: Any) -> List[str]:
    if isinstance(genres, str):
        return [g.strip() for g in genres.split("|") if g.strip()]
    return [] if genres is None else [str(g) for g in genres if g]


async def load_movies(session: AsyncSession, movies_df: pd.DataFrame):
    movies_df[["genres", "plot_summary"]] = movies_df[
        ["genres", "plot_summary"]
    ].fillna("")
    movies_data = movies_df.to_dict("records")
    total_movies = len(movies_df)
    for idx, row in enumerate(movies_data):
        movie = Movie(
            movie_id=int(row["movieId"]),
            title=str(row["primaryTitle"]),
            plot_summary=str(row.get("plot_summary", "")),
            release_year=str(row["year"]),
            runtime=float(row["runtime"]) if pd.notna(row.get("runtime")) else None,
            genres=validate_genres(row["genres"]),
            imdb_id=str(row.get("tconst", "")),
            tmdb_id=str(row.get("tmdbId", "")),
        )
        session.add(movie)

        if idx > 0 and idx % 1000 == 0:
            await session.commit()
            print(
                f"Processed {idx*100/total_movies:.2f}% - {idx}/{total_movies} movies"
            )

    await session.commit()
    print("Finished loading movies")
    return set(movies_df["movieId"].to_list())


async def load_ratings(session: AsyncSession, ratings_df: pd.DataFrame, movie_set: set):
    ratings_df.dropna(subset=["userId"], inplace=True)
    unique_users = ratings_df["userId"].unique()
    total_users = unique_users.size
    for idx, user_id in enumerate(unique_users):
        if user_id is not None:
            user = User(
                username=f"dataset_user_{user_id}",
                password="dummy_password",
                is_dataset_user=True,
            )
            session.add(user)
        if idx > 0 and idx % 100000 == 0:
            await session.commit()
            print(f"Processed {idx*100/total_users:.2f}% - {idx}/{total_users} users")

    await session.commit()
    print("Created dataset users")

    batch_size = 1000000
    total_ratings = len(ratings_df)

    for i in range(0, total_ratings, batch_size):
        batch = ratings_df.iloc[i : i + batch_size]
        for _, row in batch.iterrows():
            if row["movieId"] in movie_set and row["userId"] is not None:
                rating = Rating(
                    user_id=(row["userId"]),
                    movie_id=int(row["movieId"]),
                    rating=float(row["rating"]),
                    timestamp=datetime.fromtimestamp(row["timestamp"]),
                    is_dataset_rating=True,
                )
                session.add(rating)
        await session.commit()
        print(
            f"Processed {min(i + batch_size, total_ratings)*100/total_ratings:.2f}% - {min(i + batch_size, total_ratings)}/{total_ratings} ratings"
        )


async def main():
    movies_df = pd.read_csv(
        "../data_processing/datasets/processed_data/merged_movies_dataset.csv",
        dtype={
            "genres": "str",
            "plot_summary": "str",
            "movieId": "int",
            "primaryTitle": "str",
            "year": "str",
            "runtime": "float",
            "tconst": "str",
            "tmdbId": "str",
        },
    )
    ratings_df = pd.read_csv(
        "../data_processing/datasets/processed_data/processed_ratings.csv"
    )

    async with AsyncSession(engine) as session:
        print("Creating tables...")
        await create_tables()
        movie_set = await load_movies(session, movies_df)
        await load_ratings(session, ratings_df, movie_set)
        print("Data loading completed")


if __name__ == "__main__":
    asyncio.run(main())
