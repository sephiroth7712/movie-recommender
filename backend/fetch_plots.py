import asyncio
import aiohttp
from sqlalchemy import select, update, text
from datetime import datetime
import time
from tqdm import tqdm
from models import Movie
from database import async_session
import pandas as pd

async def get_movies_without_plot():
    async with async_session() as session:
        # Query for movies with empty plot_summary
        query = select(Movie.tmdb_id).where(Movie.plot_summary == "")
        result = await session.execute(query)
        # Get all IDs and filter out None and nan values
        ids = [row[0] for row in result.fetchall() if row[0] != "nan"]
        ids = [x for x in ids if not pd.isna(x)]  # Remove any nan values
        print("Sample TMDb IDs from database:", ids[:5] if ids else "No IDs found")
        return ids


async def update_movie_plot(tmdb_id, plot_summary):
    async with async_session() as session:
        try:
            query = (
                update(Movie)
                .where(Movie.tmdb_id == str(tmdb_id))
                .values(plot_summary=plot_summary)
            )
            await session.execute(query)
            await session.commit()
            return True
        except Exception as e:
            print(f"Error updating movie {tmdb_id}: {e}")
            return False


async def main():
    # Read the CSV file
    print("Reading CSV file...")
    df = pd.read_csv(
        "C:/VCU/CMSC591/Project/movie-recommender/data_processing/datasets/raw/tmdb/TMDB_movie_dataset_v11.csv"
    )

    # Convert id column to integer
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype(int)

    # Print sample IDs from CSV for debugging
    print("Sample IDs from CSV:", df["id"].head().tolist())
    print("CSV ID column dtype:", df["id"].dtype)

    # Get movies that need updates
    print("Getting movies without plot summaries...")
    movies_to_update = await get_movies_without_plot()
    print("Length of movies without plot:", len(movies_to_update))

    # Convert movies_to_update to same type as df['id']
    movies_to_update = [
        int(x) for x in movies_to_update if pd.notna(x) and str(x).strip()
    ]

    # Filter dataframe to only include movies we need to update
    df_to_update = df[df["id"].isin(movies_to_update)]
    print(f"Found {len(df_to_update)} movies to update")

    if len(df_to_update) == 0:
        print("\nDebug info:")
        print("First few values in movies_to_update:", movies_to_update[:5])
        print("First few values in df['id']:", df["id"].head().tolist())
        print("Value types:")
        print(
            "movies_to_update type:",
            type(movies_to_update[0]) if movies_to_update else "empty",
        )
        print("df['id'] type:", df["id"].dtype)
        return

    # Update movies with progress bar
    success_count = 0
    with tqdm(total=len(df_to_update), desc="Updating movie plots") as pbar:
        for _, row in df_to_update.iterrows():
            tmdb_id = int(row["id"])
            overview = row["overview"]

            if (
                pd.notna(overview) and overview.strip()
            ):  # Check if overview is not empty
                success = await update_movie_plot(tmdb_id, overview)
                if success:
                    success_count += 1

            pbar.update(1)

    print(
        f"\nUpdate complete! Successfully updated {success_count} out of {len(df_to_update)} movies"
    )


if __name__ == "__main__":
    asyncio.run(main())
