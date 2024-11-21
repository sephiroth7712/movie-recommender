import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import json
from typing import Dict, List, Tuple, Optional
import logging
import ast
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPathConfig:
    """Configuration class to store dataset paths"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

        # MovieLens paths
        self.movielens_path = self.base_path / "movielens"
        self.ratings_path = self.movielens_path / "ratings.csv"
        self.movies_path = self.movielens_path / "movies.csv"
        self.tags_path = self.movielens_path / "tags.csv"
        self.links_path = self.movielens_path / "links.csv"

        # CMU Movie Summary Corpus paths
        self.cmu_path = self.base_path / "cmu_summaries"
        self.plot_summaries_path = self.cmu_path / "plot_summaries.txt"
        self.movie_metadata_path = self.cmu_path / "movie.metadata.tsv"

        # IMDB paths
        self.imdb_path = self.base_path / "imdb"
        self.title_basics_path = self.imdb_path / "title.basics.tsv.gz"
        self.title_ratings_path = self.imdb_path / "title.ratings.tsv.gz"


class MovieLensLoader:
    """Class to load and preprocess MovieLens dataset"""

    def __init__(self, config: DataPathConfig):
        self.config = config

    def load_ratings(self) -> pd.DataFrame:
        """Load ratings data"""
        logger.info("Loading MovieLens ratings data...")
        ratings_df = pd.read_csv(self.config.ratings_path)

        # Calculate average rating and number of ratings per movie
        movie_stats = (
            ratings_df.groupby("movieId")
            .agg({"rating": ["count", "mean"]})
            .reset_index()
        )

        # Flatten column names
        movie_stats.columns = ["movieId", "number_of_ratings", "average_rating"]

        # Round average rating to 2 decimal places
        movie_stats["average_rating"] = movie_stats["average_rating"].round(2)

        return ratings_df, movie_stats

    def load_movies(self) -> pd.DataFrame:
        """Load movies data and process genres"""
        logger.info("Loading MovieLens movies data...")
        movies_df = pd.read_csv(self.config.movies_path)

        # Split genres into a list
        movies_df["genres"] = movies_df["genres"].str.split("|")

        # Extract year from title
        movies_df["year"] = (
            movies_df["title"].str.extract(r"\((\d{4})\)").astype("float")
        )
        movies_df["title"] = movies_df["title"].str.replace(
            r"\s*\(\d{4}\)", "", regex=True
        )

        return movies_df

    def load_links(self) -> pd.DataFrame:
        """Load links data"""
        logger.info("Loading MovieLens links data...")
        links_df = pd.read_csv(self.config.links_path)

        # Convert imdbId to string and add leading zeros
        links_df["imdbId"] = links_df["imdbId"].astype(str).str.zfill(7)

        return links_df


class CMUCorpusLoader:
    """Class to load and preprocess CMU Movie Summary Corpus"""

    def __init__(self, config: DataPathConfig):
        self.config = config

    def load_plot_summaries(self) -> pd.DataFrame:
        """Load plot summaries"""
        logger.info("Loading CMU plot summaries...")
        plot_data = []
        with open(self.config.plot_summaries_path, "r", encoding="utf-8") as f:
            for line in f:
                movie_id, plot = line.strip().split("\t", 1)
                plot_data.append(
                    {"wikipedia_movie_id": int(movie_id), "plot_summary": plot}
                )
        return pd.DataFrame(plot_data)

    def load_movie_metadata(self) -> pd.DataFrame:
        """Load and process movie metadata"""
        logger.info("Loading CMU movie metadata...")
        columns = [
            "wikipedia_movie_id",
            "freebase_movie_id",
            "movie_name",
            "release_date",
            "box_office_revenue",
            "runtime",
            "languages",
            "countries",
            "genres",
        ]

        metadata_df = pd.read_csv(
            self.config.movie_metadata_path, sep="\t", names=columns
        )

        # Convert wikipedia_movie_id to int
        metadata_df["wikipedia_movie_id"] = metadata_df["wikipedia_movie_id"].astype(
            int
        )

        # Process JSON-like strings in relevant columns
        metadata_df["genres"] = metadata_df["genres"].apply(self._parse_genres)
        metadata_df["languages"] = metadata_df["languages"].apply(
            self._parse_freebase_string
        )
        metadata_df["countries"] = metadata_df["countries"].apply(
            self._parse_freebase_string
        )

        return metadata_df

    @staticmethod
    def _parse_genres(genre_string: str) -> List[str]:
        """Parse the genre string into a list of genre names"""
        if pd.isna(genre_string) or genre_string == "":
            return []

        try:
            # Convert string representation of dictionary to actual dictionary
            genre_dict = ast.literal_eval(genre_string)
            # Extract only the genre names (values from the dictionary)
            return list(genre_dict.values())
        except:
            logger.warning(f"Failed to parse genre string: {genre_string}")
            return []

    @staticmethod
    def _parse_freebase_string(s: str) -> List[str]:
        """Parse other Freebase ID:name tuple strings"""
        if pd.isna(s) or s == "":
            return []
        try:
            # Extract just the names from the tuples
            return [item.split(":")[1] for item in s.split()]
        except:
            logger.warning(f"Failed to parse Freebase string: {s}")
            return []


class IMDbLoader:
    """Class to load and preprocess IMDb dataset"""

    def __init__(self, config: DataPathConfig):
        self.config = config

    def load_title_basics(self) -> pd.DataFrame:
        """Load and process title basics"""
        logger.info("Loading IMDb title basics...")
        df = pd.read_csv(
            self.config.title_basics_path, sep="\t", compression="gzip", na_values="\\N"
        )

        # Filter for movies only
        df = df[df["titleType"] == "movie"]

        # Convert genres to list
        df["genres"] = df["genres"].fillna("").str.split(",")

        # Clean up tconst
        df["tconst"] = df["tconst"].str.replace("tt", "")

        # Convert runtime to numeric, handling missing values
        df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")

        return df

    def load_title_ratings(self) -> pd.DataFrame:
        """Load title ratings"""
        logger.info("Loading IMDb title ratings...")
        df = pd.read_csv(
            self.config.title_ratings_path,
            sep="\t",
            compression="gzip",
            na_values="\\N",
        )

        # Clean up tconst
        df["tconst"] = df["tconst"].str.replace("tt", "")

        return df


class DatasetIntegrator:
    """Class to integrate data from different sources"""

    def __init__(
        self,
        movielens_loader: MovieLensLoader,
        cmu_loader: CMUCorpusLoader,
        imdb_loader: IMDbLoader,
    ):
        self.movielens_loader = movielens_loader
        self.cmu_loader = cmu_loader
        self.imdb_loader = imdb_loader

    def create_genre_classification_dataset(self) -> pd.DataFrame:
        """Create dataset for genre classification"""
        logger.info("Creating genre classification dataset...")

        try:
            plot_summaries = self.cmu_loader.load_plot_summaries()
            cmu_metadata = self.cmu_loader.load_movie_metadata()

            logger.info(f"Plot summaries shape: {plot_summaries.shape}")
            logger.info(f"Metadata shape: {cmu_metadata.shape}")

            # Merge plot summaries with metadata
            genre_df = plot_summaries.merge(
                cmu_metadata, on="wikipedia_movie_id", how="inner"
            )

            logger.info(f"Merged dataset shape: {genre_df.shape}")

            # Select relevant columns
            genre_df = genre_df[
                ["wikipedia_movie_id", "movie_name", "plot_summary", "genres"]
            ]

            # Add number of genres column
            genre_df["num_genres"] = genre_df["genres"].str.len()

            # Log sample of the data and genre statistics
            logger.info("\nSample of processed data:")
            logger.info(genre_df.head())

            # Log genre statistics
            all_genres = [genre for genres in genre_df["genres"] for genre in genres]
            unique_genres = sorted(set(all_genres))
            genre_counts = pd.Series(all_genres).value_counts()

            logger.info("\nUnique genres found:")
            logger.info(unique_genres)
            logger.info("\nTop 10 most common genres:")
            logger.info(genre_counts.head(10))

            return genre_df

        except Exception as e:
            logger.error(f"Error in create_genre_classification_dataset: {str(e)}")
            raise

    def create_recommendation_dataset(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create datasets for recommendation system"""
        logger.info("Creating recommendation dataset...")

        try:
            # Load MovieLens data
            ratings, movie_stats = self.movielens_loader.load_ratings()
            movies = self.movielens_loader.load_movies()
            links = self.movielens_loader.load_links()

            # Load IMDb data
            imdb_basics = self.imdb_loader.load_title_basics()
            imdb_ratings = self.imdb_loader.load_title_ratings()

            # Merge MovieLens movies with their stats
            movies_enriched = movies.merge(movie_stats, on="movieId", how="left")

            # Merge with links
            movies_enriched = movies_enriched.merge(links, on="movieId", how="left")

            # Merge with IMDb data
            movies_enriched = movies_enriched.merge(
                imdb_basics[["tconst", "runtimeMinutes", "genres", "primaryTitle"]],
                left_on="imdbId",
                right_on="tconst",
                how="left",
                suffixes=("_movielens", "_imdb"),
            ).merge(
                imdb_ratings[["tconst", "averageRating", "numVotes"]],
                on="tconst",
                how="left",
            )

            # Rename columns for clarity
            movies_enriched = movies_enriched.rename(
                columns={
                    "averageRating": "imdb_rating",
                    "numVotes": "imdb_votes",
                    "genres_movielens": "genres",
                    "genres_imdb": "imdb_genres",
                    "runtimeMinutes": "runtime",
                }
            )

            # Log some statistics
            logger.info(f"\nDataset statistics:")
            logger.info(f"Number of movies: {len(movies_enriched)}")
            logger.info(f"Number of ratings: {len(ratings)}")
            logger.info(
                f"Average ratings per movie: {len(ratings) / len(movies_enriched):.2f}"
            )
            logger.info("\nSample of processed movies:")
            logger.info(
                movies_enriched[
                    [
                        "title",
                        "year",
                        "average_rating",
                        "number_of_ratings",
                        "imdb_rating",
                        "imdb_votes",
                    ]
                ].head()
            )

            # Check for missing values
            missing_values = movies_enriched.isnull().sum()
            logger.info("\nMissing values in processed data:")
            logger.info(missing_values[missing_values > 0])

            return ratings, movies_enriched, links

        except Exception as e:
            logger.error(f"Error in create_recommendation_dataset: {str(e)}")
            raise


def main():
    """Example usage of the data loading classes"""
    try:
        # Initialize config with base path
        config = DataPathConfig(DATASETS_RAW)

        # Initialize loaders
        movielens_loader = MovieLensLoader(config)
        cmu_loader = CMUCorpusLoader(config)
        imdb_loader = IMDbLoader(config)

        # Initialize integrator
        integrator = DatasetIntegrator(movielens_loader, cmu_loader, imdb_loader)

        # Create datasets for recommendation system
        logger.info("Starting recommendation data processing...")
        ratings, movies_enriched, links = integrator.create_recommendation_dataset()

        # Save processed datasets
        output_path = Path(DATASETS_PROCESSED)
        output_path.mkdir(exist_ok=True)

        # Convert list columns to strings for CSV storage
        movies_enriched["genres"] = movies_enriched["genres"].apply(
            lambda x: "|".join(x) if isinstance(x, list) else x
        )
        movies_enriched["imdb_genres"] = movies_enriched["imdb_genres"].apply(
            lambda x: "|".join(x) if isinstance(x, list) else x
        )

        # Save files
        movies_enriched.to_csv(output_path / "processed_movies.csv", index=False)
        ratings.to_csv(output_path / "processed_ratings.csv", index=False)
        links.to_csv(output_path / "processed_links.csv", index=False)

        logger.info("Recommendation data processed and saved successfully")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
