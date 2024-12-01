import pandas as pd
import numpy as np
from config import *


def load_metadata(metadata_path):
    """
    Load movie metadata from the Wikipedia metadata TSV file.

    Parameters:
    metadata_path (str): Path to the metadata TSV file

    Returns:
    dict: Dictionary mapping wikipedia_movie_id to metadata
    """
    # Read only the required columns (ID and year)
    metadata_df = pd.read_csv(
        metadata_path,
        sep="\t",
        usecols=[0, 3, 5],
        names=["wikipedia_movie_id", "year", "runtime"],
    )

    def extract_year(date_str):
        if pd.isna(date_str):
            return 0
        try:
            # First try to parse as a full date
            if isinstance(date_str, str) and "-" in date_str:
                return int(date_str.split("-")[0])
            # If it's already a year (could be float or int)
            return int(float(date_str))
        except (ValueError, TypeError):
            return 0

    # Apply the year extraction to the year column
    metadata_df["year"] = metadata_df["year"].apply(extract_year)

    # Convert to dictionary, filtering out None values
    year_dict = dict(zip(metadata_df["wikipedia_movie_id"], metadata_df["year"]))
    runtime_dict = dict(zip(metadata_df["wikipedia_movie_id"], metadata_df["runtime"]))

    year_dict = {k: v for k, v in year_dict.items() if v is not None}
    runtime_dict = {k: v for k, v in runtime_dict.items() if v is not None}

    return year_dict, runtime_dict


def merge_datasets(aligned_path, processed_path, metadata_path):
    """
    Merge the aligned genre dataset with processed movies dataset and handle missing entries.

    Parameters:
    aligned_path (str): Path to aligned_genre_data.csv
    processed_path (str): Path to processed_movies_cleaned.csv
    metadata_path (str): Path to movie.metadata.tsv

    Returns:
    pd.DataFrame: Merged dataset
    """
    # Load datasets
    aligned_df = pd.read_csv(aligned_path)
    processed_df = pd.read_csv(processed_path)

    processed_df = processed_df.astype(
        {
            "year": "Int64",  # This is pandas nullable integer type, allows for None/NA values
            "number_of_ratings": "Int64",
            "tmdbId": "Int64",
            "tconst": "Int64",
            "imdb_votes": "Int64",
        }
    )

    # Load years from metadata
    wiki_years, wiki_runtimes = load_metadata(metadata_path)

    # Create mapping dictionaries for faster lookups
    processed_titles = set(
        processed_df["primaryTitle"].str.strip().str.lower()
        + processed_df["year"].astype(str)
    )

    # Initialize list to store new entries
    new_entries = []

    # Track existing movie IDs for generating new ones
    existing_ids = set(processed_df["movieId"])
    next_movie_id = max(existing_ids) + 1

    # Process aligned data entries
    for _, row in aligned_df.iterrows():
        movie_name = row["movie_name"].lower() + str(
            int(wiki_years.get(row["wikipedia_movie_id"], None))
        )
        if (row["movie_name"]) == "Sabrina":
            print(movie_name)

        if movie_name not in processed_titles:
            # Create new entry
            new_entry = {
                "movieId": next_movie_id,
                "title": row["movie_name"],
                "genres": row["genres_string"],
                "year": wiki_years.get(row["wikipedia_movie_id"], None),
                "number_of_ratings": None,
                "average_rating": None,
                "imdbId": None,
                "tmdbId": None,
                "tconst": None,
                "runtime": wiki_runtimes.get(row["wikipedia_movie_id"], None),
                "imdb_genres": None,
                "primaryTitle": row["movie_name"],
                "imdb_rating": None,
                "imdb_votes": None,
                "plot_summary": row["plot_summary"],
                "wikipedia_movie_id": row["wikipedia_movie_id"],
            }
            new_entries.append(new_entry)
            next_movie_id += 1

    # Create DataFrame from new entries
    new_entries_df = pd.DataFrame(new_entries)

    # Merge existing data
    processed_df["title_lower"] = processed_df["primaryTitle"].str.lower()
    aligned_df["title_lower"] = aligned_df["movie_name"].str.lower()

    # Merge based on lowercase titles
    merged_df = pd.merge(
        processed_df,
        aligned_df[["title_lower", "plot_summary", "wikipedia_movie_id"]],
        on="title_lower",
        how="left",
    )

    # Drop the temporary lowercase title column
    merged_df = merged_df.drop("title_lower", axis=1)

    # Combine with new entries
    if not new_entries_df.empty:
        final_df = pd.concat([merged_df, new_entries_df], ignore_index=True)
    else:
        final_df = merged_df

    final_df["title_year"] = final_df[
        "primaryTitle"
    ].str.strip().str.lower() + final_df["year"].astype(str)
    final_df = final_df.drop_duplicates(subset=["title_year"])
    final_df = final_df.drop("title_year", axis=1)

    final_df = final_df.astype(
        {
            "year": "Int64",  # This is pandas nullable integer type, allows for None/NA values
            "number_of_ratings": "Int64",
            "tmdbId": "Int64",
            "tconst": "Int64",
            "imdb_votes": "Int64",
            "wikipedia_movie_id": "Int64",
        }
    )

    return final_df


def save_merged_dataset(merged_df, output_path):
    """Save the merged dataset to CSV."""
    merged_df.to_csv(output_path, index=False)
    print(f"Merged dataset saved to {output_path}")

    # Print some statistics
    total_movies = len(merged_df)
    movies_with_plot = merged_df["plot_summary"].notna().sum()
    print(f"\nDataset Statistics:")
    print(f"Total movies: {total_movies}")
    print(f"Movies with plot summaries: {movies_with_plot}")
    print(f"Coverage: {(movies_with_plot/total_movies)*100:.2f}%")


# Usage example
if __name__ == "__main__":
    # Define file paths
    aligned_path = DATASETS_PROCESSED + "/aligned_genre_data.csv"
    processed_path = DATASETS_PROCESSED + "/processed_movies_cleaned.csv"
    metadata_path = DATASETS_RAW + "/cmu_summaries/movie.metadata.tsv"
    output_path = DATASETS_PROCESSED + "/merged_movies_dataset.csv"

    # Process and merge datasets
    merged_df = merge_datasets(aligned_path, processed_path, metadata_path)

    # Save results
    save_merged_dataset(merged_df, output_path)
