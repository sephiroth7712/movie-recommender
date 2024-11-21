import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from config import *


def analyze_genre_distribution(file_path: str) -> Tuple[Dict[str, int], pd.DataFrame]:
    """
    Analyze the distribution of genres in the dataset

    Args:
        file_path (str): Path to the CSV file

    Returns:
        Tuple containing:
        - Dict of genre counts
        - DataFrame with genre statistics
    """
    # Read the CSV file
    print("Reading data...")
    df = pd.read_csv(file_path)

    # Convert string representation of list to actual list
    genres_list = df["genres"].apply(ast.literal_eval)

    # Count all genres
    print("Counting genres...")
    genre_counter = Counter()
    for genres in genres_list:
        genre_counter.update(genres)

    # Create DataFrame with genre statistics
    stats_df = pd.DataFrame.from_dict(genre_counter, orient="index", columns=["count"])
    stats_df = stats_df.sort_values("count")

    # Calculate percentage
    total_movies = len(df)
    stats_df["percentage"] = (stats_df["count"] / total_movies * 100).round(2)

    return dict(genre_counter), stats_df


def print_genre_stats(stats_df: pd.DataFrame) -> None:
    """Print genre statistics"""
    print("\nGenre Distribution:")
    print("=" * 50)
    print(f"Total unique genres: {len(stats_df)}")
    print("\nTop 5 least common genres:")
    print("-" * 50)
    print(stats_df.head().to_string())
    print("\nTop 5 most common genres:")
    print("-" * 50)
    print(stats_df.tail().to_string())


def plot_genre_distribution(stats_df: pd.DataFrame, n_genres: int = 10) -> None:
    """
    Plot genre distribution

    Args:
        stats_df (pd.DataFrame): DataFrame with genre statistics
        n_genres (int): Number of genres to plot from each end
    """
    plt.figure(figsize=(15, 8))

    # Plot least common genres
    plt.subplot(1, 2, 1)
    stats_df.head(n_genres).plot(kind="barh", y="count")
    plt.title(f"{n_genres} Least Common Genres")
    plt.xlabel("Number of Movies")

    # Plot most common genres
    plt.subplot(1, 2, 2)
    stats_df.tail(n_genres).plot(kind="barh", y="count")
    plt.title(f"{n_genres} Most Common Genres")
    plt.xlabel("Number of Movies")

    plt.tight_layout()
    plt.show()


def get_genre_distribution():
    file_path = (
        DATASETS_PROCESSED
        + "/processed_genre_data.csv"  # Update this with your file path
    )

    try:
        # Analyze genre distribution
        genre_counts, stats_df = analyze_genre_distribution(file_path)

        # Print statistics
        print_genre_stats(stats_df)

        # Print detailed statistics for low-count genres
        threshold = 100  # Adjust this threshold as needed
        low_count_genres = stats_df[stats_df["count"] < threshold]

        print(f"\nGenres with less than {threshold} movies:")
        print("=" * 50)
        print(low_count_genres.to_string())

        # Plot distribution
        plot_genre_distribution(stats_df)

        # Save statistics to CSV
        output_file = DATASETS_PROCESSED + "/genre_distribution.csv"
        stats_df.to_csv(output_file)
        print(f"\nGenre distribution saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


def process_movie_genres(file_path):
    """
    Process movie genres from CSV file by merging genres and imdb_genres columns,
    removing rows with "(no genres listed)", and updating the genres column in place.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        tuple: (cleaned DataFrame with updated genres, list of unique genres)
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Remove rows with "(no genres listed)"
    df = df[df["genres"] != "(no genres listed)"]

    # Function to split genre string and handle NaN values
    def split_genres(x):
        if pd.isna(x):
            return set()
        return set(x.split("|"))

    # Convert genre strings to sets
    genres_sets = df["genres"].apply(split_genres)
    imdb_genres_sets = df["imdb_genres"].apply(split_genres)

    # Merge genres for each movie
    merged_genres = genres_sets.combine(imdb_genres_sets, lambda x, y: x.union(y))

    # Update the genres column in place with merged genres
    df["genres"] = merged_genres.apply(lambda x: "|".join(sorted(x)) if x else "")

    # Get all unique genres across both original columns
    all_genres = set()
    for genre_set in merged_genres:
        all_genres.update(genre_set)

    # Convert to sorted list
    unique_genres = sorted(list(all_genres))

    return df, unique_genres


def merge_movie_genres():
    file_path = DATASETS_PROCESSED + "/processed_movies.csv"

    # Process the file
    processed_df, unique_genres = process_movie_genres(file_path)

    # Display first few rows with updated genres
    print("Sample of updated genres:")
    print(processed_df[["title", "imdb_genres", "genres"]].head())

    print("\nAll unique genres found:")
    for genre in unique_genres:
        print(f"{genre}")

    # Save the updated DataFrame
    processed_df.to_csv(
        DATASETS_PROCESSED + "/processed_movies_cleaned.csv", index=False
    )


merge_movie_genres()
