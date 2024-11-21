import pandas as pd
import ast
from collections import Counter
from config import *


def load_genre_mappings(mapping_file):
    """Load and process genre mappings from the mapping file."""
    genre_mappings = {}
    with open(mapping_file, "r", encoding="utf-8") as f:  # Added UTF-8 encoding
        for line in f:
            line = line.strip()
            if "->" in line:
                source, target = line.split("->")
                source = source.strip().lower()
                # Handle cases with multiple target genres (separated by '+')
                targets = [t.strip() for t in target.split("+")]
                genre_mappings[source] = targets
    return genre_mappings


def load_and_process_data(mapping_file):
    # Load dataset
    genre_df = pd.read_csv(
        DATASETS_PROCESSED + "/processed_genre_data.csv", encoding="utf-8"
    )  # Added UTF-8 encoding

    # Load custom genre mappings
    genre_mapping = load_genre_mappings(mapping_file)

    # Convert string representation of list to actual list for genre_df
    genre_df["genres"] = genre_df["genres"].apply(ast.literal_eval)
    # Convert all genres to lowercase and normalize special characters
    genre_df["genres"] = genre_df["genres"].apply(
        lambda x: [g.lower().strip() for g in x]
    )

    # Print all unique genres before mapping (for debugging)
    # print("All unique genres before mapping:")
    # unique_genres = set()
    # for genres in genre_df["genres"]:
    #     unique_genres.update(genres)
    # for genre in sorted(unique_genres):
    #     print(f"'{genre}'")

    # Count original genres distribution
    all_genres = []
    for genres in genre_df["genres"]:
        all_genres.extend(genres)
    original_genre_dist = Counter(all_genres)

    print("\nOriginal genre distribution:")
    for genre, count in original_genre_dist.most_common():
        print(f"{genre}: {count}")

    def find_matching_genre(genre):
        # Convert input genre to lowercase and strip
        genre = genre.lower().strip()

        # Debug print for problematic genre
        # if "clef" in genre:
        # print(f"Processing genre: '{genre}'")
        # print(f"Available mappings: {genre_mapping}")

        # Check if genre exists in our mapping
        if genre in genre_mapping:
            return genre_mapping[genre]

        # If no mapping exists, return the original genre
        return [genre]

    # Filter and map genres
    def process_genres(genre_list):
        processed = []
        for genre in genre_list:
            matched_genres = find_matching_genre(genre)
            if matched_genres:
                processed.extend(matched_genres)
        return list(set(processed))  # Remove duplicates

    # Apply filtering and update related columns
    genre_df["filtered_genres"] = genre_df["genres"].apply(process_genres)
    genre_df["num_genres"] = genre_df["filtered_genres"].apply(len)
    genre_df["genres_string"] = genre_df["filtered_genres"].apply(lambda x: "|".join(x))

    # Count new genres distribution
    all_filtered_genres = []
    for genres in genre_df["filtered_genres"]:
        all_filtered_genres.extend(genres)
    new_genre_dist = Counter(all_filtered_genres)

    print("\nNew genre distribution after mapping:")
    for genre, count in new_genre_dist.most_common():
        print(f"{genre}: {count}")

    # Print summary of applied mappings
    # print("\nApplied genre mappings:")
    # for old_genre in original_genre_dist:
    #     if old_genre in genre_mapping:
    #         print(f"{old_genre} -> {genre_mapping[old_genre]}")

    # Print summary statistics
    print(f"\nOriginal number of movies: {len(genre_df)}")
    print(f"Original number of unique genres: {len(original_genre_dist)}")
    print(f"Number of movies after mapping: {len(genre_df)}")
    print(f"Number of unique genres after mapping: {len(new_genre_dist)}")

    # Save processed dataset
    output_columns = [
        "wikipedia_movie_id",
        "movie_name",
        "plot_summary",
        "filtered_genres",
        "num_genres",
        "genres_string",
    ]
    genre_df[output_columns].to_csv(
        DATASETS_BASE + "/aligned_genre_data.csv",
        index=False,
        encoding="utf-8",  # Added UTF-8 encoding
    )

    return genre_df, original_genre_dist, new_genre_dist


if __name__ == "__main__":
    processed_df, original_dist, new_dist = load_and_process_data(
        mapping_file=DATASETS_PROCESSED + "/genre_mappings.txt"
    )
