import pandas as pd
import ast
from collections import Counter
from difflib import get_close_matches
from config import *


def load_and_process_data(min_frequency=1000):
    # Load both datasets
    genre_df = pd.read_csv(DATASETS_PROCESSED + "/processed_genre_data.csv")
    movies_df = pd.read_csv(DATASETS_PROCESSED + "/processed_movies.csv")

    # Convert string representation of list to actual list for genre_df
    genre_df["genres"] = genre_df["genres"].apply(ast.literal_eval)
    # Convert all genres to lowercase
    genre_df["genres"] = genre_df["genres"].apply(lambda x: [g.lower() for g in x])

    # Get unique genres from movies_df and convert to lowercase
    movies_genres = set()
    for genre_list in movies_df["genres"].str.split("|"):
        movies_genres.update([genre.lower() for genre in genre_list])

    # Create genre mapping for common variations and combined genres (all in lowercase)
    genre_mapping = {
        "action/adventure": ["action", "adventure"],
        "thriller/suspense": ["thriller"],
        "crime film": ["crime"],
        "crime fiction": ["crime"],
        "criminal": ["crime"],
        "sci-fi": ["science fiction"],
        "science-fiction": ["science fiction"],
        "romantic comedy": ["romance", "comedy"],
        "rom-com": ["romance", "comedy"],
        "historical": ["history"],
        "biography": ["biographical"],
        "action/comedy": ["action", "comedy"],
        "horror/thriller": ["horror", "thriller"],
        "drama/romance": ["drama", "romance"],
        "fantasy/adventure": ["fantasy", "adventure"],
        "action-adventure": ["action", "adventure"],  # Handle hyphenated format
        "sci fi": ["science fiction"],  # Handle space format
        "scifi": ["science fiction"],  # Handle concatenated format
        "rom com": ["romance", "comedy"],
        "crime thriller": ["crime", "thriller"],
        "comedy-drama": ["comedy", "drama"],
        "science fiction": ["science fiction"],
    }

    # Count original genres distribution
    all_genres = []
    for genres in genre_df["genres"]:
        all_genres.extend(genres)
    original_genre_dist = Counter(all_genres)

    print("Original genre distribution:")
    for genre, count in original_genre_dist.most_common():
        print(f"{genre}: {count}")

    def find_matching_genre(genre):
        # Convert input genre to lowercase
        genre = genre.lower()

        # First check if it's a combined genre that should be split
        if genre in genre_mapping:
            return genre_mapping[genre]

        # Check if it contains a slash and isn't in our mapping
        if "/" in genre:
            # Split the genre and check each part
            parts = genre.split("/")
            matched_parts = []
            for part in parts:
                part = part.strip().lower()
                if part in movies_genres or original_genre_dist[part] >= min_frequency:
                    matched_parts.append(part)
                else:
                    # Try fuzzy matching for each part
                    matches = get_close_matches(part, movies_genres, n=1, cutoff=0.8)
                    if matches:
                        matched_parts.append(matches[0])
            return matched_parts if matched_parts else None

        # If it's already in movies_genres, keep it
        if genre in movies_genres:
            return [genre]

        # If it's a high-frequency genre, keep it
        if original_genre_dist[genre] >= min_frequency:
            return [genre]

        # Try to find close matches
        matches = get_close_matches(genre, movies_genres, n=1, cutoff=0.8)
        if matches:
            return [matches[0]]

        return None

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

    # Remove entries with no remaining genres after filtering
    genre_df = genre_df[genre_df["num_genres"] > 0]

    # Count new genres distribution
    all_filtered_genres = []
    for genres in genre_df["filtered_genres"]:
        all_filtered_genres.extend(genres)
    new_genre_dist = Counter(all_filtered_genres)

    print("\nNew genre distribution after filtering and mapping:")
    for genre, count in new_genre_dist.most_common():
        print(f"{genre}: {count}")

    # Print summary of changes
    print("\nGenre mapping summary:")
    for old_genre in original_genre_dist:
        if old_genre not in new_genre_dist:
            matched = find_matching_genre(old_genre)
            if matched:
                print(f"{old_genre} -> {matched}")

    # Print summary statistics
    print(f"\nOriginal number of movies: {len(genre_df)}")
    print(f"Original number of unique genres: {len(original_genre_dist)}")
    print(f"Number of movies after filtering: {len(genre_df)}")
    print(f"Number of unique genres after filtering: {len(new_genre_dist)}")

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
        DATASETS_BASE + "/aligned_genre_data.csv", index=False
    )

    return genre_df, movies_df, original_genre_dist, new_genre_dist


if __name__ == "__main__":
    # You can adjust the min_frequency threshold as needed
    processed_df, movies_df, original_dist, new_dist = load_and_process_data(
        min_frequency=1000
    )
