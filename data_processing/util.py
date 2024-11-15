import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

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
    genres_list = df['genres'].apply(ast.literal_eval)
    
    # Count all genres
    print("Counting genres...")
    genre_counter = Counter()
    for genres in genres_list:
        genre_counter.update(genres)
    
    # Create DataFrame with genre statistics
    stats_df = pd.DataFrame.from_dict(genre_counter, orient='index', columns=['count'])
    stats_df = stats_df.sort_values('count')
    
    # Calculate percentage
    total_movies = len(df)
    stats_df['percentage'] = (stats_df['count'] / total_movies * 100).round(2)
    
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
    stats_df.head(n_genres).plot(kind='barh', y='count')
    plt.title(f'{n_genres} Least Common Genres')
    plt.xlabel('Number of Movies')
    
    # Plot most common genres
    plt.subplot(1, 2, 2)
    stats_df.tail(n_genres).plot(kind='barh', y='count')
    plt.title(f'{n_genres} Most Common Genres')
    plt.xlabel('Number of Movies')
    
    plt.tight_layout()
    plt.show()

def main():
    file_path = './processed_data/processed_genre_data.csv'  # Update this with your file path
    
    try:
        # Analyze genre distribution
        genre_counts, stats_df = analyze_genre_distribution(file_path)
        
        # Print statistics
        print_genre_stats(stats_df)
        
        # Print detailed statistics for low-count genres
        threshold = 100  # Adjust this threshold as needed
        low_count_genres = stats_df[stats_df['count'] < threshold]
        
        print(f"\nGenres with less than {threshold} movies:")
        print("=" * 50)
        print(low_count_genres.to_string())
        
        # Plot distribution
        plot_genre_distribution(stats_df)
        
        # Save statistics to CSV
        output_file = 'genre_distribution.csv'
        stats_df.to_csv(output_file)
        print(f"\nGenre distribution saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()