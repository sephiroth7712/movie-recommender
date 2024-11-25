import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os

class UserBasedRecommender:
    def __init__(self):
        self.user_to_movie_ratings = {}
        self.movies_df = None
        self.valid_movie_ids = set()
        
    def fit(self, ratings_df, movies_df):
        """Fit the recommender system with user ratings data"""
        print("Processing data...")
        
        # Store valid movie IDs
        self.valid_movie_ids = set(movies_df['movieId'].unique())
        
        # Create user-movie ratings dictionary
        for _, row in ratings_df.iterrows():
            user_id = row['userId']
            if user_id not in self.user_to_movie_ratings:
                self.user_to_movie_ratings[user_id] = {}
            self.user_to_movie_ratings[user_id][row['movieId']] = row['rating']
        
        # Store movies dataframe
        self.movies_df = movies_df
        print("Model fitted successfully!")
    
    def get_movie_details(self, movie_id):
        """Safely get movie details"""
        movie_data = self.movies_df[self.movies_df['movieId'] == movie_id]
        if movie_data.empty:
            return None
        return movie_data.iloc[0]
    
    def find_similar_users(self, user_ratings, n_similar=10):
        """Find users who rated similar movies similarly"""
        user_similarities = []
        
        for user_id, ratings in self.user_to_movie_ratings.items():
            similarity_score = 0
            common_movies = 0
            
            # Compare ratings for common movies
            for movie_id, rating in user_ratings.items():
                if movie_id in ratings:
                    # Calculate rating difference
                    rating_diff = abs(rating - ratings[movie_id])
                    # Similarity scoring based on rating difference
                    if rating_diff == 0:
                        similarity_score += 1.0
                    elif rating_diff <= 0.5:
                        similarity_score += 0.9
                    elif rating_diff <= 1.0:
                        similarity_score += 0.7
                    elif rating_diff <= 1.5:
                        similarity_score += 0.5
                    elif rating_diff <= 2.0:
                        similarity_score += 0.3
                    common_movies += 1
            
            if common_movies > 0:
                # Calculate final similarity score
                avg_similarity = similarity_score / common_movies
                # Bonus for more common movies
                final_similarity = avg_similarity * (1 + min(common_movies/5, 1))
                user_similarities.append((user_id, final_similarity))
        
        # Sort and return top similar users
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        return user_similarities[:n_similar]
    
    def get_recommendations(self, user_ratings, n_recommendations=5):
        """Generate recommendations with improved error handling"""
        try:
            similar_users = self.find_similar_users(user_ratings)
            if not similar_users:
                return []
            
            # Get genres of rated movies
            user_genres = set()
            for movie_id in user_ratings:
                movie = self.get_movie_details(movie_id)
                if movie is not None:
                    user_genres.update(movie['genres'].split('|'))
            
            # Collect movie scores
            movie_scores = {}
            movie_counts = {}
            movie_similarities = {}
            
            for user_id, similarity in similar_users:
                user_ratings_dict = self.user_to_movie_ratings[user_id]
                
                for movie_id, rating in user_ratings_dict.items():
                    # Skip if movie isn't in our valid set or already rated
                    if movie_id not in self.valid_movie_ids or movie_id in user_ratings:
                        continue
                        
                    if rating >= 3.5:  # Only consider above-average ratings
                        if movie_id not in movie_scores:
                            movie = self.get_movie_details(movie_id)
                            if movie is None:
                                continue
                                
                            movie_scores[movie_id] = 0
                            movie_counts[movie_id] = 0
                            
                            # Calculate genre similarity
                            movie_genres = set(movie['genres'].split('|'))
                            genre_similarity = len(user_genres.intersection(movie_genres)) / len(user_genres.union(movie_genres)) if user_genres else 0
                            movie_similarities[movie_id] = genre_similarity
                        
                        # Weight rating by both similarities
                        weighted_rating = rating * similarity * (1 + movie_similarities[movie_id])
                        movie_scores[movie_id] += weighted_rating
                        movie_counts[movie_id] += 1
            
            # Calculate final scores
            recommendations = []
            for movie_id in movie_scores:
                if movie_counts[movie_id] >= 2:  # Require at least 2 similar users
                    avg_score = movie_scores[movie_id] / movie_counts[movie_id]
                    confidence_factor = min(movie_counts[movie_id]/5, 1)
                    final_score = avg_score * (1 + confidence_factor * 0.2)
                    recommendations.append((movie_id, final_score))
            
            if not recommendations:
                return []
            
            # Sort and prepare recommendations
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            top_recommendations = []
            for movie_id, score in recommendations[:n_recommendations]:
                movie = self.get_movie_details(movie_id)
                if movie is None:
                    continue
                    
                scaled_score = min(5, max(1, score * 3.5))
                top_recommendations.append({
                    'movieId': movie_id,
                    'title': movie['title'],
                    'genres': movie['genres'],
                    'predicted_rating': round(float(scaled_score), 2),
                    'year': movie.get('year'),
                    'average_rating': round(float(movie['average_rating']), 2) if 'average_rating' in movie else None,
                    'num_similar_users': movie_counts[movie_id],
                    'genre_similarity': round(movie_similarities[movie_id] * 100, 1)
                })
            
            return top_recommendations
            
        except Exception as e:
            print(f"Error in get_recommendations: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []
        
def search_movies(movies_df, title_query):
    """Search for movies by title"""
    matches = movies_df[
        movies_df['title'].str.contains(title_query, case=False, na=False)
    ]
    return matches[['movieId', 'title', 'genres', 'year', 'average_rating']].head()

def format_genres(genres_string):
    """Format genres string for display"""
    return genres_string.replace('|', ', ')

def main():
    print("=== Movie Recommender System ===\n")
    
    # Load data
    print("Loading data...")
    try:
        base_path = r"C:\VCU\DataScience\Project\movie-recommender\data_processing\datasets\processed_data"
        
        ratings_path = os.path.join(base_path, "processed_ratings.csv")
        movies_path = os.path.join(base_path, "merged_movies_dataset.csv")
        
        print(f"Loading ratings from: {ratings_path}")
        print(f"Loading movies from: {movies_path}")
        
        ratings_df = pd.read_csv(ratings_path)
        print(f"Total ratings loaded: {len(ratings_df)}")
        
        # Sample users who have rated at least 5 movies
        user_counts = ratings_df['userId'].value_counts()
        active_users = user_counts[user_counts >= 5].index
        if len(active_users) > 10000:
            active_users = np.random.choice(active_users, 10000, replace=False)
        
        ratings_df = ratings_df[ratings_df['userId'].isin(active_users)]
        print(f"Using {len(ratings_df)} ratings from {len(active_users)} users")
        
        movies_df = pd.read_csv(movies_path, low_memory=False)
        print(f"Movies loaded: {len(movies_df)}")
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Initialize and fit recommender
    print("\nInitializing recommender...")
    recommender = UserBasedRecommender()
    recommender.fit(ratings_df, movies_df)
    
    # Interactive loop for movie ratings
    user_ratings = {}
    print("\nStart rating movies to get personalized recommendations!")
    print("Rate movies on a scale of 1-5 (1=Terrible, 5=Excellent)")
    
    while True:
        title = input("\nEnter a movie title to search (or 'done' to finish): ")
        if title.lower() == 'done':
            break
            
        matches = search_movies(movies_df, title)
        if matches.empty:
            print("No movies found with that title.")
            continue
            
        print("\nFound these movies:")
        for i, (_, movie) in enumerate(matches.iterrows(), 1):
            print(f"\n{i}. {movie['title']} ({movie['year']})")
            print(f"   Genres: {format_genres(movie['genres'])}")
            if 'average_rating' in movie and not pd.isna(movie['average_rating']):
                print(f"   Average Rating: {round(movie['average_rating'], 2)}/5.0")
        
        selection = input("\nEnter the number of the movie you want to rate (0 to skip): ")
        if not selection.isdigit() or int(selection) == 0:
            continue
            
        selection = int(selection) - 1
        if selection < 0 or selection >= len(matches):
            print("Invalid selection")
            continue
            
        rating = input("Enter your rating (1-5): ")
        if not rating.replace('.', '').isdigit():
            print("Invalid rating")
            continue
            
        rating = float(rating)
        if rating < 1 or rating > 5:
            print("Rating must be between 1 and 5")
            continue
            
        movie_id = matches.iloc[selection]['movieId']
        user_ratings[movie_id] = rating
        
        # Show current ratings
        print("\nYour current ratings:")
        for rated_movie_id, user_rating in user_ratings.items():
            movie = movies_df[movies_df['movieId'] == rated_movie_id].iloc[0]
            print(f"- {movie['title']}: {user_rating}/5.0")
        
        # Get and show recommendations
        print(f"\nBased on users with similar taste, you might enjoy:")
        recommendations = recommender.get_recommendations(user_ratings)
        
        if recommendations:
            for i, movie in enumerate(recommendations, 1):
                print(f"\n{i}. {movie['title']} ({movie['year']})")
                print(f"   Genres: {format_genres(movie['genres'])}")
                print(f"   Predicted Rating: {movie['predicted_rating']}/5.0")
                print(f"   Average Rating: {movie['average_rating']}/5.0")
                print(f"   Genre Similarity: {movie['genre_similarity']}%")
                print(f"   Recommended by {movie['num_similar_users']} similar users")
        else:
            print("\nNeed more ratings to make recommendations. Please rate more movies.")
    
    print("\n=== Recommendation Complete ===")

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")