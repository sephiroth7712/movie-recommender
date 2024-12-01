import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Movie } from "../types/movie";
import {
  recommendationService,
  RecommendationsResponse,
} from "../services/recommendation.service";
import { movieService } from "../services/movie.service";
import { genreColors } from "../config/genres";
import Recommendation from "../components/layout/Recommendations";
import { ratingService, SubmitRatingRequest } from "../services/rating.service";
import { useAuth } from "../hooks/useAuth";
import Rating from "../components/common/Rating";

export default function MovieDetail() {
  const { id } = useParams();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [movie, setMovie] = useState<Movie>();
  const [isMovieLoading, setIsMovieLoading] = useState(true);
  const [similarMovies, setSimilarMovies] = useState<RecommendationsResponse>();
  const [isSimilarMoviesLoading, setIsSimilarMoviesLoading] = useState(true);
  const [userRating, setUserRating] = useState<number>(0);
  const [isRatingLoading, setIsRatingLoading] = useState(true);
  const [hasUserAlreadyRated, setHasUserAlreadyRated] = useState(false);

  const getSimilarMovies = async (id: string) => {
    try {
      setIsSimilarMoviesLoading(true);
      const similarMovies = await recommendationService.getSimilarMovies({
        movie_id: id,
        n_recommendations: 4,
      });
      setSimilarMovies(similarMovies);
      setIsSimilarMoviesLoading(false);
    } catch (error) {
      console.log("Error fetching similar movies due to: " + error);
    }
  };

  const fetchMovieDetails = async (id: string) => {
    try {
      setIsMovieLoading(true);
      const movie = await movieService.getMovie(id);
      setMovie(movie);
      setIsMovieLoading(false);
    } catch (error) {
      alert("Error fetching movie due to: " + error);
    }
  };

  const fetchUserRating = async (movie_id: string, user_id: string) => {
    try {
      setIsRatingLoading(true);
      const rating = await ratingService.getRatingByUserForMovie(
        movie_id,
        user_id,
      );
      setUserRating(rating?.rating || 0);
      setIsRatingLoading(false);
      setHasUserAlreadyRated(rating?.rating ? true : false);
    } catch (error) {
      console.log("Error fetching user rating: " + error);
      setIsRatingLoading(false);
    }
  };

  const handleRatingChange = async (newRating: number) => {
    if (!id || !user) return;

    try {
      const request: SubmitRatingRequest = {
        movie_id: Number(id),
        rating: newRating,
        user_id: Number(user.id),
      };

      await ratingService.submitRating(request);
      navigate(0);
      setUserRating(newRating);
    } catch (error) {
      alert("Error updating rating: " + error);
    }
  };

  useEffect(() => {
    if (id) {
      fetchMovieDetails(id);
      getSimilarMovies(id);
    }
  }, []);

  useEffect(() => {
    if (id && user && user.id) {
      fetchUserRating(id, user.id);
    }
  }, [user]);

  if (movie) {
    return (
      <div>
        <div className="w-full p-6 bg-white border border-gray-200 rounded-lg shadow dark:bg-gray-800 dark:border-gray-700 mb-3">
          <div className="flex justify-between items-start mb-3">
            <div>
              <h5 className="mb-2 text-2xl font-bold tracking-tight text-gray-900 dark:text-white">
                {movie.title}
              </h5>
            </div>
            <div>
              <span className="text-sm text-gray-500">
                {movie.release_year}
              </span>
              {!isRatingLoading && (
                <Rating
                  initialRating={userRating}
                  onChange={handleRatingChange}
                  readonly={hasUserAlreadyRated}
                />
              )}
            </div>
          </div>
          <div className="flex flex-wrap gap-2 mb-3">
            {movie.genres.map((genre) => (
              <span
                className={`${genreColors[genre]} text-white text-xs font-medium me-2 px-2.5 py-0.5 rounded`}
              >
                {genre}
              </span>
            ))}
          </div>
          <p className="mb-3 font-normal text-gray-700 dark:text-gray-400">
            {movie.plot_summary}
          </p>
        </div>
        <Recommendation
          isLoading={isSimilarMoviesLoading}
          type="sm"
          movies={similarMovies}
        />
      </div>
    );
  }
}
