import { useEffect, useState } from "react";
import { useAuth, User } from "../hooks/useAuth";
import { Movie } from "../types/movie";
import { movieService } from "../services/movie.service";
import MovieCard from "../components/features/MovieCard";
import MovieLoadingCard from "../components/features/MovieLoadingCard";

export default function Recommendation() {
  const { user } = useAuth();
  const [movies, setMovies] = useState<Movie[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchMovies = async (user: User) => {
    setIsLoading(true);
    const moviePromises = user.movies_watched.map(async (movie_id) =>
      movieService.getMovie(movie_id.toString())
    );
    const movies = await Promise.all(moviePromises);

    setMovies(movies);
    setIsLoading(false);
  };

  useEffect(() => {
    if (user) {
      fetchMovies(user);
    }
  }, [user]);

  return (
    <section className="mb-12">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-800 flex items-center">
          <div>Movies You've Watched</div>
        </h2>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {!isLoading &&
          movies.map((movie) => (
            <MovieCard
              movie_id={movie.movie_id}
              title={movie.title}
              plot_summary={movie.plot_summary}
              release_year={movie.release_year}
              genres={movie.genres}
            />
          ))}
        {isLoading && [...Array(4)].map((_, i) => <MovieLoadingCard key={i} />)}
      </div>
    </section>
  );
}
