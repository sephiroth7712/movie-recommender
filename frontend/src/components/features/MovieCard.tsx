import { FaExternalLinkAlt } from "react-icons/fa";
import { genreColors } from "../../config/genres";
import { Movie } from "../../types/movie";
import CircularProgress from "../common/CircularProgress";

interface IMovieCardProps extends Movie {
  similarity_score?: number;
}

const MovieCard = ({
  movie_id,
  title,
  genres,
  plot_summary,
  release_year,
  similarity_score,
}: IMovieCardProps) => {
  return (
    <div className="max-w-sm p-6 bg-white border border-gray-200 rounded-lg shadow dark:bg-gray-800 dark:border-gray-700">
      <div className="flex justify-between items-start mb-3">
        <div>
          <a href={`/movie/${movie_id}`}>
            <h5 className="mb-2 text-2xl font-bold tracking-tight text-gray-900 dark:text-white">
              {title}
            </h5>
          </a>
          <span className="text-sm text-gray-500">{release_year}</span>
        </div>
        {similarity_score && (
          <CircularProgress percentage={Math.round(similarity_score * 100)} />
        )}
      </div>
      <p className="mb-3 font-normal text-gray-700 dark:text-gray-400 line-clamp-2">
        {plot_summary}
      </p>
      <div className="flex flex-wrap gap-2 mb-3">
        {genres.map((genre) => (
          <span
            className={`${genreColors[genre]} text-white text-xs font-medium me-2 px-2.5 py-0.5 rounded`}
          >
            {genre}
          </span>
        ))}
      </div>
      <a
        href={`/movie/${movie_id}`}
        className="inline-flex font-medium items-center text-blue-600 hover:underline"
      >
        Read more
        <FaExternalLinkAlt className="w-3 h-3 ms-2.5" />
      </a>
    </div>
  );
};

export default MovieCard;
