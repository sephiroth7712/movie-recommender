import { MdLocalMovies } from "react-icons/md";
import { IoIosPeople } from "react-icons/io";
import { RecommendationsResponse } from "../../services/recommendation.service";
import { useEffect, useState } from "react";
import MovieCard from "../features/MovieCard";
import MovieLoadingCard from "../features/MovieLoadingCard";

export interface IRecommendationsProps {
  isLoading: boolean;
  type: "cbf" | "cobf" | "sm";
  movies?: RecommendationsResponse;
}

export default function Recommendation({
  isLoading,
  type,
  movies,
}: IRecommendationsProps) {
  const [title, setTitle] = useState("");
  const [subtitle, setSubtitle] = useState("");
  const [icon, setIcon] = useState(<></>);

  useEffect(() => {
    if (type === "cbf") {
      setTitle("Based on What You've Watched");
      setSubtitle("Content-Based");
      setIcon(<MdLocalMovies className="w-5 h-5 mr-2" />);
    } else if (type == "cobf") {
      setTitle("People Also Watched");
      setSubtitle("Collaborative");
      setIcon(<IoIosPeople className="w-5 h-5 mr-2" />);
    } else {
      setTitle("Similar to This");
      setSubtitle("Content-Based");
      setIcon(<MdLocalMovies className="w-5 h-5 mr-2" />);
    }
  }, [type]);

  return (
    <section className="mb-12">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-800 flex items-center">
          {icon}
          <div>{title}</div>
          <span className="bg-blue-100 text-blue-600 text-xs ml-2 px-2 py-1 rounded-full">
            {subtitle}
          </span>
        </h2>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {!isLoading &&
          movies?.recommendations.map((movie) => (
            <MovieCard
              movie_id={movie.movie_id}
              title={movie.title}
              plot_summary={movie.plot_summary}
              release_year={movie.release_year}
              genres={movie.genres}
              similarity_score={movie.similarity_score}
            />
          ))}
        {isLoading && [...Array(4)].map((_, i) => <MovieLoadingCard key={i} />)}
      </div>
    </section>
  );
}
