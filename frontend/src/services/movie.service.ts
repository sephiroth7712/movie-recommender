import axios from "axios";
import { Movie } from "../types/movie";

const API_URL = import.meta.env.VITE_API_URL;

export interface SearchMovieRequest {
  name: string;
  // filters:
}

export interface SearchMovieResponse {
  movies: Movie[];
}

export const movieService = {
  async searchMovie(data: SearchMovieRequest): Promise<SearchMovieResponse> {
    const params = {
      name: data.name,
    };
    const response = await axios.get(`${API_URL}/search/movies`, { params });
    return { movies: response.data };
  },

  async getMovie(movie_id: string): Promise<Movie> {
    const response = await axios.get(`${API_URL}/movies/${movie_id}`);
    return response.data;
  },
};
