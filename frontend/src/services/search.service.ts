import axios from "axios";
import { Movie } from "../types/movie";

export interface SearchMovieRequest {
  name: string;
  // filters:
}

export interface SearchMovieResponse {
  movies: Movie[];
}
const dummyMovies: Movie[] = [
  {
    id: "1",
    name: "Your Name",
    rating: 5,
    numberOfReviews: 1000,
    plot: "some dummy plot",
  },
  {
    id: "2",
    name: "Silent Voice",
    rating: 5,
    numberOfReviews: 1000,
    plot: "some dummy plot",
  },
  {
    id: "3",
    name: "Weathering With You",
    rating: 5,
    numberOfReviews: 1000,
    plot: "some dummy plot",
  },
];

export const searchService = {
  async searchMovie(data: SearchMovieRequest): Promise<SearchMovieResponse> {
    //   const response = await axios.post(`${API_URL}/auth/login`, data);
    //   return response.data;
    return { movies: dummyMovies };
  },
};
