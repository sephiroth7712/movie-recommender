import axios from "axios";
import { Movie } from "../types/movie";

const API_URL = import.meta.env.VITE_API_URL;

export interface UserRecommendationsRequest {
  type: "cbf" | "cobf";
  userId: string;
  n_recommendations?: number;
}

export interface SimilarMoviesRequest {
  movie_id: string;
  n_recommendations?: number;
}

export interface RecommendationsResponse {
  recommendations: (Movie & {
    similarity_score: number;
  })[];
}

export const recommendationService = {
  async getUserRecommendations(
    data: UserRecommendationsRequest
  ): Promise<RecommendationsResponse> {
    const params = {
      n_recommendations: data.n_recommendations,
      type: data.type,
    };
    const response = await axios.get(
      `${API_URL}/users/${data.userId}/recommendations`,
      { params }
    );
    return response.data;
  },

  async getSimilarMovies(
    data: SimilarMoviesRequest
  ): Promise<RecommendationsResponse> {
    const params = {
      n_recommendations: data.n_recommendations,
    };
    const response = await axios.get(
      `${API_URL}/movies/${data.movie_id}/recommendations`,
      { params }
    );
    return response.data;
  },
};
