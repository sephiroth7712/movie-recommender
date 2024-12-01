import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL;

export interface SubmitRatingRequest {
  user_id: number;
  movie_id: number;
  rating: number;
}

export interface SubmitRatingResponse {
  rating_id: number;
}

export interface Rating {
    rating_id: number;
    movie_id: number;
    user_id: number;
    rating: number;
}

export const ratingService = {
  async submitRating(data: SubmitRatingRequest): Promise<SubmitRatingResponse> {
    const response = await axios.post(`${API_URL}/ratings/`, data);
    if (!response.data.rating_id) {
      throw Error("Error submitting rating");
    }
    return response.data;
  },

  async getRatingByUserForMovie(
    movie_id: string,
    user_id: string
  ): Promise<Rating> {
    const params = {
      movie_id,
      user_id,
    };
    const response = await axios.get<Rating[]>(`${API_URL}/ratings/`, { params });
    return response.data[0];
  },
};
