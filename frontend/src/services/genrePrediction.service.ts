import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL;

export interface GenrePrediction {
  genre: string;
  confidence: number;
}

export interface GenrePredictionRequest {
  plot_summary: string;
}

export interface GenrePredictionResponse {
  predictions: GenrePrediction[];
  plot_summary: string;
}

class GenreClassificationService {
  async predictGenres(request: GenrePredictionRequest): Promise<GenrePredictionResponse> {
    const response = await axios.post<GenrePredictionResponse>(`${API_URL}/classify`, request);
    return response.data;
  }
}

export const genreClassificationService = new GenreClassificationService();