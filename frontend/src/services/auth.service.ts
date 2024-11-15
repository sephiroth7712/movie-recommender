// src/services/auth.service.ts
import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL;

// Add auth token to all requests
axios.interceptors.request.use((config) => {
  const token = JSON.parse(localStorage.getItem("auth-storage") || "{}")?.state
    ?.token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export interface LoginResponse {
  user: {
    id: string;
    email: string;
    name: string;
    imageUrl?:string;
  };
  token: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export const authService = {
  async login(data: LoginRequest): Promise<LoginResponse> {
    const response = await axios.post(`${API_URL}/auth/login`, data);
    return response.data;
  },

  async logout(): Promise<void> {
    // Call logout endpoint if needed
    return Promise.resolve();
  },
  async validateToken() {
    try {
      const response = await axios.get(`${API_URL}/auth/validate`);
      return response.data.user;
    } catch (error) {
      throw new Error("Invalid token");
    }
  },
};
