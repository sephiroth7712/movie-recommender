// src/services/auth.service.ts
import axios from "axios";
import { User } from "../hooks/useAuth";

const API_URL = import.meta.env.VITE_API_URL;

export interface LoginResponse extends User {}

export interface LoginRequest {
  username: string;
  password: string;
}

export const authService = {
  async login(data: LoginRequest): Promise<LoginResponse> {
    const response = await axios.post(`${API_URL}/login/`, data);
    return response.data;
  },
  async register(data: LoginRequest): Promise<LoginResponse> {
    const response = await axios.post(`${API_URL}/users/`, data);
    return response.data;
  },
  async logout(): Promise<void> {
    // Call logout endpoint if needed
    return Promise.resolve();
  },
};
