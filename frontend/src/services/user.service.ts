// src/services/auth.service.ts
import axios from "axios";
import { User } from "../hooks/useAuth";

const API_URL = import.meta.env.VITE_API_URL;

export interface UserResponse extends User {}

export const userService = {
  async getUser(user_id: string): Promise<UserResponse> {
    const response = await axios.get(`${API_URL}/users/${user_id}`);
    return response.data;
  },
};
