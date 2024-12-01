import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { authService } from "../services/auth.service";
import { userService } from "../services/user.service";

export interface User {
  id: string;
  username: string;
  movies_watched: number[];
}

interface AuthState {
  user: User | null;
  isLoading: boolean;
  isInitialized: boolean;
  login: (
    username: string,
    password: string,
    rememberMe?: boolean,
  ) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => void;
  initialize: () => Promise<void>;
}

export const useAuth = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      isLoading: true,
      isInitialized: false,
      initialize: async () => {
        try {
          set({ isLoading: true });
          let user = get().user;
          if (user) {
            user = await userService.getUser(user.id);
            set({ user });
          }
        } catch (error) {
          set({ user: null });
        } finally {
          set({ isLoading: false, isInitialized: true });
        }
      },
      login: async (
        username: string,
        password: string,
        rememberMe?: boolean,
      ) => {
        try {
          const loginData = {
            username,
            password,
          };
          const response = await authService.login(loginData);

          if (!response) {
            throw new Error("Invalid credentials");
          }

          set({ user: response });
        } catch (error) {
          throw error;
        } finally {
          set({ isLoading: false });
        }
      },
      register: async (username: string, password: string) => {
        try {
          const loginData = {
            username,
            password,
          };
          await authService.register(loginData);
        } catch (error) {
          throw error;
        } finally {
          set({ isLoading: false });
        }
      },
      logout: async () => {
        try {
          set({ isLoading: true });
          await authService.logout();
          set({ user: null });
        } finally {
          set({ isLoading: false });
        }
      },
    }),
    {
      name: "auth-storage", // name of the item in localStorage
      partialize: (state) => ({
        user: state.user,
      }),
    },
  ),
);

// Add a custom hook for protected routes
export function useRequireAuth() {
  const { user } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!user) {
      navigate("/login");
    }
  }, [user, navigate]);

  return user;
}
