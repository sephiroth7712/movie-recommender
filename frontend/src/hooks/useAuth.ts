import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { authService } from "../services/auth.service";

interface User {
  id: string;
  email: string;
  name: string;
  imageUrl?: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  isInitialized: boolean;
  login: (
    email: string,
    password: string,
    rememberMe?: boolean
  ) => Promise<void>;
  logout: () => void;
  initialize: () => Promise<void>;
}

export const useAuth = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isLoading: true,
      isInitialized: false,
      initialize: async () => {
        try {
          set({ isLoading: true });
          // Check if token is valid
          const token = get().token;
          if (token) {
            const user = await authService.validateToken();
            set({ user, isInitialized: true });
          }
        } catch (error) {
          // If token validation fails, clear the auth state
          set({ user: null, token: null });
        } finally {
          set({ isLoading: false, isInitialized: true });
        }
      },
      login: async (email: string, password: string, rememberMe?: boolean) => {
        try {
          //   // TODO: Replace with your actual API call
          //   const response = await fetch('/api/auth/login', {
          //     method: 'POST',
          //     headers: {
          //       'Content-Type': 'application/json',
          //     },
          //     body: JSON.stringify({ email, password }),
          //   });

          //   if (!response.ok) {
          //     throw new Error('Invalid credentials');
          //   }

          //   const data = await response.json();
          //   set({ user: data.user, token: data.token });
          set({
            user: {
              id: "sephiroth7712",
              email: "roshanjames7@gmail.com",
              name: "Roshan",
              imageUrl:"https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
            },
            token: "abcd1234",
          });
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
          set({ user: null, token: null });
        } finally {
          set({ isLoading: false });
        }
      },
    }),
    {
      name: "auth-storage", // name of the item in localStorage
      partialize: (state) => ({
        user: state.user,
        token: state.token,
      }),
    }
  )
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
