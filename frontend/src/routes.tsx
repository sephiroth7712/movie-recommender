import { Routes, Route, Navigate } from "react-router-dom";
import { Suspense, lazy } from "react";
import { ProtectedRoute } from "./components/auth/ProtectedRoute";
import Layout from "./components/layout/Layout";
import { Circles } from "react-loading-icons";

// Public pages
const Login = lazy(() => import("./pages/auth/Login"));
const Register = lazy(() => import("./pages/auth/Register"));
// const ForgotPassword = lazy(() => import("./pages/auth/ForgotPassword"));

// // Protected pages
const Dashboard = lazy(() => import("./pages/Dashboard"));
// const Profile = lazy(() => import("./pages/Profile"));
const MovieDetail = lazy(() => import("./pages/MovieDetail"));
const MovieGenreClassification = lazy(
  () => import("./pages/GenreClassification"),
);
const Watchlist = lazy(() => import("./pages/Watchlist"));

export function AppRoutes() {
  return (
    <Suspense
      fallback={
        <div className="h-screen flex items-center justify-center">
          <Circles />
        </div>
      }
    >
      <Routes>
        {/* Public Routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        {/* <Route path="/forgot-password" element={<ForgotPassword />} /> */}

        {/* Protected Routes */}
        <Route element={<ProtectedRoute />}>
          <Route element={<Layout />}>
            <Route path="/" element={<Dashboard />} />
            {/* <Route path="/profile" element={<Profile />} /> */}
            <Route path="/movie/:id" element={<MovieDetail />} />
            <Route path="/classify" element={<MovieGenreClassification />} />
            <Route path="/watchlist" element={<Watchlist />} />
            {/* Add more protected routes here */}
          </Route>
        </Route>

        {/* 404 Route */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
}
