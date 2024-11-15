import { Navigate, useLocation, Outlet } from "react-router-dom";
import { useAuth } from "../../hooks/useAuth";
import { Circles } from "react-loading-icons";

export function ProtectedRoute() {
  const { user, isLoading } = useAuth();
  const location = useLocation();

  // Show loading spinner while checking authentication status
  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <Circles />
      </div>
    );
  }

  // Redirect to login if not authenticated
  if (!user) {
    // Save the attempted URL for redirecting after login
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Render child routes if authenticated
  return <Outlet />;
}
