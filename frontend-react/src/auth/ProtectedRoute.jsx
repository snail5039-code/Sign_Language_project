import { Navigate, Outlet } from "react-router-dom";
import { useAuth } from "./AuthProvider";

export default function ProtectedRoute() {
  const { isAuthed, loading } = useAuth();
  if (loading) return null;
  if (!isAuthed) return <Navigate to="/login" replace />;
  return <Outlet />;
}
