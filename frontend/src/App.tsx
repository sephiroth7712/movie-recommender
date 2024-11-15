// src/App.tsx
import { useEffect } from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppRoutes } from './routes';
import { useAuth } from './hooks/useAuth';
import {Circles} from 'react-loading-icons';

const queryClient = new QueryClient();

function App() {
  const { initialize, isInitialized } = useAuth();

  useEffect(() => {
    initialize();
  }, [initialize]);

  if (!isInitialized) {
    return (
      <div className="h-screen flex items-center justify-center">
        <Circles />
      </div>
    );
  }

  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <AppRoutes />
      </Router>
    </QueryClientProvider>
  );
}

export default App;