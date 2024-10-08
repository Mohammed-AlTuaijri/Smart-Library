import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from './Auth';

interface ProtectedRouteProps {
  component: React.ComponentType<any>;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ component: Component, ...rest }) => {
  const { getUserInfo } = useAuth();
  const userInfo = getUserInfo();
  if (!userInfo) {
    return <Navigate to="/" />;
  }

  return <Component {...rest} />;
};

export default ProtectedRoute;
