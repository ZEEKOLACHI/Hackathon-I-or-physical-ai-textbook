/**
 * AuthProvider - React context for authentication state management
 */

import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import type { User, UserProfile, SignUpData, SignInData, ProfileUpdateData } from '../../lib/auth';
import { authApi, userApi } from '../../lib/auth';

interface AuthContextType {
  user: User | null;
  profile: UserProfile | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  signUp: (data: SignUpData) => Promise<void>;
  signIn: (data: SignInData) => Promise<void>;
  signOut: () => Promise<void>;
  updateProfile: (data: ProfileUpdateData) => Promise<void>;
  refreshSession: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

interface AuthProviderProps {
  children: React.ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps): JSX.Element {
  const [user, setUser] = useState<User | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const isAuthenticated = user !== null;

  // Refresh session on mount
  const refreshSession = useCallback(async () => {
    try {
      const sessionInfo = await authApi.getSession();
      setUser(sessionInfo.user);

      if (sessionInfo.user) {
        // Fetch full profile
        try {
          const userProfile = await userApi.getProfile();
          setProfile(userProfile);
        } catch {
          // Profile fetch failed, but user is still authenticated
          setProfile(null);
        }
      } else {
        setProfile(null);
      }
    } catch {
      // Silently handle session refresh failures (backend unavailable or not logged in)
      setUser(null);
      setProfile(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshSession();
  }, [refreshSession]);

  const signUp = async (data: SignUpData): Promise<void> => {
    const response = await authApi.signUp(data);
    setUser(response.user);
    // Fetch full profile after signup
    const userProfile = await userApi.getProfile();
    setProfile(userProfile);
  };

  const signIn = async (data: SignInData): Promise<void> => {
    const response = await authApi.signIn(data);
    setUser(response.user);
    // Fetch full profile after signin
    const userProfile = await userApi.getProfile();
    setProfile(userProfile);
  };

  const signOut = async (): Promise<void> => {
    await authApi.signOut();
    setUser(null);
    setProfile(null);
  };

  const updateProfile = async (data: ProfileUpdateData): Promise<void> => {
    const updatedProfile = await userApi.updateProfile(data);
    setProfile(updatedProfile);
  };

  const value: AuthContextType = {
    user,
    profile,
    isLoading,
    isAuthenticated,
    signUp,
    signIn,
    signOut,
    updateProfile,
    refreshSession,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export default AuthProvider;
