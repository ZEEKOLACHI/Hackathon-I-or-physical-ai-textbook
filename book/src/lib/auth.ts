/**
 * Authentication client for Better-Auth style authentication
 */

import apiClient from '../services/api';

// Types
export interface User {
  id: string;
  email: string;
  created_at: string;
}

export interface UserProfile extends User {
  programming_level: 'none' | 'beginner' | 'intermediate' | 'advanced';
  robotics_level: 'none' | 'beginner' | 'intermediate' | 'advanced';
  hardware_available: string[];
  updated_at: string;
}

export interface Session {
  id: string;
  expires_at: string;
  created_at: string;
}

export interface AuthResponse {
  user: User;
  session: Session;
}

export interface SessionInfo {
  user: User | null;
  session: Session | null;
}

export interface SignUpData {
  email: string;
  password: string;
  programming_level?: 'none' | 'beginner' | 'intermediate' | 'advanced';
  robotics_level?: 'none' | 'beginner' | 'intermediate' | 'advanced';
  hardware_available?: string[];
}

export interface SignInData {
  email: string;
  password: string;
}

export interface ProfileUpdateData {
  programming_level?: 'none' | 'beginner' | 'intermediate' | 'advanced';
  robotics_level?: 'none' | 'beginner' | 'intermediate' | 'advanced';
  hardware_available?: string[];
}

// Auth API
export const authApi = {
  /**
   * Register a new user
   */
  signUp: async (data: SignUpData): Promise<AuthResponse> => {
    const response = await apiClient.post<AuthResponse>('/auth/signup', data);
    return response.data;
  },

  /**
   * Sign in an existing user
   */
  signIn: async (data: SignInData): Promise<AuthResponse> => {
    const response = await apiClient.post<AuthResponse>('/auth/signin', data);
    return response.data;
  },

  /**
   * Sign out the current user
   */
  signOut: async (): Promise<void> => {
    await apiClient.post('/auth/signout');
  },

  /**
   * Get current session info
   */
  getSession: async (): Promise<SessionInfo> => {
    const response = await apiClient.get<SessionInfo>('/auth/session');
    return response.data;
  },
};

// User Profile API
export const userApi = {
  /**
   * Get current user's full profile
   */
  getProfile: async (): Promise<UserProfile> => {
    const response = await apiClient.get<UserProfile>('/users/me');
    return response.data;
  },

  /**
   * Update current user's profile
   */
  updateProfile: async (data: ProfileUpdateData): Promise<UserProfile> => {
    const response = await apiClient.patch<UserProfile>('/users/me', data);
    return response.data;
  },
};

export default authApi;
