/**
 * API client for backend communication
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

// Get backend URL from Docusaurus config or fallback
const getBackendUrl = (): string => {
  // In browser, try to get from Docusaurus config
  if (typeof window !== 'undefined') {
    try {
      // Access the global docusaurus data if available
      const docusaurusData = (window as any).__DOCUSAURUS__;
      if (docusaurusData?.siteConfig?.customFields?.backendUrl) {
        return docusaurusData.siteConfig.customFields.backendUrl;
      }
    } catch (e) {
      // Fallback silently
    }
  }
  // Default: production URL (will be overridden in dev by local proxy or direct calls)
  return 'https://physical-ai-backend.vercel.app';
};

const API_BASE_URL = getBackendUrl();

// Create axios instance with default config
const apiClient: AxiosInstance = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // Send cookies for auth
});

// Request interceptor for adding auth token
apiClient.interceptors.request.use(
  (config) => {
    // Auth token is handled via cookies (Better-Auth)
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<{ error: string; message: string }>) => {
    // Silently handle expected errors (auth checks, backend unavailable)
    const isAuthEndpoint = error.config?.url?.includes('/auth/');
    const isNetworkError = !error.response;

    if (error.response) {
      const { status } = error.response;
      // Don't log 401/404 on auth endpoints - these are expected when not logged in
      if (!(isAuthEndpoint && (status === 401 || status === 404))) {
        // Only log unexpected errors in development
        if (process.env.NODE_ENV === 'development') {
          console.debug(`API Error [${status}]:`, error.message);
        }
      }
    } else if (isNetworkError) {
      // Silently handle network errors (backend unavailable)
      // Only log in development if it's not an auth check
      if (process.env.NODE_ENV === 'development' && !isAuthEndpoint) {
        console.debug('Backend unavailable');
      }
    }

    return Promise.reject(error);
  }
);

// API types
export interface ChatSession {
  id: string;
  user_id: string | null;
  context_chapter: string | null;
  created_at: string;
  last_message_at: string;
}

export interface Citation {
  chapter_id: string;
  section_id: string;
  section_title: string;
  relevance_score: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations: Citation[];
  created_at: string;
}

export interface ChatResponse {
  message: ChatMessage;
  session: ChatSession;
}

export interface SearchResult {
  chunk_id: string;
  chapter_id: string;
  section_id: string;
  section_title: string;
  content_preview: string;
  has_code: boolean;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  score: number;
}

// Chat API
export const chatApi = {
  createSession: async (contextChapter?: string): Promise<ChatSession> => {
    const response = await apiClient.post<ChatSession>('/chat/sessions', {
      context_chapter: contextChapter,
    });
    return response.data;
  },

  getMessages: async (
    sessionId: string,
    limit?: number,
    before?: string
  ): Promise<{ messages: ChatMessage[]; has_more: boolean }> => {
    const response = await apiClient.get(`/chat/sessions/${sessionId}/messages`, {
      params: { limit, before },
    });
    return response.data;
  },

  sendMessage: async (
    sessionId: string,
    content: string,
    selectedText?: string
  ): Promise<ChatResponse> => {
    const response = await apiClient.post<ChatResponse>(
      `/chat/sessions/${sessionId}/messages`,
      {
        content,
        selected_text: selectedText,
      }
    );
    return response.data;
  },
};

// Search API
export const searchApi = {
  search: async (
    query: string,
    options?: {
      chapter_id?: string;
      difficulty?: 'beginner' | 'intermediate' | 'advanced';
      limit?: number;
    }
  ): Promise<{ results: SearchResult[]; query: string }> => {
    const response = await apiClient.get('/search', {
      params: { q: query, ...options },
    });
    return response.data;
  },
};

// Content API (personalization & translation)
export const contentApi = {
  personalize: async (chapterId: string): Promise<{ content: string; is_rtl: boolean }> => {
    const response = await apiClient.post('/content/personalize', {
      chapter_id: chapterId,
    });
    return response.data;
  },

  translate: async (
    chapterId: string,
    targetLanguage: 'urdu'
  ): Promise<{ content: string; is_rtl: boolean }> => {
    const response = await apiClient.post('/content/translate', {
      chapter_id: chapterId,
      target_language: targetLanguage,
    });
    return response.data;
  },
};

// Health check
export const healthApi = {
  check: async (): Promise<{ status: string; version: string }> => {
    const response = await apiClient.get('/health');
    return response.data;
  },
};

export default apiClient;
