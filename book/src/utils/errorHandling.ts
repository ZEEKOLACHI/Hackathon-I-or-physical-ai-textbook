/**
 * Error handling utilities for user-friendly error messages
 */

import { AxiosError } from 'axios';

/**
 * API error response structure
 */
interface ApiErrorResponse {
  error: string;
  message: string;
}

/**
 * User-friendly error messages for common error codes
 */
const ERROR_MESSAGES: Record<string, string> = {
  // Network errors
  NETWORK_ERROR: 'Unable to connect to the server. Please check your internet connection.',
  TIMEOUT: 'The request took too long. Please try again.',

  // Auth errors
  unauthorized: 'Please sign in to continue.',
  invalid_credentials: 'Invalid email or password. Please try again.',
  session_expired: 'Your session has expired. Please sign in again.',
  email_exists: 'An account with this email already exists.',

  // Content errors
  not_found: 'The requested content could not be found.',
  chapter_not_found: 'This chapter could not be found.',

  // Rate limiting
  rate_limited: 'Too many requests. Please wait a moment and try again.',

  // Server errors
  internal_error: 'Something went wrong on our end. Please try again later.',
  service_unavailable: 'The service is temporarily unavailable. Please try again later.',

  // Generic
  unknown: 'An unexpected error occurred. Please try again.',
};

/**
 * Get a user-friendly error message from an error object
 */
export function getErrorMessage(error: unknown): string {
  // Handle Axios errors
  if (isAxiosError(error)) {
    const axiosError = error as AxiosError<ApiErrorResponse>;

    // Network error (no response)
    if (!axiosError.response) {
      if (axiosError.code === 'ECONNABORTED') {
        return ERROR_MESSAGES.TIMEOUT;
      }
      return ERROR_MESSAGES.NETWORK_ERROR;
    }

    // Server responded with error
    const { status, data } = axiosError.response;

    // Check for specific error code in response
    if (data?.error && ERROR_MESSAGES[data.error]) {
      return ERROR_MESSAGES[data.error];
    }

    // Check for message in response
    if (data?.message) {
      return data.message;
    }

    // Fall back to HTTP status code
    switch (status) {
      case 400:
        return 'Invalid request. Please check your input.';
      case 401:
        return ERROR_MESSAGES.unauthorized;
      case 403:
        return 'You do not have permission to perform this action.';
      case 404:
        return ERROR_MESSAGES.not_found;
      case 429:
        return ERROR_MESSAGES.rate_limited;
      case 500:
        return ERROR_MESSAGES.internal_error;
      case 502:
      case 503:
      case 504:
        return ERROR_MESSAGES.service_unavailable;
      default:
        return ERROR_MESSAGES.unknown;
    }
  }

  // Handle standard Error objects
  if (error instanceof Error) {
    // Don't expose raw error messages to users in production
    if (process.env.NODE_ENV === 'development') {
      return error.message;
    }
    return ERROR_MESSAGES.unknown;
  }

  // Handle string errors
  if (typeof error === 'string') {
    return error;
  }

  return ERROR_MESSAGES.unknown;
}

/**
 * Type guard for Axios errors
 */
function isAxiosError(error: unknown): error is AxiosError {
  return (
    typeof error === 'object' &&
    error !== null &&
    'isAxiosError' in error &&
    (error as AxiosError).isAxiosError === true
  );
}

/**
 * Error notification component props
 */
export interface ErrorNotificationProps {
  message: string;
  onDismiss?: () => void;
  autoHideDuration?: number;
}

/**
 * Create an error handler that shows user-friendly messages
 */
export function createErrorHandler(
  setError: (message: string | null) => void,
  options?: { autoHide?: boolean; duration?: number }
): (error: unknown) => void {
  return (error: unknown) => {
    const message = getErrorMessage(error);
    setError(message);

    // Auto-hide error after duration
    if (options?.autoHide) {
      setTimeout(() => {
        setError(null);
      }, options.duration || 5000);
    }
  };
}

/**
 * Wrapper for async operations with error handling
 */
export async function withErrorHandling<T>(
  operation: () => Promise<T>,
  onError: (message: string) => void
): Promise<T | null> {
  try {
    return await operation();
  } catch (error) {
    onError(getErrorMessage(error));
    return null;
  }
}
