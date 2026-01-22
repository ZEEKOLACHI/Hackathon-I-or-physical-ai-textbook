/**
 * PersonalizeButton - Component for personalizing chapter content
 *
 * Features:
 * - Triggers personalization based on user's background profile
 * - Loading state during API call
 * - Error handling with dismissable messages
 * - Toggle to show original content
 */

import React, { useState, useCallback } from 'react';
import { contentApi } from '../../services/api';
import { useAuth } from '../Auth/AuthProvider';
import styles from './PersonalizeButton.module.css';

interface PersonalizeButtonProps {
  chapterId: string;
  onPersonalized?: (content: string) => void;
  onShowOriginal?: () => void;
}

interface PersonalizationState {
  isLoading: boolean;
  error: string | null;
  personalizedContent: string | null;
  isShowingPersonalized: boolean;
}

export function PersonalizeButton({
  chapterId,
  onPersonalized,
  onShowOriginal,
}: PersonalizeButtonProps): JSX.Element | null {
  const { isAuthenticated, profile } = useAuth();

  const [state, setState] = useState<PersonalizationState>({
    isLoading: false,
    error: null,
    personalizedContent: null,
    isShowingPersonalized: false,
  });

  const handlePersonalize = useCallback(async () => {
    if (!isAuthenticated) return;

    setState((prev) => ({
      ...prev,
      isLoading: true,
      error: null,
    }));

    try {
      const response = await contentApi.personalize(chapterId);

      setState((prev) => ({
        ...prev,
        isLoading: false,
        personalizedContent: response.content,
        isShowingPersonalized: true,
      }));

      onPersonalized?.(response.content);
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : 'Failed to personalize content. Please try again.';

      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, [chapterId, isAuthenticated, onPersonalized]);

  const handleToggle = useCallback(() => {
    if (state.isShowingPersonalized) {
      // Switch to original
      setState((prev) => ({
        ...prev,
        isShowingPersonalized: false,
      }));
      onShowOriginal?.();
    } else if (state.personalizedContent) {
      // Switch back to personalized
      setState((prev) => ({
        ...prev,
        isShowingPersonalized: true,
      }));
      onPersonalized?.(state.personalizedContent);
    }
  }, [state.isShowingPersonalized, state.personalizedContent, onPersonalized, onShowOriginal]);

  const dismissError = useCallback(() => {
    setState((prev) => ({
      ...prev,
      error: null,
    }));
  }, []);

  // Don't render for unauthenticated users
  if (!isAuthenticated) {
    return null;
  }

  const { isLoading, error, personalizedContent, isShowingPersonalized } = state;
  const hasPersonalized = personalizedContent !== null;

  // Generate hint text based on user profile
  const getHintText = () => {
    if (!profile) return '';
    return `Adapt for ${profile.programming_level} programming / ${profile.robotics_level} robotics level`;
  };

  return (
    <div className={styles.personalizeContainer}>
      {/* Main personalize button - only show if not yet personalized */}
      {!hasPersonalized && (
        <button
          className={styles.personalizeButton}
          onClick={handlePersonalize}
          disabled={isLoading}
          title={getHintText()}
        >
          {isLoading ? (
            <>
              <span className={styles.loadingSpinner} />
              <span>Personalizing...</span>
            </>
          ) : (
            <>
              <span className={styles.icon}>&#10024;</span>
              <span>Personalize for Me</span>
            </>
          )}
        </button>
      )}

      {/* Toggle button - show after personalization */}
      {hasPersonalized && (
        <>
          <span className={styles.personalizedBadge}>
            <span>&#10003;</span>
            <span>Personalized</span>
          </span>

          <button
            className={`${styles.toggleButton} ${!isShowingPersonalized ? styles.active : ''}`}
            onClick={handleToggle}
          >
            {isShowingPersonalized ? (
              <>
                <span>&#128196;</span>
                <span>Show Original</span>
              </>
            ) : (
              <>
                <span>&#10024;</span>
                <span>Show Personalized</span>
              </>
            )}
          </button>
        </>
      )}

      {/* Hint text */}
      {!hasPersonalized && !isLoading && (
        <span className={styles.hint}>{getHintText()}</span>
      )}

      {/* Error message */}
      {error && (
        <div className={styles.errorContainer}>
          <span className={styles.errorIcon}>&#9888;</span>
          <span className={styles.errorText}>{error}</span>
          <button className={styles.dismissError} onClick={dismissError}>
            &times;
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * PersonalizedContent - Wrapper component for displaying personalized content
 */
interface PersonalizedContentProps {
  content: string;
  onShowOriginal: () => void;
  children?: React.ReactNode;
}

export function PersonalizedContent({
  content,
  onShowOriginal,
  children,
}: PersonalizedContentProps): JSX.Element {
  return (
    <div className={styles.personalizedContent}>
      <div className={styles.personalizedHeader}>
        <div className={styles.personalizedTitle}>
          <span>&#10024;</span>
          <span>Personalized Content</span>
        </div>
        <button className={styles.toggleButton} onClick={onShowOriginal}>
          <span>&#128196;</span>
          <span>Show Original</span>
        </button>
      </div>
      <div className={styles.personalizedBody}>
        {children || (
          <div
            dangerouslySetInnerHTML={{ __html: content }}
          />
        )}
      </div>
    </div>
  );
}

export default PersonalizeButton;
