/**
 * TranslateButton - Component for translating chapter content to Urdu
 *
 * Features:
 * - Triggers translation to Urdu
 * - Loading state during API call
 * - Error handling with dismissable messages
 * - Toggle to show English (original) content
 * - RTL support for translated content
 */

import React, { useState, useCallback } from 'react';
import { contentApi } from '../../services/api';
import styles from './TranslateButton.module.css';

interface TranslateButtonProps {
  chapterId: string;
  onTranslated?: (content: string, isRtl: boolean) => void;
  onShowOriginal?: () => void;
}

interface TranslationState {
  isLoading: boolean;
  error: string | null;
  translatedContent: string | null;
  isShowingTranslated: boolean;
  isRtl: boolean;
}

export function TranslateButton({
  chapterId,
  onTranslated,
  onShowOriginal,
}: TranslateButtonProps): JSX.Element {
  const [state, setState] = useState<TranslationState>({
    isLoading: false,
    error: null,
    translatedContent: null,
    isShowingTranslated: false,
    isRtl: false,
  });

  const handleTranslate = useCallback(async () => {
    setState((prev) => ({
      ...prev,
      isLoading: true,
      error: null,
    }));

    try {
      const response = await contentApi.translate(chapterId, 'urdu');

      setState((prev) => ({
        ...prev,
        isLoading: false,
        translatedContent: response.content,
        isShowingTranslated: true,
        isRtl: response.is_rtl,
      }));

      onTranslated?.(response.content, response.is_rtl);
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : 'Failed to translate content. Please try again.';

      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));
    }
  }, [chapterId, onTranslated]);

  const handleToggle = useCallback(() => {
    if (state.isShowingTranslated) {
      // Switch to original
      setState((prev) => ({
        ...prev,
        isShowingTranslated: false,
      }));
      onShowOriginal?.();
    } else if (state.translatedContent) {
      // Switch back to translated
      setState((prev) => ({
        ...prev,
        isShowingTranslated: true,
      }));
      onTranslated?.(state.translatedContent, state.isRtl);
    }
  }, [state.isShowingTranslated, state.translatedContent, state.isRtl, onTranslated, onShowOriginal]);

  const dismissError = useCallback(() => {
    setState((prev) => ({
      ...prev,
      error: null,
    }));
  }, []);

  const { isLoading, error, translatedContent, isShowingTranslated } = state;
  const hasTranslated = translatedContent !== null;

  return (
    <div className={styles.translateContainer}>
      {/* Main translate button - only show if not yet translated */}
      {!hasTranslated && (
        <button
          className={styles.translateButton}
          onClick={handleTranslate}
          disabled={isLoading}
          title="Translate to Urdu (اردو میں ترجمہ)"
        >
          {isLoading ? (
            <>
              <span className={styles.loadingSpinner} />
              <span>Translating...</span>
            </>
          ) : (
            <>
              <span className={styles.icon}>&#127760;</span>
              <span>Translate to Urdu</span>
              <span className={styles.urduText}>(اردو)</span>
            </>
          )}
        </button>
      )}

      {/* Toggle button - show after translation */}
      {hasTranslated && (
        <>
          <span className={styles.translatedBadge}>
            <span>&#10003;</span>
            <span className={styles.urduLabel}>اردو</span>
          </span>

          <button
            className={`${styles.toggleButton} ${!isShowingTranslated ? styles.active : ''}`}
            onClick={handleToggle}
          >
            {isShowingTranslated ? (
              <>
                <span>&#127468;&#127463;</span>
                <span>Show English</span>
              </>
            ) : (
              <>
                <span>&#127477;&#127472;</span>
                <span>Show Urdu</span>
                <span className={styles.urduText}>(اردو)</span>
              </>
            )}
          </button>
        </>
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

export default TranslateButton;
