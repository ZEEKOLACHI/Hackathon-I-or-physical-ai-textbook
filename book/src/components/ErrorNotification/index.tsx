/**
 * ErrorNotification - Reusable error notification component
 */

import React, { useEffect, useState } from 'react';
import styles from './ErrorNotification.module.css';

interface ErrorNotificationProps {
  message: string | null;
  onDismiss?: () => void;
  autoHideDuration?: number;
  variant?: 'error' | 'warning' | 'info';
}

export function ErrorNotification({
  message,
  onDismiss,
  autoHideDuration,
  variant = 'error',
}: ErrorNotificationProps): JSX.Element | null {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (message) {
      setIsVisible(true);

      if (autoHideDuration && autoHideDuration > 0) {
        const timer = setTimeout(() => {
          setIsVisible(false);
          onDismiss?.();
        }, autoHideDuration);

        return () => clearTimeout(timer);
      }
    } else {
      setIsVisible(false);
    }
  }, [message, autoHideDuration, onDismiss]);

  if (!message || !isVisible) {
    return null;
  }

  const handleDismiss = () => {
    setIsVisible(false);
    onDismiss?.();
  };

  const icons = {
    error: '\u26A0', // Warning sign
    warning: '\u26A0',
    info: '\u2139', // Information
  };

  return (
    <div className={`${styles.notification} ${styles[variant]}`} role="alert">
      <span className={styles.icon}>{icons[variant]}</span>
      <span className={styles.message}>{message}</span>
      {onDismiss && (
        <button
          className={styles.dismissButton}
          onClick={handleDismiss}
          aria-label="Dismiss"
        >
          &times;
        </button>
      )}
    </div>
  );
}

export default ErrorNotification;
