/**
 * LoadingSpinner - Reusable loading indicator components
 */

import React from 'react';
import styles from './LoadingSpinner.module.css';

interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  color?: 'primary' | 'white' | 'gray';
  className?: string;
}

/**
 * Circular spinning loader
 */
export function LoadingSpinner({
  size = 'medium',
  color = 'primary',
  className = '',
}: LoadingSpinnerProps): JSX.Element {
  return (
    <div
      className={`${styles.spinner} ${styles[size]} ${styles[color]} ${className}`}
      role="status"
      aria-label="Loading"
    >
      <span className={styles.srOnly}>Loading...</span>
    </div>
  );
}

interface LoadingDotsProps {
  color?: 'primary' | 'white' | 'gray';
  className?: string;
}

/**
 * Animated dots loader (good for chat/typing indicators)
 */
export function LoadingDots({
  color = 'gray',
  className = '',
}: LoadingDotsProps): JSX.Element {
  return (
    <div
      className={`${styles.dots} ${styles[color]} ${className}`}
      role="status"
      aria-label="Loading"
    >
      <span className={styles.dot} />
      <span className={styles.dot} />
      <span className={styles.dot} />
      <span className={styles.srOnly}>Loading...</span>
    </div>
  );
}

interface LoadingOverlayProps {
  message?: string;
  children?: React.ReactNode;
}

/**
 * Full overlay with loading spinner (for blocking operations)
 */
export function LoadingOverlay({
  message = 'Loading...',
  children,
}: LoadingOverlayProps): JSX.Element {
  return (
    <div className={styles.overlay}>
      <div className={styles.overlayContent}>
        <LoadingSpinner size="large" />
        {message && <p className={styles.overlayMessage}>{message}</p>}
        {children}
      </div>
    </div>
  );
}

interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  borderRadius?: string | number;
  className?: string;
}

/**
 * Skeleton loading placeholder
 */
export function Skeleton({
  width = '100%',
  height = '1em',
  borderRadius = '4px',
  className = '',
}: SkeletonProps): JSX.Element {
  return (
    <div
      className={`${styles.skeleton} ${className}`}
      style={{
        width: typeof width === 'number' ? `${width}px` : width,
        height: typeof height === 'number' ? `${height}px` : height,
        borderRadius: typeof borderRadius === 'number' ? `${borderRadius}px` : borderRadius,
      }}
      aria-hidden="true"
    />
  );
}

/**
 * Skeleton for text paragraphs
 */
export function SkeletonText({
  lines = 3,
  className = '',
}: {
  lines?: number;
  className?: string;
}): JSX.Element {
  return (
    <div className={`${styles.skeletonText} ${className}`}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          height="1em"
          width={i === lines - 1 ? '60%' : '100%'}
        />
      ))}
    </div>
  );
}

export default LoadingSpinner;
