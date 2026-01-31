/**
 * UserMenu - Navbar component for authentication status
 */

import React, { useState } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import { useAuth } from './AuthProvider';
import { SignInForm } from './SignInForm';
import { SignUpForm } from './SignUpForm';
import styles from './Auth.module.css';

type AuthMode = 'signin' | 'signup' | null;

// Inner component that uses auth hooks (browser-only)
function UserMenuContent(): JSX.Element {
  const { user, profile, isLoading, isAuthenticated, signOut } = useAuth();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [authMode, setAuthMode] = useState<AuthMode>(null);

  const handleSignOut = async () => {
    await signOut();
    setIsMenuOpen(false);
  };

  const handleAuthSuccess = () => {
    setAuthMode(null);
  };

  if (isLoading) {
    return <div className={styles.userMenuLoading}>...</div>;
  }

  if (!isAuthenticated) {
    return (
      <>
        <button
          className={styles.signInButton}
          onClick={() => setAuthMode('signin')}
        >
          Sign In
        </button>

        {authMode && (
          <div className={styles.modal} onClick={() => setAuthMode(null)}>
            <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
              <button
                className={styles.closeButton}
                onClick={() => setAuthMode(null)}
                aria-label="Close"
              >
                &times;
              </button>
              {authMode === 'signin' ? (
                <SignInForm
                  onSuccess={handleAuthSuccess}
                  onSwitchToSignUp={() => setAuthMode('signup')}
                />
              ) : (
                <SignUpForm
                  onSuccess={handleAuthSuccess}
                  onSwitchToSignIn={() => setAuthMode('signin')}
                />
              )}
            </div>
          </div>
        )}
      </>
    );
  }

  return (
    <div className={styles.userMenu}>
      <button
        className={styles.userButton}
        onClick={() => setIsMenuOpen(!isMenuOpen)}
        aria-expanded={isMenuOpen}
      >
        <span className={styles.userAvatar}>
          {user?.email?.charAt(0).toUpperCase()}
        </span>
        <span className={styles.userEmail}>{user?.email}</span>
        <svg
          className={`${styles.chevron} ${isMenuOpen ? styles.open : ''}`}
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {isMenuOpen && (
        <div className={styles.dropdown}>
          <div className={styles.dropdownHeader}>
            <div className={styles.email}>{user?.email}</div>
            {profile && (
              <div className={styles.profileInfo}>
                <span>Programming: {profile.programming_level}</span>
                <span>Robotics: {profile.robotics_level}</span>
              </div>
            )}
          </div>
          <div className={styles.dropdownDivider} />
          <button className={styles.dropdownItem} onClick={handleSignOut}>
            Sign Out
          </button>
        </div>
      )}
    </div>
  );
}

// Wrapper component that handles SSR
export function UserMenu(): JSX.Element {
  return (
    <BrowserOnly fallback={<div className={styles.userMenuLoading}>...</div>}>
      {() => <UserMenuContent />}
    </BrowserOnly>
  );
}

export default UserMenu;
