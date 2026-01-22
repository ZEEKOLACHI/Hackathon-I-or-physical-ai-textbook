/**
 * SignUpForm - User registration form with background questions
 */

import React, { useState } from 'react';
import { useAuth } from './AuthProvider';
import styles from './Auth.module.css';

interface SignUpFormProps {
  onSuccess?: () => void;
  onSwitchToSignIn?: () => void;
}

const EXPERIENCE_LEVELS = [
  { value: 'none', label: 'No experience' },
  { value: 'beginner', label: 'Beginner' },
  { value: 'intermediate', label: 'Intermediate' },
  { value: 'advanced', label: 'Advanced' },
];

const HARDWARE_OPTIONS = [
  { value: 'simulation_only', label: 'Simulation only' },
  { value: 'raspberry_pi', label: 'Raspberry Pi' },
  { value: 'jetson_nano', label: 'NVIDIA Jetson Nano' },
  { value: 'jetson_orin', label: 'NVIDIA Jetson Orin' },
  { value: 'turtlebot', label: 'TurtleBot' },
  { value: 'custom_robot', label: 'Custom Robot' },
];

export function SignUpForm({ onSuccess, onSwitchToSignIn }: SignUpFormProps): JSX.Element {
  const { signUp } = useAuth();
  const [step, setStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [programmingLevel, setProgrammingLevel] = useState('beginner');
  const [roboticsLevel, setRoboticsLevel] = useState('beginner');
  const [hardwareAvailable, setHardwareAvailable] = useState<string[]>(['simulation_only']);

  const handleHardwareToggle = (value: string) => {
    setHardwareAvailable((prev) =>
      prev.includes(value) ? prev.filter((v) => v !== value) : [...prev, value]
    );
  };

  const handleStep1Submit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    setStep(2);
  };

  const handleStep2Submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      await signUp({
        email,
        password,
        programming_level: programmingLevel as any,
        robotics_level: roboticsLevel as any,
        hardware_available: hardwareAvailable,
      });
      onSuccess?.();
    } catch (err: any) {
      const message = err.response?.data?.message || 'Failed to create account';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.authForm}>
      <h2 className={styles.title}>Create Account</h2>

      {error && <div className={styles.error}>{error}</div>}

      {step === 1 ? (
        <form onSubmit={handleStep1Submit}>
          <div className={styles.field}>
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder="you@example.com"
            />
          </div>

          <div className={styles.field}>
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
              placeholder="At least 8 characters"
            />
          </div>

          <div className={styles.field}>
            <label htmlFor="confirmPassword">Confirm Password</label>
            <input
              id="confirmPassword"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              placeholder="Confirm your password"
            />
          </div>

          <button type="submit" className={styles.submitButton}>
            Continue
          </button>
        </form>
      ) : (
        <form onSubmit={handleStep2Submit}>
          <p className={styles.stepDescription}>
            Help us personalize your learning experience
          </p>

          <div className={styles.field}>
            <label htmlFor="programmingLevel">Programming Experience</label>
            <select
              id="programmingLevel"
              value={programmingLevel}
              onChange={(e) => setProgrammingLevel(e.target.value)}
            >
              {EXPERIENCE_LEVELS.map((level) => (
                <option key={level.value} value={level.value}>
                  {level.label}
                </option>
              ))}
            </select>
          </div>

          <div className={styles.field}>
            <label htmlFor="roboticsLevel">Robotics Experience</label>
            <select
              id="roboticsLevel"
              value={roboticsLevel}
              onChange={(e) => setRoboticsLevel(e.target.value)}
            >
              {EXPERIENCE_LEVELS.map((level) => (
                <option key={level.value} value={level.value}>
                  {level.label}
                </option>
              ))}
            </select>
          </div>

          <div className={styles.field}>
            <label>Available Hardware</label>
            <div className={styles.checkboxGroup}>
              {HARDWARE_OPTIONS.map((option) => (
                <label key={option.value} className={styles.checkbox}>
                  <input
                    type="checkbox"
                    checked={hardwareAvailable.includes(option.value)}
                    onChange={() => handleHardwareToggle(option.value)}
                  />
                  {option.label}
                </label>
              ))}
            </div>
          </div>

          <div className={styles.buttonGroup}>
            <button
              type="button"
              className={styles.backButton}
              onClick={() => setStep(1)}
            >
              Back
            </button>
            <button
              type="submit"
              className={styles.submitButton}
              disabled={isLoading}
            >
              {isLoading ? 'Creating...' : 'Create Account'}
            </button>
          </div>
        </form>
      )}

      <div className={styles.switchPrompt}>
        Already have an account?{' '}
        <button type="button" className={styles.linkButton} onClick={onSwitchToSignIn}>
          Sign in
        </button>
      </div>
    </div>
  );
}

export default SignUpForm;
