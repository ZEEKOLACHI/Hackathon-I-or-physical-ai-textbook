/**
 * ChatInput component for message submission
 */

import React, { useState, useRef, useEffect } from 'react';
import styles from './ChatBot.module.css';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
  selectedText?: string;
  onClearSelection?: () => void;
}

export function ChatInput({
  onSend,
  disabled = false,
  placeholder = 'Ask a question about the textbook...',
  selectedText,
  onClearSelection,
}: ChatInputProps): JSX.Element {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [message]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSend(message.trim());
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form className={styles.inputForm} onSubmit={handleSubmit}>
      {selectedText && (
        <div className={styles.selectionPreview}>
          <span className={styles.selectionLabel}>Selected text:</span>
          <span className={styles.selectionText}>
            {selectedText.length > 100 ? `${selectedText.slice(0, 100)}...` : selectedText}
          </span>
          <button
            type="button"
            className={styles.clearSelection}
            onClick={onClearSelection}
            aria-label="Clear selection"
          >
            &times;
          </button>
        </div>
      )}
      <div className={styles.inputWrapper}>
        <textarea
          ref={textareaRef}
          className={styles.input}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          aria-label="Message input"
        />
        <button
          type="submit"
          className={styles.sendButton}
          disabled={disabled || !message.trim()}
          aria-label="Send message"
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </div>
    </form>
  );
}

export default ChatInput;
