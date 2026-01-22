/**
 * ChatMessage component for rendering individual messages
 */

import React from 'react';
import type { ChatMessage as ChatMessageType, Citation } from '../../services/api';
import styles from './ChatBot.module.css';

interface ChatMessageProps {
  message: ChatMessageType;
  onCitationClick?: (citation: Citation) => void;
}

export function ChatMessage({ message, onCitationClick }: ChatMessageProps): JSX.Element {
  const isUser = message.role === 'user';

  return (
    <div className={`${styles.message} ${isUser ? styles.userMessage : styles.assistantMessage}`}>
      <div className={styles.messageContent}>
        <div className={styles.messageRole}>
          {isUser ? 'You' : 'AI Assistant'}
        </div>
        <div className={styles.messageText}>
          {message.content.split('\n').map((line, i) => (
            <React.Fragment key={i}>
              {line}
              {i < message.content.split('\n').length - 1 && <br />}
            </React.Fragment>
          ))}
        </div>
        {!isUser && message.citations && message.citations.length > 0 && (
          <div className={styles.citations}>
            <div className={styles.citationsLabel}>Sources:</div>
            <div className={styles.citationsList}>
              {message.citations.map((citation, index) => (
                <button
                  key={index}
                  className={styles.citationChip}
                  onClick={() => onCitationClick?.(citation)}
                  title={`Relevance: ${Math.round(citation.relevance_score * 100)}%`}
                >
                  {citation.section_title || citation.chapter_id}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ChatMessage;
