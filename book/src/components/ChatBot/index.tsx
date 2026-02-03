/**
 * ChatBot component - Stateless Q&A chat interface
 */

import React, { useState, useEffect, useRef } from 'react';
import type { ChatMessage as ChatMessageType, Citation } from '../../services/api';
import { chatApi } from '../../services/api';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import styles from './ChatBot.module.css';

interface ChatBotProps {
  contextChapter?: string;
  onCitationClick?: (citation: Citation) => void;
}

export function ChatBot({ contextChapter, onCitationClick }: ChatBotProps): JSX.Element {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedText, setSelectedText] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle sending a message (stateless - no session required)
  const handleSend = async (content: string) => {
    setError(null);
    setIsLoading(true);

    // Add user message optimistically
    const userMessage: ChatMessageType = {
      id: `user-${Date.now()}`,
      role: 'user',
      content,
      citations: [],
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);

    try {
      // Use stateless ask endpoint
      const response = await chatApi.ask(content, selectedText || undefined);

      // Add assistant response
      const assistantMessage: ChatMessageType = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: response.answer,
        citations: response.citations,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMessage]);

      // Clear selection
      setSelectedText(null);
    } catch (err) {
      console.error('Failed to send message:', err);
      setError('Failed to get response. Please try again.');
      // Remove optimistic user message on error
      setMessages((prev) => prev.filter((m) => m.id !== userMessage.id));
    } finally {
      setIsLoading(false);
    }
  };

  // Handle text selection for "Ask about selection" feature
  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      if (selection && selection.toString().trim().length > 10) {
        setSelectedText(selection.toString().trim());
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => document.removeEventListener('mouseup', handleSelection);
  }, []);

  // Handle citation click
  const handleCitationClick = (citation: Citation) => {
    if (onCitationClick) {
      onCitationClick(citation);
    } else {
      // Default behavior: navigate to chapter
      const link = `/docs/${citation.chapter_id.replace('ch-', 'part-').replace(/-(\d+)$/, '/$1')}#${citation.section_id}`;
      window.location.href = link;
    }
  };

  // Toggle chat open/closed
  const toggleChat = () => {
    setIsOpen((prev) => !prev);
  };

  return (
    <div className={styles.chatBotContainer}>
      {/* Chat toggle button */}
      <button
        className={`${styles.toggleButton} ${isOpen ? styles.open : ''}`}
        onClick={toggleChat}
        aria-label={isOpen ? 'Close chat' : 'Open chat'}
      >
        {isOpen ? (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        ) : (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        )}
      </button>

      {/* Chat panel */}
      {isOpen && (
        <div className={styles.chatPanel} ref={chatContainerRef}>
          <div className={styles.chatHeader}>
            <h3>AI Assistant</h3>
            <span className={styles.headerSubtitle}>Ask questions about the textbook</span>
          </div>

          <div className={styles.messagesContainer}>
            {messages.length === 0 && (
              <div className={styles.welcomeMessage}>
                <p>Welcome! I can help you understand concepts from the Physical AI & Humanoid Robotics textbook.</p>
                <p>Try asking about:</p>
                <ul>
                  <li>ROS 2 fundamentals</li>
                  <li>Robot perception</li>
                  <li>Motion planning</li>
                  <li>Machine learning for robotics</li>
                </ul>
              </div>
            )}

            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                onCitationClick={handleCitationClick}
              />
            ))}

            {isLoading && (
              <div className={styles.loadingIndicator}>
                <span className={styles.dot} />
                <span className={styles.dot} />
                <span className={styles.dot} />
              </div>
            )}

            {error && (
              <div className={styles.errorMessage}>
                {error}
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <ChatInput
            onSend={handleSend}
            disabled={isLoading}
            selectedText={selectedText || undefined}
            onClearSelection={() => setSelectedText(null)}
          />
        </div>
      )}
    </div>
  );
}

export default ChatBot;
