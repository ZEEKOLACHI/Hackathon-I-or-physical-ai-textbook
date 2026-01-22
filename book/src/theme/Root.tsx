/**
 * Root component wrapper for Docusaurus
 * Adds global components like ChatBot and AuthProvider
 */

import React from 'react';
import { AuthProvider } from '../components/Auth';
import { ChatBot } from '../components/ChatBot';

interface RootProps {
  children: React.ReactNode;
}

export default function Root({ children }: RootProps): JSX.Element {
  return (
    <AuthProvider>
      {children}
      <ChatBot />
    </AuthProvider>
  );
}
