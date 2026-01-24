/**
 * Root component wrapper for Docusaurus
 * Adds global components like ChatBot and AuthProvider
 */

import React from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import { AuthProvider } from '../components/Auth';

interface RootProps {
  children: React.ReactNode;
}

// Lazy load ChatBot to avoid SSR issues
const ChatBotLazy = React.lazy(() => import('../components/ChatBot'));

export default function Root({ children }: RootProps): JSX.Element {
  return (
    <AuthProvider>
      {children}
      <BrowserOnly fallback={null}>
        {() => (
          <React.Suspense fallback={null}>
            <ChatBotLazy />
          </React.Suspense>
        )}
      </BrowserOnly>
    </AuthProvider>
  );
}
