/**
 * Custom DocItem Layout wrapper
 * Adds PersonalizeButton and TranslateButton to chapter documentation pages
 */

import React, { useState, useCallback } from 'react';
import Layout from '@theme-original/DocItem/Layout';
import type LayoutType from '@theme/DocItem/Layout';
import type { WrapperProps } from '@docusaurus/types';
import { useDoc } from '@docusaurus/plugin-content-docs/client';
import { PersonalizeButton } from '../../../components/PersonalizeButton';
import { TranslateButton } from '../../../components/TranslateButton';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import '../../../css/rtl.css';

type Props = WrapperProps<typeof LayoutType>;

type ContentMode = 'original' | 'personalized' | 'translated';

interface ContentState {
  mode: ContentMode;
  personalizedContent: string | null;
  translatedContent: string | null;
  isRtl: boolean;
}

export default function LayoutWrapper(props: Props): JSX.Element {
  const { frontMatter } = useDoc();
  const [contentState, setContentState] = useState<ContentState>({
    mode: 'original',
    personalizedContent: null,
    translatedContent: null,
    isRtl: false,
  });

  // Extract chapter ID from frontmatter (e.g., "ch-1-01")
  const chapterId = (frontMatter as { id?: string }).id;
  const isChapter = chapterId && chapterId.startsWith('ch-');

  // Personalization handlers
  const handlePersonalized = useCallback((content: string) => {
    setContentState((prev) => ({
      ...prev,
      mode: 'personalized',
      personalizedContent: content,
    }));
  }, []);

  const handleShowOriginalFromPersonalized = useCallback(() => {
    setContentState((prev) => ({
      ...prev,
      mode: 'original',
    }));
  }, []);

  // Translation handlers
  const handleTranslated = useCallback((content: string, isRtl: boolean) => {
    setContentState((prev) => ({
      ...prev,
      mode: 'translated',
      translatedContent: content,
      isRtl,
    }));
  }, []);

  const handleShowOriginalFromTranslated = useCallback(() => {
    setContentState((prev) => ({
      ...prev,
      mode: 'original',
    }));
  }, []);

  // If not a chapter page, just render the original layout
  if (!isChapter) {
    return <Layout {...props} />;
  }

  const { mode, personalizedContent, translatedContent, isRtl } = contentState;

  return (
    <>
      {/* Action buttons at the top of chapter content */}
      <div style={{
        display: 'flex',
        gap: '12px',
        marginBottom: '1rem',
        flexWrap: 'wrap',
        alignItems: 'flex-start',
      }}>
        <PersonalizeButton
          chapterId={chapterId}
          onPersonalized={handlePersonalized}
          onShowOriginal={handleShowOriginalFromPersonalized}
        />
        <TranslateButton
          chapterId={chapterId}
          onTranslated={handleTranslated}
          onShowOriginal={handleShowOriginalFromTranslated}
        />
      </div>

      {/* Render content based on mode */}
      {mode === 'personalized' && personalizedContent ? (
        <PersonalizedContentView
          content={personalizedContent}
          onShowOriginal={handleShowOriginalFromPersonalized}
        />
      ) : mode === 'translated' && translatedContent ? (
        <TranslatedContentView
          content={translatedContent}
          isRtl={isRtl}
          onShowOriginal={handleShowOriginalFromTranslated}
        />
      ) : (
        <Layout {...props} />
      )}
    </>
  );
}

/**
 * Component to render personalized markdown content
 */
interface PersonalizedContentViewProps {
  content: string;
  onShowOriginal: () => void;
}

function PersonalizedContentView({
  content,
  onShowOriginal,
}: PersonalizedContentViewProps): JSX.Element {
  const contentWithoutFrontmatter = content.replace(/^---[\s\S]*?---\n*/, '');

  return (
    <div className="personalized-content-wrapper">
      <div
        style={{
          padding: '12px 16px',
          marginBottom: '16px',
          background: 'var(--ifm-color-success-lightest)',
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          border: '1px solid var(--ifm-color-success-light)',
          flexWrap: 'wrap',
          gap: '8px',
        }}
      >
        <span
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontWeight: 500,
            color: 'var(--ifm-color-success-darkest)',
          }}
        >
          <span style={{ fontSize: '18px' }}>&#10024;</span>
          Viewing personalized content adapted for your experience level
        </span>
        <button
          onClick={onShowOriginal}
          style={{
            padding: '6px 12px',
            background: 'white',
            border: '1px solid var(--ifm-color-emphasis-300)',
            borderRadius: '16px',
            cursor: 'pointer',
            fontSize: '13px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <span>&#128196;</span>
          Show Original
        </button>
      </div>

      <article className="markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {contentWithoutFrontmatter}
        </ReactMarkdown>
      </article>
    </div>
  );
}

/**
 * Component to render translated markdown content with RTL support
 */
interface TranslatedContentViewProps {
  content: string;
  isRtl: boolean;
  onShowOriginal: () => void;
}

function TranslatedContentView({
  content,
  isRtl,
  onShowOriginal,
}: TranslatedContentViewProps): JSX.Element {
  const contentWithoutFrontmatter = content.replace(/^---[\s\S]*?---\n*/, '');

  return (
    <div className="translated-content-wrapper">
      {/* Translation banner */}
      <div className="translation-banner">
        <div className="translation-banner-text">
          <span className="translation-banner-icon">&#127760;</span>
          <span className="translation-banner-label">Viewing Urdu translation</span>
          <span className="translation-banner-urdu">اردو ترجمہ</span>
        </div>
        <button className="show-english-btn" onClick={onShowOriginal}>
          <span>&#127468;&#127463;</span>
          <span>Show English</span>
        </button>
      </div>

      {/* Translated content with RTL support */}
      <article className={`markdown ${isRtl ? 'rtl-content' : ''}`}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {contentWithoutFrontmatter}
        </ReactMarkdown>
      </article>
    </div>
  );
}
