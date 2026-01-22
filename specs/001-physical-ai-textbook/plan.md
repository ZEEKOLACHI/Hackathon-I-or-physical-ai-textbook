# Implementation Plan: Physical AI & Humanoid Robotics Textbook

**Branch**: `001-physical-ai-textbook` | **Date**: 2025-01-21 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-physical-ai-textbook/spec.md`

## Summary

Build an AI-native educational platform for Physical AI and Humanoid Robotics with 21 chapters organized into 7 parts. Core deliverables: Docusaurus static site with integrated RAG chatbot (FastAPI + Qdrant + OpenAI Agents SDK). Bonus features: Better-Auth authentication, content personalization based on learner background, and Urdu translation.

## Technical Context

**Language/Version**: TypeScript 5.x (frontend), Python 3.11+ (backend)
**Primary Dependencies**: Docusaurus 3.x, FastAPI, Qdrant Client, OpenAI Agents SDK, Better-Auth
**Storage**: Qdrant Cloud (vectors), Neon Serverless Postgres (users, sessions)
**Testing**: Vitest (frontend), pytest (backend)
**Target Platform**: Web (GitHub Pages/Vercel for frontend, serverless for backend)
**Project Type**: web (frontend + backend separation)
**Performance Goals**: <3s page load, <10s chatbot response (excluding LLM), <5min build
**Constraints**: Free tier limits (Qdrant, Neon, Vercel), <500KB JS bundle gzipped
**Scale/Scope**: 21 chapters, ~200 sections, ~100 users concurrent (hackathon demo)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Content-First Development | ✅ PASS | P1 is Browse/Read content; code examples verified before deployment |
| II. Simulation-First Pedagogy | ✅ PASS | All practicals use Gazebo/Isaac Sim; no physical hardware required |
| III. Spec-Driven Content Creation | ✅ PASS | Using Spec-Kit Plus workflow; spec exists before plan |
| IV. RAG-Native Architecture | ✅ PASS | Content structured for 512-token chunks with metadata |
| V. Incremental Deployability | ✅ PASS | Milestone order: Static book → Chatbot → Auth → Personalization → Translation |
| VI. Test-Verified Content | ✅ PASS | CI extracts and tests code blocks; Docker configs provided |
| VII. Accessibility & I18n | ✅ PASS | Simple English, translation-ready structure, difficulty tagging |
| VIII. Security & Privacy | ✅ PASS | Better-Auth with OWASP guidelines, env vars for secrets |
| IX. Performance & Efficiency | ✅ PASS | <3s load, <500KB bundle, free tier compliance |
| X. Documentation & Maintainability | ✅ PASS | READMEs, OpenAPI docs, ADRs for decisions |

**Gate Result**: PASS - All 10 principles satisfied. Proceeding to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/001-physical-ai-textbook/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (OpenAPI specs)
│   ├── chatbot-api.yaml
│   └── auth-api.yaml
└── tasks.md             # Phase 2 output (/sp.tasks command)
```

### Source Code (repository root)

```text
book/                              # Docusaurus frontend
├── docs/                          # Markdown chapter content
│   ├── part-1-foundations/
│   │   ├── 01-introduction.md
│   │   ├── 02-ros2-fundamentals.md
│   │   └── 03-simulation-basics.md
│   ├── part-2-perception/
│   ├── part-3-planning/
│   ├── part-4-control/
│   ├── part-5-learning/
│   ├── part-6-humanoids/
│   └── part-7-integration/
├── src/
│   ├── components/
│   │   ├── ChatBot/               # RAG chatbot UI component
│   │   ├── PersonalizeButton/     # Personalization trigger
│   │   └── TranslateButton/       # Translation trigger
│   ├── theme/                     # Docusaurus theme overrides
│   └── services/
│       └── api.ts                 # Backend API client
├── static/
│   └── img/                       # Diagrams and images
├── docusaurus.config.ts
├── sidebars.ts
└── package.json

backend/                           # FastAPI backend
├── src/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── chat.py            # Chatbot endpoints
│   │   │   ├── auth.py            # Authentication endpoints
│   │   │   ├── personalize.py     # Personalization endpoints
│   │   │   └── translate.py       # Translation endpoints
│   │   └── dependencies.py        # FastAPI dependencies
│   ├── services/
│   │   ├── rag_service.py         # RAG pipeline orchestration
│   │   ├── embedding_service.py   # OpenAI embeddings
│   │   ├── llm_service.py         # OpenAI Agents SDK integration
│   │   ├── auth_service.py        # Better-Auth integration
│   │   └── translation_service.py # Urdu translation
│   ├── models/
│   │   ├── user.py                # User entity
│   │   ├── chat.py                # Chat session/message entities
│   │   └── content.py             # Content variant entity
│   └── db/
│       ├── postgres.py            # Neon connection
│       └── qdrant.py              # Qdrant client
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
├── scripts/
│   └── index_content.py           # Content indexing to Qdrant
├── requirements.txt
└── pyproject.toml

scripts/                           # Build and CI scripts
├── extract-code-blocks.py         # Extract code for testing
├── validate-chapters.py           # Verify chapter completeness
└── generate-embeddings.py         # Batch embedding generation
```

**Structure Decision**: Web application (frontend + backend) chosen because:
1. Docusaurus requires Node.js runtime for SSG build
2. RAG backend requires Python for FastAPI + OpenAI SDK
3. Separation enables independent deployment (static frontend, serverless backend)

## Complexity Tracking

> No constitution violations to justify. All principles satisfied with standard architecture.

| Area | Complexity | Justification |
|------|------------|---------------|
| Two-language stack | Medium | Required by hackathon (Docusaurus + FastAPI). Clear separation at API boundary. |
| RAG pipeline | Medium | Standard pattern: embed → retrieve → generate. No custom components. |
| Multi-feature scope | High | Mitigated by priority ordering and incremental deployment milestones. |
