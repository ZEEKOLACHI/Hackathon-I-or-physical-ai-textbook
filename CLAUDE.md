# Physical AI & Humanoid Robotics Textbook - Development Guide

> **Constitution**: `.specify/memory/constitution.md` (v1.0.0)
> **Branch**: `001-physical-ai-textbook`

## Project Overview

An AI-native educational platform for Physical AI and Humanoid Robotics featuring:
- 21-chapter interactive textbook (Docusaurus)
- RAG-powered chatbot (FastAPI + Qdrant + OpenAI)
- User authentication with Better-Auth
- Content personalization based on user background
- Urdu translation support

## Core Principles (from Constitution)

### I. Content-First Development
- All code examples MUST be runnable without modification
- Verify against official docs (ROS 2, NVIDIA Isaac, Gazebo)
- Progressive complexity within each chapter

### II. Simulation-First Pedagogy
- Every practical section MUST include Gazebo or Isaac Sim
- Physical hardware is OPTIONAL, never prerequisite
- Simulation democratizes learning regardless of budget

### III. RAG-Native Architecture
- Maximum chunk size: 512 tokens with 50-token overlap
- Every code block MUST have descriptive preceding paragraph
- Cross-references use explicit chapter/section identifiers

### IV. Incremental Deployability
- Base book → Chatbot → Auth → Personalization → Translation
- Each feature independently deployable and valuable
- Rollback safety: any feature can be disabled

## Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | Docusaurus 3.x, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.11+ |
| Vector DB | Qdrant Cloud (Free Tier) |
| Relational DB | Neon Serverless Postgres |
| LLM | OpenAI Agents SDK, text-embedding-3-small |
| Auth | Better-Auth |
| Hosting | Vercel / GitHub Pages |

## Project Structure

```
book/                    # Docusaurus frontend
  docs/                  # 21 chapters in 7 parts
  src/components/        # React components
  src/theme/             # Theme customizations

backend/                 # FastAPI backend
  src/api/routes/        # API endpoints
  src/services/          # Business logic
  src/models/            # SQLAlchemy models
  src/db/                # Database clients
  alembic/               # Migrations

specs/001-physical-ai-textbook/
  spec.md                # Feature specification
  plan.md                # Architecture plan
  tasks.md               # Implementation tasks

.specify/memory/
  constitution.md        # Project principles (authoritative)
```

## Code Standards

### Python (Backend)
- PEP 8 compliant
- Type hints required for public functions
- Ruff + Black for formatting
- No hardcoded secrets - use `.env`

### TypeScript (Frontend)
- Strict mode enabled
- Explicit types for exports
- ESLint + Prettier for formatting

### Markdown (Content)
- CommonMark specification
- Frontmatter with: id, title, difficulty, estimated_time, prerequisites
- Code blocks must specify language

## Performance Budgets

| Metric | Target |
|--------|--------|
| Page Load | < 3 seconds (3G) |
| RAG Query | < 5 seconds (excluding LLM) |
| JS Bundle | < 500KB gzipped |
| Build Time | < 5 minutes |

## Quality Gates

Before any PR:
- [ ] Linting passes (ESLint, Ruff)
- [ ] Type checking passes (tsc, mypy)
- [ ] Build succeeds without warnings
- [ ] No hardcoded secrets
- [ ] Code examples tested

## Task Status

### Completed
- Phase 1: Setup (T001-T007)
- Phase 2: Foundational (T008-T022)
- Phase 3: User Story 1 - Browse Content (T023-T038)
- Phase 4: User Story 2 - RAG Chatbot (T039-T057)
- Phase 5: User Story 3 - Authentication (T060-T073)
- Phase 6: User Story 4 - Personalization (T074-T082)
- Phase 7: User Story 5 - Translation (T083-T089)
- Phase 8: Polish (T090-T095, T097)

### Pending
- T058: Run content indexing script
- T059: Verify chatbot answers with citations
- T096: Validate <10s chatbot response
- T098: Run quickstart.md verification
- T099: Final Vercel deployment

## Key Files Reference

| Purpose | Path |
|---------|------|
| API Entry | `backend/src/main.py` |
| DB Config | `backend/src/config.py` |
| Chat Routes | `backend/src/api/routes/chat.py` |
| RAG Service | `backend/src/services/rag_service.py` |
| Docusaurus Config | `book/docusaurus.config.ts` |
| Sidebar | `book/sidebars.ts` |
| ChatBot Component | `book/src/components/ChatBot/` |
| Auth Provider | `book/src/components/Auth/AuthProvider.tsx` |

## Environment Variables

Required in `.env` (see `.env.example`):
```
DATABASE_URL=           # Neon Postgres connection
QDRANT_URL=             # Qdrant Cloud endpoint
QDRANT_API_KEY=         # Qdrant API key
OPENAI_API_KEY=         # OpenAI API key
BETTER_AUTH_SECRET=     # Auth session secret
```

## Commit Convention

```
<type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, test, chore
Scope: chapter-XX, chatbot, auth, frontend, backend
```

## Resources

- [Spec](specs/001-physical-ai-textbook/spec.md) - Feature requirements
- [Plan](specs/001-physical-ai-textbook/plan.md) - Architecture decisions
- [Tasks](specs/001-physical-ai-textbook/tasks.md) - Implementation checklist
- [Quickstart](specs/001-physical-ai-textbook/quickstart.md) - Setup guide
