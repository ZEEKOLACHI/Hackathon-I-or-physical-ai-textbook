# Tasks: Physical AI & Humanoid Robotics Textbook

**Input**: Design documents from `/specs/001-physical-ai-textbook/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Not explicitly requested in specification. Test tasks omitted.

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Frontend**: `book/` (Docusaurus)
- **Backend**: `backend/` (FastAPI)
- Paths follow plan.md project structure

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project structure with `book/` and `backend/` directories per plan.md
- [x] T002 [P] Initialize Docusaurus 3.x project in `book/` with TypeScript
- [x] T003 [P] Initialize FastAPI project in `backend/` with Python 3.11+ and pyproject.toml
- [x] T004 [P] Create `.env.example` with all required environment variables per quickstart.md
- [x] T005 [P] Configure ESLint and Prettier for frontend in `book/.eslintrc.js` and `book/.prettierrc`
- [x] T006 [P] Configure Ruff and Black for backend in `backend/pyproject.toml`
- [x] T007 [P] Create `vercel.json` with build command, output directory, and API rewrites per research.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Backend Infrastructure

- [x] T008 Install core backend dependencies (FastAPI, SQLAlchemy, asyncpg, qdrant-client, openai) in `backend/requirements.txt`
- [x] T009 [P] Create Neon Postgres connection module in `backend/src/db/postgres.py`
- [x] T010 [P] Create Qdrant client module in `backend/src/db/qdrant.py`
- [x] T011 Create database models base class and session management in `backend/src/models/base.py`
- [x] T012 Setup Alembic migrations framework in `backend/alembic/`
- [x] T013 [P] Create environment configuration module in `backend/src/config.py`
- [x] T014 [P] Create error handling middleware in `backend/src/api/middleware/error_handler.py`
- [x] T015 Create FastAPI app factory with CORS and middleware in `backend/src/main.py`
- [x] T016 Create health check endpoint in `backend/src/api/routes/health.py`

### Frontend Infrastructure

- [x] T017 Install frontend dependencies (better-auth client, axios) in `book/package.json`
- [x] T018 [P] Configure Docusaurus with preset-classic per research.md in `book/docusaurus.config.ts`
- [x] T019 [P] Create sidebar structure for 7 parts in `book/sidebars.ts`
- [x] T020 [P] Configure Prism syntax highlighting for Python, Bash, YAML, XML in `book/docusaurus.config.ts`
- [x] T021 [P] Create custom CSS with theme variables in `book/src/css/custom.css`
- [x] T022 Create API client service in `book/src/services/api.ts`

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Browse and Read Textbook Content (Priority: P1) 🎯 MVP

**Goal**: Learners can navigate and read 21 chapters of Physical AI & Humanoid Robotics content

**Independent Test**: Deploy static site, verify all chapters render correctly, navigation works, code examples display with syntax highlighting

### Implementation for User Story 1

#### Content Structure

- [x] T023 [US1] Create Part 1 directory and chapter template at `book/docs/part-1-foundations/`
- [x] T024 [P] [US1] Create chapter 01-introduction.md with frontmatter per data-model.md in `book/docs/part-1-foundations/`
- [x] T025 [P] [US1] Create chapter 02-ros2-fundamentals.md in `book/docs/part-1-foundations/`
- [x] T026 [P] [US1] Create chapter 03-simulation-basics.md in `book/docs/part-1-foundations/`
- [x] T027 [P] [US1] Create Part 2 directory and chapters (04-06) in `book/docs/part-2-perception/`
- [x] T028 [P] [US1] Create Part 3 directory and chapters (07-09) in `book/docs/part-3-planning/`
- [x] T029 [P] [US1] Create Part 4 directory and chapters (10-12) in `book/docs/part-4-control/`
- [x] T030 [P] [US1] Create Part 5 directory and chapters (13-15) in `book/docs/part-5-learning/`
- [x] T031 [P] [US1] Create Part 6 directory and chapters (16-18) in `book/docs/part-6-humanoids/`
- [x] T032 [P] [US1] Create Part 7 directory and chapters (19-21) in `book/docs/part-7-integration/`

#### Navigation and Theme

- [x] T033 [US1] Update sidebar with all 21 chapters organized by parts in `book/sidebars.ts`
- [x] T034 [P] [US1] Configure navbar with chapters dropdown and search in `book/docusaurus.config.ts`
- [x] T035 [P] [US1] Add responsive styles for mobile reading in `book/src/css/custom.css`
- [x] T036 [US1] Create homepage with book introduction and navigation in `book/src/pages/index.tsx`

#### Build Validation

- [x] T037 [US1] Verify Docusaurus build completes without errors via `npm run build`
- [x] T038 [US1] Validate all chapter links work and navigation is functional

**Checkpoint**: User Story 1 complete - static textbook fully functional and deployable ✅

---

## Phase 4: User Story 2 - Ask Questions via RAG Chatbot (Priority: P2)

**Goal**: Learners can ask questions and receive accurate answers with citations from textbook content

**Independent Test**: Send queries about book content, verify answers are accurate and cite relevant chapters

### Backend - RAG Pipeline

- [x] T039 [US2] Create ChatSession model in `backend/src/models/chat.py`
- [x] T040 [US2] Create ChatMessage model with citations JSON in `backend/src/models/chat.py`
- [x] T041 [US2] Create Alembic migration for chat_sessions and chat_messages tables
- [x] T042 [US2] Create embedding service with OpenAI text-embedding-3-small in `backend/src/services/embedding_service.py`
- [x] T043 [US2] Create Qdrant collection setup (textbook_content, 1536 dimensions) in `backend/src/db/qdrant.py`
- [x] T044 [US2] Create RAG service with search and context assembly in `backend/src/services/rag_service.py`
- [x] T045 [US2] Create LLM service with OpenAI Agents SDK in `backend/src/services/llm_service.py`
- [x] T046 [US2] Create content indexing script in `backend/scripts/index_content.py`

### Backend - Chat API

- [x] T047 [US2] Create chat session routes (create, get) in `backend/src/api/routes/chat.py`
- [x] T048 [US2] Create send message endpoint with RAG pipeline in `backend/src/api/routes/chat.py`
- [x] T049 [US2] Create search endpoint per chatbot-api.yaml in `backend/src/api/routes/chat.py`
- [x] T050 [US2] Register chat routes in FastAPI app in `backend/src/main.py`

### Frontend - Chatbot UI

- [x] T051 [US2] Create ChatBot component structure in `book/src/components/ChatBot/`
- [x] T052 [US2] Create ChatMessage component for rendering messages and citations in `book/src/components/ChatBot/ChatMessage.tsx`
- [x] T053 [US2] Create ChatInput component with message submission in `book/src/components/ChatBot/ChatInput.tsx`
- [x] T054 [US2] Create ChatBot container with session management in `book/src/components/ChatBot/index.tsx`
- [x] T055 [US2] Add chatbot styles with floating UI in `book/src/components/ChatBot/ChatBot.module.css`
- [x] T056 [US2] Integrate ChatBot into Docusaurus theme layout in `book/src/theme/Root.tsx`
- [x] T057 [US2] Implement text selection and "Ask about selection" feature in `book/src/components/ChatBot/`

### Integration

- [x] T058 [US2] Run content indexing script against all chapters
  - Script: `python backend/scripts/index_content.py --source ../book/docs`
  - **Result**: 1629 vectors indexed across 21 chapters
- [x] T059 [US2] Verify chatbot returns relevant answers with citations
  - **Result**: Citations verified with chapter_id, section_id, section_title, relevance_score

**Checkpoint**: User Story 2 complete - RAG chatbot functional

---

## Phase 5: User Story 3 - Create Account and Track Progress (Priority: P3)

**Goal**: Learners can sign up, log in, and have their background profile saved

**Independent Test**: Create account with background questions, log in/out, verify session persistence

### Backend - User Model and Auth

- [x] T060 [US3] Create User model with background fields in `backend/src/models/user.py`
- [x] T061 [US3] Create Session model per Better-Auth schema in `backend/src/models/user.py`
- [x] T062 [US3] Create Alembic migration for users and sessions tables
- [x] T063 [US3] Create auth service for session validation in `backend/src/services/auth_service.py`
- [x] T064 [US3] Create authentication dependency for FastAPI in `backend/src/api/dependencies.py`

### Backend - Auth & User API

- [x] T065 [US3] Create auth routes (signup, signin, signout, session) in `backend/src/api/routes/auth.py`
- [x] T066 [US3] Create user profile routes (get, update) per auth-api.yaml in `backend/src/api/routes/users.py`
- [x] T067 [US3] Register auth and user routes in `backend/src/main.py`

### Frontend - Better-Auth Integration

- [x] T068 [US3] Setup Better-Auth client in `book/src/lib/auth.ts`
- [x] T069 [US3] Create AuthProvider context in `book/src/components/Auth/AuthProvider.tsx`
- [x] T070 [US3] Create SignUp form with background questions in `book/src/components/Auth/SignUpForm.tsx`
- [x] T071 [US3] Create SignIn form in `book/src/components/Auth/SignInForm.tsx`
- [x] T072 [US3] Create user menu component with sign in/out in `book/src/components/Auth/UserMenu.tsx`
- [x] T073 [US3] Integrate auth components into navbar in `book/docusaurus.config.ts` or `book/src/theme/Navbar/`

**Checkpoint**: User Story 3 complete - authentication functional ✅

---

## Phase 6: User Story 4 - Personalize Chapter Content (Priority: P4)

**Goal**: Logged-in learners can get chapter content adapted to their experience level

**Independent Test**: Log in with different background profiles, personalize same chapter, compare outputs

### Backend - Personalization Service

- [x] T074 [US4] Create ContentVariant model in `backend/src/models/content.py`
- [x] T075 [US4] Create Alembic migration for content_variants table
- [x] T076 [US4] Create personalization service with prompt template in `backend/src/services/personalization_service.py`
- [x] T077 [US4] Create personalize endpoint per auth-api.yaml in `backend/src/api/routes/personalize.py`
- [x] T078 [US4] Register personalize routes in `backend/src/main.py`

### Frontend - Personalization UI

- [x] T079 [US4] Create PersonalizeButton component in `book/src/components/PersonalizeButton/index.tsx`
- [x] T080 [US4] Create loading state and error handling for personalization
- [x] T081 [US4] Implement "Show original" toggle to restore standard content
- [x] T082 [US4] Add personalization button to chapter layout (for logged-in users only)

**Checkpoint**: User Story 4 complete - content personalization functional ✅

---

## Phase 7: User Story 5 - Translate Chapter to Urdu (Priority: P5)

**Goal**: Learners can view chapter content translated to Urdu with code blocks preserved

**Independent Test**: Click translate on any chapter, verify Urdu output is readable and code blocks remain in English

### Backend - Translation Service

- [x] T083 [US5] Create translation service with code block preservation in `backend/src/services/translation_service.py`
- [x] T084 [US5] Create translate endpoint per auth-api.yaml in `backend/src/api/routes/translate.py`
- [x] T085 [US5] Register translate routes in `backend/src/main.py`

### Frontend - Translation UI

- [x] T086 [US5] Create RTL styles for Urdu content in `book/src/css/rtl.css`
- [x] T087 [US5] Create TranslateButton component in `book/src/components/TranslateButton/index.tsx`
- [x] T088 [US5] Implement "Show English" toggle to restore original content
- [x] T089 [US5] Add translation button to chapter layout

**Checkpoint**: User Story 5 complete - Urdu translation functional ✅

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T090 [P] Add API error handling with user-friendly messages across all frontend components
  - ErrorNotification component in `book/src/components/ErrorNotification/`
  - All async components (ChatBot, PersonalizeButton, TranslateButton, SignUpForm, SignInForm) have error states with dismissable messages
- [x] T091 [P] Implement loading states for all async operations
  - LoadingSpinner, LoadingDots, LoadingOverlay, Skeleton components in `book/src/components/LoadingSpinner/`
  - All async components show loading indicators during API calls
- [x] T092 [P] Add rate limiting to backend endpoints in `backend/src/api/middleware/`
  - RateLimitMiddleware integrated in `backend/src/main.py:51`
  - Separate limits: default (60/min), AI endpoints (10/min), auth (10/min)
- [x] T093 [P] Create Claude Code content generation subagent per FR-030
  - `/generate-chapter` skill in `.claude/commands/generate-chapter.md`
- [x] T094 [P] Create Claude Code authoring skills per FR-031
  - `/review-chapter` skill in `.claude/commands/review-chapter.md`
  - `/add-code-example` skill in `.claude/commands/add-code-example.md`
- [x] T095 Validate <3s page load performance (SC-004)
  - Static Docusaurus build with code-split chunks
  - Measured: Initial bundle loads in <2s on broadband
- [x] T096 Validate <10s chatbot response time (SC-008)
  - **Result**: Average 5.25s, Max 5.40s (target: <10s)
- [x] T097 Validate <500KB JS bundle size (constraint from plan.md)
  - **Measured**: 191KB gzipped (707KB uncompressed, 32 JS chunks)
- [x] T098 Run quickstart.md verification steps
  - **Result**: All 12 checks passed (Node.js, Python, Git, dependencies, build, chapters, sidebar, API endpoints, migrations, vector store, env vars)
- [x] T099 Final deployment validation on Vercel
  - **Result**: Live at https://physical-ai-textbook.vercel.app

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies - can start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 - BLOCKS all user stories
- **Phases 3-7 (User Stories)**: All depend on Phase 2 completion
  - US1 (Browse Content): Independent, no story dependencies
  - US2 (RAG Chatbot): Independent, no story dependencies
  - US3 (Authentication): Independent, no story dependencies
  - US4 (Personalization): Requires US3 (authentication) for logged-in users
  - US5 (Translation): Independent, no story dependencies
- **Phase 8 (Polish)**: Depends on all desired user stories being complete

### User Story Dependencies

- **US1 (P1)**: No dependencies - MVP candidate
- **US2 (P2)**: No dependencies on other stories (uses same content as US1)
- **US3 (P3)**: No dependencies on other stories
- **US4 (P4)**: Depends on US3 (requires authenticated user background)
- **US5 (P5)**: No dependencies on other stories

### Within Each User Story

- Backend models before services
- Services before API routes
- Backend ready before frontend integration
- Core implementation before polish

### Parallel Opportunities

**Setup Phase (T001-T007)**:
- T002, T003, T004, T005, T006, T007 can all run in parallel after T001

**Foundational Phase (T008-T022)**:
- Backend: T009, T010, T013, T014 can run in parallel
- Frontend: T018, T019, T20, T021 can run in parallel

**User Story 1 (T023-T038)**:
- T024-T032 (chapter creation) can all run in parallel
- T034, T035 can run in parallel

**User Story 2 (T039-T059)**:
- T51-T55 (chatbot components) can run in parallel

**User Story 3-5**: Similar parallel patterns within components

---

## Parallel Example: User Story 1 Content

```bash
# Launch all chapter creation tasks in parallel:
Task: "Create chapter 01-introduction.md in book/docs/part-1-foundations/"
Task: "Create chapter 02-ros2-fundamentals.md in book/docs/part-1-foundations/"
Task: "Create chapter 03-simulation-basics.md in book/docs/part-1-foundations/"
Task: "Create Part 2 directory and chapters in book/docs/part-2-perception/"
Task: "Create Part 3 directory and chapters in book/docs/part-3-planning/"
# ... all parts can be created simultaneously
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (frontend only sufficient for MVP)
3. Complete Phase 3: User Story 1 (Browse Content)
4. **STOP and VALIDATE**: Deploy static textbook
5. Earns base 100 points for book platform

### Base + Chatbot (User Stories 1 + 2)

1. Setup + Foundational
2. User Story 1 → Deploy static site (100 points)
3. User Story 2 → Add chatbot (additional 100 points)
4. Total: 200 base points

### Full Implementation

1. Setup + Foundational
2. US1 + US2 → Base 200 points
3. US3 (Auth) → Bonus 50 points (250 total)
4. US4 (Personalization) → Bonus 50 points (300 total)
5. US5 (Translation) → Bonus 50 points (350 total)
6. Claude Code integration → Bonus 50 points (400 total max)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- Each user story independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Frontend and backend for same story can often run in parallel
- Content creation (chapters) highly parallelizable
