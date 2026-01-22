# Quickstart: Physical AI & Humanoid Robotics Textbook

**Date**: 2025-01-21 | **Plan**: [plan.md](./plan.md)

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Node.js | 18.x or 20.x | Docusaurus frontend |
| Python | 3.11+ | FastAPI backend |
| Git | 2.x | Version control |

### Required Accounts (Free Tier)

| Service | Sign Up | Purpose |
|---------|---------|---------|
| OpenAI | https://platform.openai.com | Embeddings + LLM |
| Qdrant Cloud | https://cloud.qdrant.io | Vector database |
| Neon | https://neon.tech | Postgres database |
| Vercel | https://vercel.com | Deployment |
| GitHub | https://github.com | Repository + Pages |

---

## Local Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/physical-ai-textbook.git
cd physical-ai-textbook
```

### 2. Environment Configuration

Create `.env` file in repository root:

```env
# OpenAI
OPENAI_API_KEY=sk-your-api-key

# Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.cloud
QDRANT_API_KEY=your-qdrant-api-key

# Neon Postgres
DATABASE_URL=postgresql://user:pass@ep-xxx.region.aws.neon.tech/dbname?sslmode=require

# Better-Auth
BETTER_AUTH_SECRET=generate-a-random-32-char-string
BETTER_AUTH_URL=http://localhost:3000

# Development
NODE_ENV=development
```

### 3. Install Dependencies

```bash
# Frontend (Docusaurus)
cd book
npm install
cd ..

# Backend (FastAPI)
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

### 4. Database Setup

Run migrations for Neon Postgres:

```bash
cd backend
alembic upgrade head
cd ..
```

### 5. Index Content to Qdrant

Index textbook content for RAG (run after adding chapters):

```bash
cd backend
python scripts/index_content.py --source ../book/docs
cd ..
```

### 6. Start Development Servers

**Terminal 1 - Frontend:**
```bash
cd book
npm run start
# Runs on http://localhost:3000
```

**Terminal 2 - Backend:**
```bash
cd backend
source .venv/bin/activate
uvicorn src.main:app --reload --port 8000
# Runs on http://localhost:8000
```

---

## Quick Verification

### Test Frontend
1. Open http://localhost:3000
2. Navigate to any chapter
3. Verify code blocks have syntax highlighting
4. Verify sidebar navigation works

### Test Backend
```bash
# Health check
curl http://localhost:8000/health

# Search endpoint
curl "http://localhost:8000/api/v1/search?q=ROS%202"

# Create chat session
curl -X POST http://localhost:8000/api/v1/chat/sessions \
  -H "Content-Type: application/json"
```

### Test RAG Chatbot
```bash
# Send message (replace SESSION_ID with actual ID from above)
curl -X POST http://localhost:8000/api/v1/chat/sessions/SESSION_ID/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "What is ROS 2?"}'
```

---

## Deployment

### Vercel Deployment (Recommended)

1. Connect GitHub repository to Vercel
2. Configure environment variables in Vercel dashboard
3. Deploy triggers automatically on push to main

**Build Settings:**
- Framework Preset: Other
- Build Command: `cd book && npm run build`
- Output Directory: `book/build`

### Environment Variables for Production

Set these in Vercel dashboard:

```
OPENAI_API_KEY=sk-...
QDRANT_URL=https://...
QDRANT_API_KEY=...
DATABASE_URL=postgresql://...
BETTER_AUTH_SECRET=...
BETTER_AUTH_URL=https://your-domain.vercel.app
```

---

## Project Structure Reference

```
.
├── book/                    # Docusaurus frontend
│   ├── docs/               # Chapter content (Markdown)
│   ├── src/
│   │   ├── components/    # React components
│   │   └── css/           # Styles
│   └── docusaurus.config.ts
├── backend/                 # FastAPI backend
│   ├── src/
│   │   ├── api/           # Route handlers
│   │   ├── services/      # Business logic
│   │   ├── models/        # SQLAlchemy models
│   │   └── db/            # Database clients
│   ├── scripts/           # CLI utilities
│   └── tests/             # Test suites
├── scripts/                 # Build scripts
├── specs/                   # Feature specifications
└── .env                     # Local config (not committed)
```

---

## Common Tasks

### Add a New Chapter

1. Create file: `book/docs/part-{N}-{name}/{NN}-{slug}.md`
2. Add frontmatter:
   ```yaml
   ---
   id: ch-{part}-{num}
   title: Chapter Title
   difficulty: beginner
   estimated_time: 25
   prerequisites: []
   ---
   ```
3. Write content in Markdown
4. Update `book/sidebars.ts` to include new chapter
5. Re-index content: `python backend/scripts/index_content.py`

### Test Code Blocks

Extract and test all code blocks:

```bash
python scripts/extract-code-blocks.py --source book/docs --output /tmp/code-tests
cd /tmp/code-tests
# Run tests for each language
```

### Update Dependencies

```bash
# Frontend
cd book && npm update && npm audit fix

# Backend
cd backend && pip-compile requirements.in && pip install -r requirements.txt
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Docusaurus build fails | Check Node version (18+), run `npm ci` |
| Backend won't start | Verify Python 3.11+, check `.env` file exists |
| Qdrant connection fails | Verify QDRANT_URL and API key |
| Database errors | Run `alembic upgrade head` |
| Chatbot returns empty | Re-index content after adding chapters |
| Auth not working | Verify BETTER_AUTH_SECRET is set |

---

## Next Steps

1. **Content Creation**: Start writing chapters in `book/docs/`
2. **Testing**: Add integration tests in `backend/tests/`
3. **Styling**: Customize theme in `book/src/css/`
4. **Deployment**: Set up CI/CD with GitHub Actions
