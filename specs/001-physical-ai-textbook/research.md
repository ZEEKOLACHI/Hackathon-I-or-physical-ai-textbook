# Research: Physical AI & Humanoid Robotics Textbook

**Date**: 2025-01-21 | **Plan**: [plan.md](./plan.md)

## Research Questions

This document consolidates research findings to resolve all technical unknowns before Phase 1 design.

---

## R1: Docusaurus 3.x Configuration for Technical Books

### Decision
Use Docusaurus 3.7+ with `@docusaurus/preset-classic` and custom plugins for enhanced code blocks.

### Rationale
- Docusaurus 3.x is the hackathon requirement
- Preset-classic provides docs, blog, and pages out of the box
- MDX support enables interactive components within Markdown
- Built-in versioning (if needed for future updates)

### Key Configuration

```typescript
// docusaurus.config.ts
const config: Config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'From Theory to Simulation',
  url: 'https://your-domain.vercel.app',
  baseUrl: '/',

  presets: [
    ['classic', {
      docs: {
        sidebarPath: './sidebars.ts',
        routeBasePath: '/', // Docs at root
        showLastUpdateTime: true,
      },
      theme: {
        customCss: './src/css/custom.css',
      },
    }],
  ],

  themeConfig: {
    navbar: {
      title: 'Physical AI Textbook',
      items: [
        { type: 'docSidebar', sidebarId: 'bookSidebar', position: 'left', label: 'Chapters' },
        { type: 'search', position: 'right' },
      ],
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'yaml', 'xml', 'python'],
    },
  },
};
```

### Sidebar Structure

```typescript
// sidebars.ts
const sidebars = {
  bookSidebar: [
    {
      type: 'category',
      label: 'Part 1: Foundations',
      items: ['part-1-foundations/01-introduction', 'part-1-foundations/02-ros2-fundamentals', 'part-1-foundations/03-simulation-basics'],
    },
    // ... Parts 2-7
  ],
};
```

### Alternatives Considered
1. **VitePress**: Lighter, but less plugin ecosystem. Rejected for hackathon compliance.
2. **Nextra**: Next.js-based. Rejected for complexity and unfamiliarity with team.
3. **GitBook**: Hosted only, no self-hosting option. Rejected.

---

## R2: RAG Pipeline Architecture with Qdrant + OpenAI

### Decision
Use semantic chunking with 512-token windows, OpenAI `text-embedding-3-small` for embeddings, Qdrant Cloud for vector storage, and OpenAI Agents SDK for response generation.

### Rationale
- 512 tokens provides good semantic coherence for technical content
- `text-embedding-3-small` balances cost ($0.02/1M tokens) and quality
- Qdrant Cloud free tier: 1GB storage, sufficient for ~10K chunks
- OpenAI Agents SDK required by hackathon for orchestration

### Architecture

```
User Query → Embed Query → Qdrant Search (top-5) → Context Assembly → LLM Generation → Response
```

### Chunking Strategy

```python
# Content chunking for RAG
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 50  # tokens

def chunk_chapter(markdown: str, chapter_id: str) -> list[Chunk]:
    """Split chapter into semantic chunks with metadata."""
    sections = split_by_headers(markdown)
    chunks = []

    for section in sections:
        if token_count(section.content) <= CHUNK_SIZE:
            chunks.append(Chunk(
                text=section.content,
                metadata={
                    "chapter_id": chapter_id,
                    "section_id": section.id,
                    "section_title": section.title,
                    "has_code": contains_code_block(section.content),
                }
            ))
        else:
            # Split large sections with overlap
            chunks.extend(split_with_overlap(section, CHUNK_SIZE, CHUNK_OVERLAP))

    return chunks
```

### Qdrant Collection Schema

```python
from qdrant_client.models import VectorParams, Distance

collection_config = {
    "name": "textbook_content",
    "vectors": VectorParams(
        size=1536,  # text-embedding-3-small dimension
        distance=Distance.COSINE,
    ),
}

# Payload schema
payload_schema = {
    "chapter_id": "str",
    "section_id": "str",
    "section_title": "str",
    "has_code": "bool",
    "difficulty": "str",  # beginner/intermediate/advanced
}
```

### OpenAI Agents SDK Integration

```python
from openai import OpenAI
from openai.agents import Agent, Tool

def create_rag_agent():
    client = OpenAI()

    return Agent(
        model="gpt-4o-mini",
        instructions="""You are a helpful tutor for Physical AI and Humanoid Robotics.
        Answer questions using ONLY the provided context from the textbook.
        Always cite the chapter and section when providing information.
        If the answer is not in the context, say so clearly.""",
        tools=[
            Tool(
                name="search_textbook",
                description="Search the textbook for relevant content",
                function=search_qdrant,
            )
        ],
    )
```

### Alternatives Considered
1. **LangChain**: More features but adds complexity. Rejected for simplicity.
2. **LlamaIndex**: Good RAG support but OpenAI SDK is required.
3. **Pinecone**: Higher free tier limits but Qdrant is required by hackathon.

---

## R3: Better-Auth Integration with FastAPI

### Decision
Use Better-Auth TypeScript library on frontend with FastAPI backend handling session validation via shared Neon Postgres database.

### Rationale
- Better-Auth is required for bonus points
- Better-Auth is primarily a TypeScript library
- FastAPI can validate sessions by reading the shared session store

### Architecture Pattern

```
Frontend (Docusaurus + Better-Auth)
    ↓
Neon Postgres (Shared Session Store)
    ↓
Backend (FastAPI - Session Validation)
```

### Better-Auth Frontend Setup

```typescript
// book/src/lib/auth.ts
import { createAuthClient } from "better-auth/client";

export const authClient = createAuthClient({
  baseURL: "/api/auth", // Proxied to backend
});

// Sign up with background questions
export async function signUp(email: string, password: string, background: UserBackground) {
  return authClient.signUp({
    email,
    password,
    additionalData: {
      programming_level: background.programmingLevel,
      robotics_level: background.roboticsLevel,
      hardware_available: background.hardwareAvailable,
    },
  });
}
```

### FastAPI Session Validation

```python
# backend/src/services/auth_service.py
from sqlalchemy import select
from src.db.postgres import get_session

async def validate_session(session_token: str) -> User | None:
    """Validate Better-Auth session token from shared database."""
    async with get_session() as db:
        result = await db.execute(
            select(Session).where(Session.token == session_token)
        )
        session = result.scalar_one_or_none()

        if session and session.expires_at > datetime.utcnow():
            user = await db.get(User, session.user_id)
            return user
    return None
```

### Database Schema (Neon Postgres)

```sql
-- Better-Auth managed tables
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    programming_level VARCHAR(50),
    robotics_level VARCHAR(50),
    hardware_available TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Alternatives Considered
1. **Auth.js (NextAuth)**: Doesn't work with Docusaurus. Rejected.
2. **Clerk/Auth0**: Third-party services; Better-Auth is required.
3. **FastAPI-Users**: Python-only; frontend auth would be custom. Rejected.

---

## R4: Content Personalization Architecture

### Decision
Use LLM-based content adaptation with prompt engineering, generating personalized content on-demand and caching in Postgres.

### Rationale
- LLM can adapt explanation depth, add/remove prerequisites, adjust examples
- On-demand generation avoids pre-generating all variants
- Caching prevents redundant API calls for same user/chapter

### Personalization Prompt Template

```python
PERSONALIZATION_PROMPT = """
You are an expert educational content adapter. Rewrite the following chapter content
for a learner with this background:

Programming Experience: {programming_level}
Robotics Experience: {robotics_level}
Available Hardware: {hardware}

Adaptation Rules:
- For beginners: Add more context, explain prerequisites, use simpler language
- For intermediate: Balance explanation with practical focus
- For advanced: Be concise, skip basics, add advanced tips

IMPORTANT:
- Preserve ALL code blocks exactly as written
- Preserve ALL technical terms (ROS 2, Gazebo, etc.)
- Maintain the same section structure
- Do not change the technical accuracy

Original Content:
{original_content}

Personalized Content:
"""
```

### Caching Strategy

```python
# Content variant storage
class ContentVariant(Base):
    __tablename__ = "content_variants"

    id = Column(UUID, primary_key=True)
    chapter_id = Column(String, nullable=False)
    user_id = Column(UUID, ForeignKey("users.id"))
    variant_type = Column(String)  # "personalized" or "translated"
    variant_key = Column(String)   # e.g., "beginner" or "urdu"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('chapter_id', 'user_id', 'variant_type', 'variant_key'),
    )
```

### Alternatives Considered
1. **Pre-generated variants**: Would require 3x content (beginner/intermediate/advanced). Rejected for storage overhead.
2. **Template-based adaptation**: Limited flexibility. LLM provides better natural adaptation.
3. **Difficulty toggle per section**: More granular but complex UX. Rejected.

---

## R5: Urdu Translation Architecture

### Decision
Use OpenAI GPT-4o for translation with code block preservation via preprocessing markers.

### Rationale
- GPT-4o has strong Urdu capability
- Code blocks must be preserved exactly (never translated)
- Technical terms need transliteration guidance

### Translation Pipeline

```python
def translate_to_urdu(content: str) -> str:
    """Translate chapter content to Urdu, preserving code blocks."""

    # Step 1: Extract and mark code blocks
    code_blocks = []
    def replace_code(match):
        idx = len(code_blocks)
        code_blocks.append(match.group(0))
        return f"[[CODE_BLOCK_{idx}]]"

    marked_content = re.sub(r'```[\s\S]*?```', replace_code, content)

    # Step 2: Translate prose
    translated = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "system",
            "content": """Translate to Urdu. Rules:
            - Preserve [[CODE_BLOCK_N]] markers exactly
            - Keep technical terms in English: ROS 2, Gazebo, NVIDIA Isaac, Python, etc.
            - Add Urdu transliteration in parentheses for key terms: ROS 2 (آر او ایس ٹو)
            - Use formal Urdu appropriate for educational content"""
        }, {
            "role": "user",
            "content": marked_content
        }]
    ).choices[0].message.content

    # Step 3: Restore code blocks
    for idx, code in enumerate(code_blocks):
        translated = translated.replace(f"[[CODE_BLOCK_{idx}]]", code)

    return translated
```

### RTL CSS Support

```css
/* book/src/css/rtl.css */
.translated-urdu {
  direction: rtl;
  text-align: right;
  font-family: 'Noto Nastaliq Urdu', serif;
}

.translated-urdu pre,
.translated-urdu code {
  direction: ltr;
  text-align: left;
}
```

### Alternatives Considered
1. **Google Translate API**: Lower quality for technical content. Rejected.
2. **DeepL**: Doesn't support Urdu. Rejected.
3. **Pre-translated content**: Would require manual translation of 21 chapters. Rejected for time.

---

## R6: Deployment Architecture

### Decision
- Frontend: Vercel (Docusaurus static build)
- Backend: Vercel Serverless Functions (FastAPI via Mangum)
- Database: Neon Serverless Postgres
- Vectors: Qdrant Cloud

### Rationale
- Vercel handles both static and serverless in one platform
- Neon serverless = auto-scaling, no cold starts for Postgres
- Qdrant Cloud free tier sufficient for hackathon scale

### Vercel Configuration

```json
// vercel.json
{
  "buildCommand": "cd book && npm run build",
  "outputDirectory": "book/build",
  "functions": {
    "backend/api/**/*.py": {
      "runtime": "python3.11"
    }
  },
  "rewrites": [
    { "source": "/api/:path*", "destination": "/backend/api/:path*" },
    { "source": "/(.*)", "destination": "/book/build/$1" }
  ]
}
```

### Environment Variables

```env
# .env.example
OPENAI_API_KEY=sk-...
QDRANT_URL=https://xxx.qdrant.cloud
QDRANT_API_KEY=...
NEON_DATABASE_URL=postgresql://...
BETTER_AUTH_SECRET=...
```

### Alternatives Considered
1. **GitHub Pages + Railway**: Two platforms to manage. Rejected for simplicity.
2. **Cloudflare Pages + Workers**: Less familiar, Python Workers are beta.
3. **AWS Lambda + S3**: More configuration, no free tier simplicity.

---

## Summary of Decisions

| Area | Decision | Key Reason |
|------|----------|------------|
| Frontend | Docusaurus 3.7+ on Vercel | Hackathon requirement |
| Backend | FastAPI on Vercel Serverless | Hackathon requirement + unified platform |
| Vector DB | Qdrant Cloud | Hackathon requirement |
| Relational DB | Neon Serverless Postgres | Hackathon requirement |
| Auth | Better-Auth (frontend) + session sharing | Hackathon bonus requirement |
| Embeddings | text-embedding-3-small | Cost-effective, sufficient quality |
| LLM | GPT-4o-mini (chat), GPT-4o (translation) | Balance cost/quality |
| Chunking | 512 tokens, 50 overlap | Semantic coherence for technical content |
| Personalization | On-demand LLM with Postgres caching | Flexible, avoids pre-generation |
| Translation | GPT-4o with code block markers | High quality Urdu, code preservation |

All technical unknowns resolved. Ready for Phase 1 design.
