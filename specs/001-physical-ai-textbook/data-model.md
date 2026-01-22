# Data Model: Physical AI & Humanoid Robotics Textbook

**Date**: 2025-01-21 | **Plan**: [plan.md](./plan.md)

## Overview

This document defines all data entities, their relationships, and validation rules for the textbook platform.

---

## Entity Relationship Diagram

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│     Chapter     │       │      User       │       │   ChatSession   │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ id (PK)         │       │ id (PK)         │◄──────│ user_id (FK)    │
│ part_number     │       │ email           │       │ id (PK)         │
│ chapter_number  │       │ hashed_password │       │ created_at      │
│ title           │       │ programming_lvl │       │ last_message_at │
│ slug            │       │ robotics_lvl    │       └────────┬────────┘
│ difficulty      │       │ hardware[]      │                │
│ content_md      │       │ created_at      │                │
└────────┬────────┘       └────────┬────────┘       ┌────────▼────────┐
         │                         │                │   ChatMessage   │
         │                         │                ├─────────────────┤
         │                ┌────────▼────────┐       │ id (PK)         │
         │                │     Session     │       │ session_id (FK) │
         │                ├─────────────────┤       │ role            │
         │                │ id (PK)         │       │ content         │
         │                │ user_id (FK)    │       │ citations[]     │
         │                │ token           │       │ created_at      │
         │                │ expires_at      │       └─────────────────┘
         │                └─────────────────┘
         │
┌────────▼────────┐       ┌─────────────────┐
│  ContentChunk   │       │  ContentVariant │
├─────────────────┤       ├─────────────────┤
│ id (PK)         │       │ id (PK)         │
│ chapter_id (FK) │◄──────│ chapter_id (FK) │
│ section_id      │       │ user_id (FK)    │
│ section_title   │       │ variant_type    │
│ content         │       │ variant_key     │
│ has_code        │       │ content         │
│ difficulty      │       │ created_at      │
│ embedding[]     │       └─────────────────┘
└─────────────────┘
```

---

## Entities

### 1. Chapter (Static - Markdown files)

Represents a unit of educational content. Stored as Markdown files, indexed in Qdrant.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | string | PK, pattern: `ch-{part}-{num}` | Unique identifier (e.g., `ch-1-01`) |
| part_number | integer | 1-7 | Part of the book (7 total) |
| chapter_number | integer | 1-21 | Chapter sequence number |
| title | string | max 200 chars | Chapter title |
| slug | string | lowercase, hyphenated | URL-friendly identifier |
| difficulty | enum | beginner/intermediate/advanced | Target audience level |
| estimated_time | integer | minutes | Estimated reading time |
| prerequisites | string[] | chapter ids | Required prior chapters |

**File Location**: `book/docs/part-{N}-{name}/{NN}-{slug}.md`

**Frontmatter Example**:
```yaml
---
id: ch-1-01
title: Introduction to Physical AI
difficulty: beginner
estimated_time: 25
prerequisites: []
---
```

### 2. User (Neon Postgres)

Represents a registered learner with background profile.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK, auto-generated | Unique identifier |
| email | string | unique, max 255, email format | Login identifier |
| hashed_password | string | bcrypt hash | Secure password storage |
| programming_level | enum | none/beginner/intermediate/advanced | Self-reported skill |
| robotics_level | enum | none/beginner/intermediate/advanced | Self-reported skill |
| hardware_available | string[] | optional | Available hardware (e.g., ["jetson_nano", "turtlebot"]) |
| created_at | timestamp | auto | Account creation time |
| updated_at | timestamp | auto | Last profile update |

**Validation Rules**:
- Email must be valid format and unique
- Password minimum 8 characters
- programming_level and robotics_level default to "beginner"

**SQL Schema**:
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    programming_level VARCHAR(20) DEFAULT 'beginner'
        CHECK (programming_level IN ('none', 'beginner', 'intermediate', 'advanced')),
    robotics_level VARCHAR(20) DEFAULT 'beginner'
        CHECK (robotics_level IN ('none', 'beginner', 'intermediate', 'advanced')),
    hardware_available TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
```

### 3. Session (Neon Postgres - Better-Auth managed)

Represents an active user session for authentication.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | Session identifier |
| user_id | UUID | FK → users.id | Associated user |
| token | string | unique, 64 chars | Session token for cookies |
| expires_at | timestamp | required | Expiration time |
| created_at | timestamp | auto | Session start time |
| ip_address | string | optional | Client IP for security |
| user_agent | string | optional | Browser info |

**SQL Schema**:
```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45),
    user_agent TEXT
);

CREATE INDEX idx_sessions_token ON sessions(token);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);
```

### 4. ChatSession (Neon Postgres)

Represents a conversation between a user and the RAG chatbot.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | Session identifier |
| user_id | UUID | FK → users.id, nullable | Associated user (null for anonymous) |
| created_at | timestamp | auto | Conversation start |
| last_message_at | timestamp | auto-updated | Last activity time |
| context_chapter | string | optional | Chapter being viewed when started |

**SQL Schema**:
```sql
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_message_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    context_chapter VARCHAR(20)
);

CREATE INDEX idx_chat_sessions_user_id ON chat_sessions(user_id);
```

### 5. ChatMessage (Neon Postgres)

Represents a single message in a chat conversation.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | Message identifier |
| session_id | UUID | FK → chat_sessions.id | Parent conversation |
| role | enum | user/assistant | Message sender |
| content | text | required | Message text |
| citations | jsonb | optional | Array of chapter/section references |
| created_at | timestamp | auto | Message timestamp |

**Citations Schema**:
```json
{
  "citations": [
    {
      "chapter_id": "ch-1-02",
      "section_id": "ros2-nodes",
      "section_title": "Understanding ROS 2 Nodes",
      "relevance_score": 0.92
    }
  ]
}
```

**SQL Schema**:
```sql
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(10) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    citations JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
```

### 6. ContentChunk (Qdrant Vector Store)

Represents an indexed chunk of chapter content for RAG retrieval.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | Chunk identifier |
| chapter_id | string | required | Source chapter |
| section_id | string | required | Source section slug |
| section_title | string | max 200 | Section heading |
| content | text | max 2000 chars | Chunk text |
| has_code | boolean | required | Contains code block |
| difficulty | enum | beginner/intermediate/advanced | Content difficulty |
| embedding | float[1536] | required | text-embedding-3-small vector |

**Qdrant Point Schema**:
```python
{
    "id": "uuid",
    "vector": [0.1, 0.2, ...],  # 1536 dimensions
    "payload": {
        "chapter_id": "ch-1-02",
        "section_id": "ros2-nodes",
        "section_title": "Understanding ROS 2 Nodes",
        "content": "ROS 2 nodes are the fundamental...",
        "has_code": false,
        "difficulty": "beginner"
    }
}
```

### 7. ContentVariant (Neon Postgres)

Represents a personalized or translated version of chapter content.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | UUID | PK | Variant identifier |
| chapter_id | string | required | Source chapter |
| user_id | UUID | FK → users.id, nullable | For personalized (null for translations) |
| variant_type | enum | personalized/translated | Transformation type |
| variant_key | string | required | Level (beginner/etc) or language (urdu) |
| content | text | required | Transformed content (Markdown) |
| created_at | timestamp | auto | Generation time |

**SQL Schema**:
```sql
CREATE TABLE content_variants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chapter_id VARCHAR(20) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    variant_type VARCHAR(20) NOT NULL CHECK (variant_type IN ('personalized', 'translated')),
    variant_key VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chapter_id, user_id, variant_type, variant_key)
);

CREATE INDEX idx_content_variants_lookup ON content_variants(chapter_id, variant_type, variant_key);
```

---

## State Transitions

### User Registration Flow

```
[Anonymous] → [Registered] → [Logged In] → [Session Expired] → [Logged In]
                                   ↓
                              [Logged Out]
```

### Content Personalization Flow

```
[Original Content] → [Personalization Requested]
                            ↓
                    [Check Cache]
                    ↓           ↓
              [Cache Hit]  [Cache Miss]
                    ↓           ↓
              [Return]    [Generate via LLM]
                                ↓
                          [Store in Cache]
                                ↓
                          [Return]
```

### Chat Message Flow

```
[User Sends Message] → [Create/Resume Session]
                              ↓
                       [Save User Message]
                              ↓
                       [Embed Query]
                              ↓
                       [Search Qdrant]
                              ↓
                       [Assemble Context]
                              ↓
                       [Generate Response]
                              ↓
                       [Save Assistant Message with Citations]
                              ↓
                       [Return Response]
```

---

## Indexes and Performance

### Neon Postgres Indexes

| Table | Index | Purpose |
|-------|-------|---------|
| users | idx_users_email | Login lookup |
| sessions | idx_sessions_token | Session validation |
| sessions | idx_sessions_expires_at | Cleanup job |
| chat_sessions | idx_chat_sessions_user_id | User history |
| chat_messages | idx_chat_messages_session_id | Message retrieval |
| content_variants | idx_content_variants_lookup | Cache lookup |

### Qdrant Indexes

| Collection | Index | Purpose |
|------------|-------|---------|
| textbook_content | HNSW on embedding | Semantic search |
| textbook_content | Filter on chapter_id | Chapter-scoped search |
| textbook_content | Filter on difficulty | Level-filtered search |

---

## Data Retention

| Entity | Retention | Cleanup |
|--------|-----------|---------|
| Users | Indefinite | Manual deletion on request |
| Sessions | 30 days after expiry | Scheduled job |
| ChatSessions | 90 days inactive | Scheduled job |
| ChatMessages | With parent session | Cascade delete |
| ContentVariants | 30 days unused | Scheduled job |
| ContentChunks | Indefinite | Re-indexed on content update |
