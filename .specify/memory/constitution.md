<!--
  ============================================================================
  SYNC IMPACT REPORT
  ============================================================================
  Version Change: 0.0.0 → 1.0.0 (MAJOR - Initial constitution creation)

  Modified Principles: N/A (Initial creation)

  Added Sections:
    - 10 Core Principles (I through X)
    - Technology Stack section
    - Quality Standards section
    - Development Workflow section
    - Governance section

  Removed Sections: N/A (Initial creation)

  Templates Requiring Updates:
    ✅ plan-template.md - Constitution Check aligns with principles
    ✅ spec-template.md - Requirements align with FR/NFR standards
    ✅ tasks-template.md - Task categories reflect principle-driven types

  Follow-up TODOs: None
  ============================================================================
-->

# Physical AI & Humanoid Robotics Textbook Constitution

## Mission Statement

To create the definitive AI-native educational resource for Physical AI and Humanoid Robotics that empowers learners to build embodied intelligent systems through hands-on, simulation-first methodology. This textbook bridges the gap between theoretical AI knowledge and practical robotics implementation, preparing the next generation of engineers for the human-robot partnership era.

## Core Principles

### I. Content-First Development

Every chapter MUST deliver genuine educational value before any technical embellishment. Content creation follows a strict hierarchy:

1. **Educational Accuracy**: All technical content MUST be verified against official documentation (ROS 2, NVIDIA Isaac, Gazebo)
2. **Practical Relevance**: Every concept MUST include real-world application context
3. **Progressive Complexity**: Content MUST build from fundamentals to advanced topics within each chapter
4. **Code Completeness**: All code examples MUST be runnable without modification

**Rationale**: Textbook credibility depends on technical accuracy. Readers invest significant time; inaccurate content wastes that investment and damages trust.

### II. Simulation-First Pedagogy

All robotics concepts MUST be demonstrated in simulation before any physical hardware discussion:

1. **Safety**: Simulation allows experimentation without risk of hardware damage or injury
2. **Accessibility**: Not all learners have access to physical robots; simulation democratizes learning
3. **Iteration Speed**: Simulation enables rapid prototyping and failure recovery
4. **Reproducibility**: Simulated environments ensure consistent learning experiences

**Implementation**:
- Every practical section MUST include Gazebo or Isaac Sim instructions
- Physical hardware sections are OPTIONAL supplements, never prerequisites
- Sim-to-Real transfer techniques MUST be covered before any physical deployment

**Rationale**: Physical robots cost $3,000-$90,000+. Simulation ensures every learner can complete the course regardless of budget.

### III. Spec-Driven Content Creation

All content MUST flow through the Spec-Kit Plus workflow:

1. **Specification**: Define chapter objectives, learning outcomes, and acceptance criteria BEFORE writing
2. **Planning**: Architecture the chapter structure, dependencies, and practical exercises
3. **Task Breakdown**: Decompose into atomic, testable content units
4. **Implementation**: Write content following the approved plan
5. **Review**: Validate against specification before marking complete

**Non-Negotiable**: No chapter content may be written without an approved spec. Ad-hoc content creation is prohibited.

**Rationale**: Educational content benefits from the same rigor as software development. Specifications prevent scope creep and ensure completeness.

### IV. RAG-Native Architecture

The textbook MUST be designed for AI-augmented learning from inception:

1. **Chunking Strategy**: Content MUST be structured for effective vector embedding (semantic sections, clear boundaries)
2. **Metadata Richness**: Every section MUST include searchable metadata (topics, prerequisites, difficulty level)
3. **Question Anticipation**: Content MUST anticipate and address common learner questions inline
4. **Context Preservation**: Code examples MUST include sufficient context for RAG retrieval

**Technical Requirements**:
- Maximum chunk size: 512 tokens with 50-token overlap
- Every code block MUST have a descriptive preceding paragraph
- Cross-references MUST use explicit chapter/section identifiers

**Rationale**: The RAG chatbot is a core deliverable. Content not optimized for retrieval fails the product requirements.

### V. Incremental Deployability

Every feature and chapter MUST be independently deployable and valuable:

1. **Chapter Independence**: Each chapter MUST be readable without requiring unwritten chapters
2. **Feature Slicing**: Features (auth, personalization, translation) MUST not block core book deployment
3. **Progressive Enhancement**: Base book → Chatbot → Auth → Personalization → Translation
4. **Rollback Safety**: Any feature can be disabled without breaking the core experience

**Deployment Milestones**:
- Milestone 1: Static book on GitHub Pages (earns base points)
- Milestone 2: Book + RAG chatbot (completes base requirements)
- Milestone 3+: Bonus features (incremental point accumulation)

**Rationale**: Hackathon deadline is fixed. Incremental delivery ensures partial credit even if time runs short.

### VI. Test-Verified Content

All practical exercises and code examples MUST be verified:

1. **Code Testing**: Every code snippet MUST be extracted and tested in CI
2. **Environment Reproducibility**: Docker/devcontainer configurations MUST be provided
3. **Version Pinning**: All dependencies MUST specify exact versions
4. **Platform Testing**: Primary support for Ubuntu 22.04; document Windows/Mac limitations

**Acceptance Criteria**:
- Code blocks with `python` or `bash` fences MUST be executable
- Expected outputs MUST be documented
- Error scenarios MUST be covered with troubleshooting guidance

**Rationale**: Nothing destroys learner confidence faster than broken code examples. Verified content builds trust.

### VII. Accessibility & Internationalization

Content MUST be accessible to diverse learners:

1. **Language Clarity**: Use simple, direct English; avoid idioms and colloquialisms
2. **Visual Descriptions**: All diagrams MUST have text descriptions for screen readers
3. **Translation-Ready**: Content structure MUST support Urdu translation feature
4. **Prerequisite Transparency**: Clearly state required background knowledge per chapter

**Personalization Support**:
- Content MUST be tagged with difficulty levels (Beginner/Intermediate/Advanced)
- Alternative explanations MUST be available for complex concepts
- Hardware/software background questions inform content adaptation

**Rationale**: Bonus points require translation and personalization. Architecture must support these from the start.

### VIII. Security & Privacy by Design

All features handling user data MUST follow security best practices:

1. **Authentication**: Better-Auth implementation MUST follow OWASP guidelines
2. **Data Minimization**: Collect only data necessary for personalization
3. **Secret Management**: No hardcoded credentials; use environment variables
4. **API Security**: All endpoints MUST validate input and authenticate requests

**Prohibited**:
- Storing passwords in plain text
- Exposing API keys in client-side code
- Logging personally identifiable information (PII)
- Disabling HTTPS in production

**Rationale**: User trust is non-negotiable. Security vulnerabilities would disqualify the submission.

### IX. Performance & Resource Efficiency

The deployed book and chatbot MUST perform well on standard hardware:

1. **Page Load**: Initial page load under 3 seconds on 3G connection
2. **Chatbot Response**: RAG query response under 5 seconds (excluding LLM generation)
3. **Build Time**: Full Docusaurus build under 5 minutes
4. **Bundle Size**: JavaScript bundle under 500KB gzipped

**Free Tier Compliance**:
- Qdrant Cloud free tier limits MUST be respected
- Neon Postgres free tier limits MUST be respected
- Vercel/GitHub Pages deployment limits MUST be respected

**Rationale**: Hackathon uses free tiers. Exceeding limits causes deployment failures.

### X. Documentation & Maintainability

All code and configuration MUST be documented for future maintainers:

1. **README**: Every directory with code MUST have a README explaining its purpose
2. **Inline Comments**: Complex logic MUST have explanatory comments
3. **API Documentation**: All endpoints MUST have OpenAPI/Swagger documentation
4. **Architecture Decision Records**: Significant decisions MUST be documented in ADRs

**Code Standards**:
- Python: Follow PEP 8, type hints required for public functions
- TypeScript: Strict mode enabled, explicit types for exports
- Markdown: Follow CommonMark specification

**Rationale**: Panaversity may extend this work. Clean, documented code enables future development.

## Technology Stack

### Book Platform
| Component | Technology | Justification |
|-----------|------------|---------------|
| Static Site Generator | Docusaurus 3.x | Required by hackathon |
| Hosting | GitHub Pages / Vercel | Required by hackathon |
| Language | TypeScript | Type safety for React components |
| Styling | Tailwind CSS | Rapid UI development |

### RAG Chatbot Backend
| Component | Technology | Justification |
|-----------|------------|---------------|
| API Framework | FastAPI | Required by hackathon |
| Vector Database | Qdrant Cloud (Free Tier) | Required by hackathon |
| Relational Database | Neon Serverless Postgres | Required by hackathon |
| LLM Integration | OpenAI Agents SDK | Required by hackathon |
| Embedding Model | text-embedding-3-small | Cost-effective, sufficient quality |

### Authentication (Bonus)
| Component | Technology | Justification |
|-----------|------------|---------------|
| Auth Library | Better-Auth | Required for bonus points |
| Session Storage | Neon Postgres | Reuse existing database |

### Development Tools
| Component | Technology | Justification |
|-----------|------------|---------------|
| AI Assistant | Claude Code CLI | Required by hackathon |
| Workflow | Spec-Kit Plus | Required by hackathon |
| Version Control | Git + GitHub | Industry standard |
| CI/CD | GitHub Actions | Free for public repos |

## Quality Standards

### Content Quality Gates
- [ ] Technical accuracy verified against official documentation
- [ ] All code examples tested and runnable
- [ ] Learning objectives clearly stated
- [ ] Practical exercises included with solutions
- [ ] Prerequisites explicitly listed

### Code Quality Gates
- [ ] Linting passes (ESLint for TS, Ruff for Python)
- [ ] Type checking passes (tsc strict, mypy)
- [ ] No hardcoded secrets
- [ ] Tests pass (where applicable)
- [ ] Documentation complete

### Deployment Quality Gates
- [ ] Build succeeds without warnings
- [ ] All links functional (no 404s)
- [ ] Responsive design verified (mobile/desktop)
- [ ] Accessibility audit passes (WCAG 2.1 AA)
- [ ] Performance budget met

## Development Workflow

### Branch Strategy
```
main (production)
  └── develop (integration)
       ├── feature/chapter-01-intro
       ├── feature/rag-chatbot
       ├── feature/auth-system
       └── feature/personalization
```

### Commit Convention
```
<type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, test, chore
Scope: chapter-XX, chatbot, auth, frontend, backend
```

### Review Requirements
- All PRs require self-review checklist completion
- Code changes require passing CI
- Content changes require accuracy verification

## Governance

### Constitution Authority
This constitution supersedes all other project documentation in case of conflict. All development decisions MUST align with these principles.

### Amendment Process
1. **Proposal**: Document the proposed change with rationale
2. **Impact Analysis**: Identify affected artifacts and templates
3. **Approval**: Explicit approval required before implementation
4. **Migration**: Update all dependent documents
5. **Version Bump**: Increment constitution version appropriately

### Version Policy
- **MAJOR**: Principle removal, redefinition, or backward-incompatible governance change
- **MINOR**: New principle added, section materially expanded
- **PATCH**: Clarifications, typo fixes, non-semantic refinements

### Compliance
- Every PR MUST include a Constitution Check confirming alignment
- Violations MUST be documented with justification if unavoidable
- Regular audits against constitution principles during development

### Guidance Files
- Runtime development guidance: `CLAUDE.md`
- Feature specifications: `specs/<feature>/spec.md`
- Implementation plans: `specs/<feature>/plan.md`
- Task tracking: `specs/<feature>/tasks.md`

**Version**: 1.0.0 | **Ratified**: 2025-01-21 | **Last Amended**: 2025-01-21
