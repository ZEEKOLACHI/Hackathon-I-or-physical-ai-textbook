# Physical AI & Humanoid Robotics Textbook

An AI-native educational platform for Physical AI and Humanoid Robotics.

**Live Demo**: [physical-ai-textbook.vercel.app](https://physical-ai-textbook.vercel.app)

## Features

- **21-Chapter Interactive Textbook** - Comprehensive coverage from fundamentals to advanced topics
- **RAG-Powered Chatbot** - Ask questions and get context-aware answers with citations
- **User Authentication** - Personalized learning experience
- **Content Personalization** - Adapts to your background (student, professional, researcher)
- **Urdu Translation** - Multilingual support

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Docusaurus 3.x, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.11+ |
| Vector DB | Local Vector Store |
| Database | Neon Serverless Postgres |
| LLM | OpenAI (text-embedding-3-small, GPT-4) |
| Auth | Better-Auth |
| Hosting | Vercel |

## Project Structure

```
book/                    # Docusaurus frontend
  docs/                  # 21 chapters in 7 parts
  src/components/        # React components

backend/                 # FastAPI backend
  src/api/routes/        # API endpoints
  src/services/          # RAG, LLM, personalization
  src/db/                # Database clients
```

## Chapters Overview

### Part 1: Foundations
1. Introduction to Physical AI
2. ROS 2 Fundamentals
3. Simulation Basics

### Part 2: Perception
4. Computer Vision
5. Sensor Fusion
6. 3D Perception

### Part 3: Planning
7. Motion Planning
8. Task Planning
9. Behavior Trees

### Part 4: Control
10. PID Control
11. Force Control
12. Whole-Body Control

### Part 5: Learning
13. Reinforcement Learning
14. Imitation Learning
15. VLA Models

### Part 6: Humanoids
16. Humanoid Kinematics
17. Bipedal Locomotion
18. Manipulation

### Part 7: Integration
19. System Integration
20. Safety Standards
21. Future Directions

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.11+
- OpenAI API Key

### Frontend (Book)
```bash
cd book
npm install
npm run start
```

### Backend (API)
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
uvicorn src.main:app --reload
```

## Environment Variables

```env
DATABASE_URL=           # Neon Postgres connection
OPENAI_API_KEY=         # OpenAI API key
BETTER_AUTH_SECRET=     # Auth session secret
```

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [Docusaurus](https://docusaurus.io/)
- Powered by [OpenAI](https://openai.com/)
- Deployed on [Vercel](https://vercel.com/)
