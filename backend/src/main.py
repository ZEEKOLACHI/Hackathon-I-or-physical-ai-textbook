"""FastAPI application factory and configuration."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware.error_handler import setup_error_handlers
from src.api.middleware.rate_limiter import RateLimitMiddleware
from src.api.routes import auth, chat, health, personalize, translate, users
from src.config import settings
from src.db.postgres import close_db, init_db
from src.db.qdrant import close_qdrant, init_qdrant


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    # Startup
    await init_db()
    await init_qdrant()
    yield
    # Shutdown
    await close_db()
    await close_qdrant()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Physical AI Textbook API",
        description="RAG-based chatbot and content API for Physical AI & Humanoid Robotics Textbook",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting middleware
    app.add_middleware(RateLimitMiddleware)

    # Error handlers
    setup_error_handlers(app)

    # Routes
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(auth.router, prefix="/api/v1", tags=["auth"])
    app.include_router(users.router, prefix="/api/v1", tags=["users"])
    app.include_router(chat.router, prefix="/api/v1", tags=["chat", "search"])
    app.include_router(personalize.router, prefix="/api/v1", tags=["content"])
    app.include_router(translate.router, prefix="/api/v1", tags=["content"])

    return app


app = create_app()
