"""Vercel serverless function entry point for FastAPI."""

import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

_import_error: str | None = None

try:
    from src.main import app
except Exception as e:
    # Capture error for fallback routes
    _import_error = str(e)

    # Fallback app if import fails
    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"error": "Import failed", "message": _import_error}

    @app.get("/health")
    async def health():
        return {"status": "error", "message": _import_error}
