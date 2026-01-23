"""Vercel serverless function entry point for FastAPI."""

import sys
from pathlib import Path

# Add the backend src directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from src.main import app

# Export the FastAPI app for Vercel
handler = app
