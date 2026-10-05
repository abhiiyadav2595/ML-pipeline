"""
Vercel Serverless Function entrypoint for FastAPI.
Exposes the FastAPI ASGI application for serverless invocation.
"""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure ARTIFACTS_DIR defaults to the repository artifacts directory
if not os.getenv("ARTIFACTS_DIR"):
    os.environ["ARTIFACTS_DIR"] = str(ROOT_DIR / "artifacts")

from src.app.main import app

__all__ = ["app"]
