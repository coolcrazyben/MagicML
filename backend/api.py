"""
backend/api.py
FastAPI entry point for the Railway backend.

Start locally:
  cd backend
  uvicorn api:app --reload --port 8000

Railway (via Procfile / railway.json):
  uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import os
import sys

# ---------------------------------------------------------------------------
# Ensure backend directory is the working directory so all relative paths
# in the imported Python modules resolve to backend/
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.analyzer import analyze_decklist, build_deck as _build_deck

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="MagicML API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    # Covers all *.vercel.app deployments
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    decklist: str


class BuildDeckRequest(BaseModel):
    commander: str
    budget: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(body: AnalyzeRequest):
    if not body.decklist.strip():
        raise HTTPException(status_code=400, detail="No decklist provided")
    try:
        return analyze_decklist(body.decklist)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/build-deck")
async def build_deck_endpoint(body: BuildDeckRequest):
    if not body.commander.strip():
        raise HTTPException(status_code=400, detail="No commander provided")
    if body.budget <= 0:
        raise HTTPException(status_code=400, detail="Budget must be a positive number")
    try:
        return _build_deck(body.commander, body.budget)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
