"""
FastAPI backend for Devanagari Syllabification.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.inference import SyllableSegmenter
from src.config import BILSTM_CRF_MODEL_PATH, MODEL_PATH

app = FastAPI(
    title="Devanagari Syllabification API",
    description="ML-powered API to segment Devanagari words into syllables using BiLSTM+CRF model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
segmenter = None

@app.on_event("startup")
async def load_model():
    global segmenter
    model_path = BILSTM_CRF_MODEL_PATH if Path(BILSTM_CRF_MODEL_PATH).exists() else MODEL_PATH
    segmenter = SyllableSegmenter(model_path)
    print(f"Model loaded from {model_path}")


class SegmentRequest(BaseModel):
    word: str
    
    class Config:
        json_schema_extra = {
            "example": {"word": "भारत"}
        }


class SegmentResponse(BaseModel):
    word: str
    syllables: List[str]
    hyphenated: str
    count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "word": "भारत",
                "syllables": ["भा", "रत"],
                "hyphenated": "भा-रत",
                "count": 2
            }
        }


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Devanagari Syllabification API",
        "docs": "/docs",
        "api": "/api"
    }


@app.get("/api")
async def api_info():
    """List all available API endpoints."""
    return {
        "name": "Devanagari Syllabification API",
        "version": "1.0.0",
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Health check"
            },
            {
                "path": "/api",
                "method": "GET",
                "description": "List all endpoints (this page)"
            },
            {
                "path": "/docs",
                "method": "GET",
                "description": "Swagger UI documentation"
            },
            {
                "path": "/redoc",
                "method": "GET",
                "description": "ReDoc documentation"
            },
            {
                "path": "/segment",
                "method": "POST",
                "description": "Segment a single Devanagari word into syllables",
                "body": {"word": "string"},
                "example": {
                    "request": {"word": "भारत"},
                    "response": {
                        "word": "भारत",
                        "syllables": ["भा", "रत"],
                        "hyphenated": "भा-रत",
                        "count": 2
                    }
                }
            },
            {
                "path": "/segment/batch",
                "method": "POST",
                "description": "Segment multiple words at once",
                "body": ["word1", "word2", "..."],
                "example": {
                    "request": ["भारत", "नमस्ते"],
                    "response": [
                        {"word": "भारत", "syllables": ["भा", "रत"], "hyphenated": "भा-रत", "count": 2},
                        {"word": "नमस्ते", "syllables": ["न", "म", "स्ते"], "hyphenated": "न-म-स्ते", "count": 3}
                    ]
                }
            }
        ]
    }


@app.post("/segment", response_model=SegmentResponse)
async def segment_word(request: SegmentRequest):
    """
    Segment a Devanagari word into syllables.
    
    - **word**: The Devanagari word to segment (e.g., भारत)
    """
    if not request.word.strip():
        raise HTTPException(status_code=400, detail="Word cannot be empty")
    
    try:
        syllables = segmenter.segment_word(request.word.strip())
        return SegmentResponse(
            word=request.word.strip(),
            syllables=syllables,
            hyphenated="-".join(syllables),
            count=len(syllables)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment/batch")
async def segment_batch(words: List[str]):
    """
    Segment multiple Devanagari words at once.
    
    - **words**: List of Devanagari words to segment
    """
    results = []
    for word in words:
        if word.strip():
            try:
                syllables = segmenter.segment_word(word.strip())
                results.append({
                    "word": word.strip(),
                    "syllables": syllables,
                    "hyphenated": "-".join(syllables),
                    "count": len(syllables)
                })
            except Exception as e:
                results.append({"word": word, "error": str(e)})
    return results
