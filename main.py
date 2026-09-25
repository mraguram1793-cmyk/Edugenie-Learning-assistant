import sys
import os

# ── Fix: ensure D:\PyPackages is always in sys.path ──────────────────────────
_extra = r"D:\PyPackages\Python313\site-packages"
if _extra not in sys.path:
    sys.path.insert(0, _extra)
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Import all modules
from qna import answer_question_with_gemini
from explanation_module import explain_topic
from summary_module import summarize_text
from quiz_module import generate_quiz
from learning_path import get_learning_recommendations

app = FastAPI(title="EduGenie - AI Learning Assistant")

# Allow requests from Live Server (5500) and any localhost origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500",
                   "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Frontend is served by Live Server (port 5500); no static mount needed.

# ─────────────────────────────────────────
# Global exception handler – always JSON
# ─────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": f"Server error: {str(exc)}"}
    )

def _check_api_key():
    """Return a JSONResponse error if the key is missing/placeholder, else None."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        return JSONResponse(
            status_code=500,
            content={"error": "GEMINI_API_KEY is not configured. Open your .env file and set a valid key."}
        )
    return None

# ─────────────────────────────────────────
# Root – Health check
# ─────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "EduGenie API is running"}

# ─────────────────────────────────────────
# Q&A – GET API using Gemini
# ─────────────────────────────────────────
@app.get("/qa")
async def answer_question(question: str = Query(...)):
    if (err := _check_api_key()): return err
    answer = answer_question_with_gemini(question, GEMINI_API_KEY, GEMINI_MODEL)
    return {"answer": answer}

# ─────────────────────────────────────────
# Explanation – POST API
# ─────────────────────────────────────────
@app.post("/explain/")
async def explain_api(request: Request):
    if (err := _check_api_key()): return err
    data = await request.json()
    topic = data.get("topic")
    if not topic:
        return JSONResponse(content={"error": "Please provide a topic."}, status_code=400)
    explanation = explain_topic(topic, GEMINI_API_KEY, GEMINI_MODEL)
    return {"topic": topic, "explanation": explanation}

# ─────────────────────────────────────────
# Summarization – POST API
# ─────────────────────────────────────────
@app.post("/summarize/")
async def summarize_api(request: Request):
    if (err := _check_api_key()): return err
    data = await request.json()
    text = data.get("text")
    if not text:
        return JSONResponse(content={"error": "Please provide text to summarize."}, status_code=400)
    summary = summarize_text(text, GEMINI_API_KEY, GEMINI_MODEL)
    return {"summary": summary}

# ─────────────────────────────────────────
# Quiz Generation – POST API
# ─────────────────────────────────────────
@app.post("/quiz")
async def quiz_api(request: Request):
    if (err := _check_api_key()): return err
    data = await request.json()
    text = data.get("text")
    if not text:
        return JSONResponse(content={"error": "Please provide text for quiz."}, status_code=400)
    quiz = generate_quiz(text, GEMINI_API_KEY, GEMINI_MODEL)
    return JSONResponse(content={"quiz": quiz})

# ─────────────────────────────────────────
# Learning Recommendations – GET API
# ─────────────────────────────────────────
@app.get("/learn/recommendations")
async def learning_recommendation_api(topic: str = Query(...)):
    if (err := _check_api_key()): return err
    recommendation = get_learning_recommendations(topic, GEMINI_API_KEY, GEMINI_MODEL)
    return {"topic": topic, "recommendation": recommendation}
