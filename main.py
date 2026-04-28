"""
main.py — PromptGuard API
Run: uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from detector import detect, DetectionResult

app = FastAPI(title="PromptGuard API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptIn(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)

class DetectOut(BaseModel):
    label: str
    risk_score: float
    risk_percent: int
    triggered_rules: list[str]
    ml_label: str
    ml_confidence: float
    blocked: bool
    message: str

@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse("frontend/index.html")

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}

@app.post("/detect", response_model=DetectOut)
def detect_prompt(body: PromptIn):
    r: DetectionResult = detect(body.prompt)

    if r.label == "malicious":
        msg = f"Blocked — malicious prompt detected ({int(r.risk_score * 100)}%)"
    elif r.label == "suspicious":
        msg = f"Warning — suspicious prompt ({int(r.risk_score * 100)}%)"
    else:
        msg = f"Safe prompt ({int(r.risk_score * 100)}%)"

    return DetectOut(
        label=r.label,
        risk_score=r.risk_score,
        risk_percent=int(r.risk_score * 100),
        triggered_rules=r.triggered_rules,
        ml_label=r.ml_label,
        ml_confidence=r.ml_confidence,
        blocked=r.blocked,
        message=msg,
    )