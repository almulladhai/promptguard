"""
detector.py — Two-layer prompt injection detector

Layer 1: regex rule engine
Layer 2: trained ML model (model/model.pkl)

This version:
- loads the correct saved pipeline
- uses predict_proba safely
- reduces false positives
- uses better thresholds
"""

import os
import re
import joblib
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    label: str                    # "safe" | "suspicious" | "malicious"
    risk_score: float             # 0.0 – 1.0
    triggered_rules: list[str] = field(default_factory=list)
    ml_label: str = "unknown"
    ml_confidence: float = 0.0
    blocked: bool = False


# ─────────────────────────────────────────────────────────────
# Layer 1: Rule engine
# Make these stricter to reduce false positives
# ─────────────────────────────────────────────────────────────

RULES = [
    (
        "ignore_instructions",
        r"\bignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|rules?|commands?|guidelines?|directives?|prompts?|training)\b"
    ),
    (
        "forget_instructions",
        r"\bforget\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|rules?|commands?|guidelines?|directives?|prompts?|training|everything)\b"
    ),
    (
        "disregard_instructions",
        r"\bdisregard\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|rules?|commands?|guidelines?|directives?|prompts?|training|everything)\b"
    ),
    (
        "override_instructions",
        r"\boverride\s+(your\s+)?(previous\s+)?(instructions?|rules?|training|safety|restrictions?|filters?|settings?)\b"
    ),
    (
        "bypass_safety",
        r"\bbypass\s+(your\s+)?(safety|content|security|ethical|all)?\s*(filters?|restrictions?|rules?|guidelines?|policy|policies|checks?)\b"
    ),
    (
        "reveal_prompt",
        r"\b(reveal|show|print|output|display|leak|expose)\s+(your\s+)?(hidden\s+|secret\s+|internal\s+|base\s+|full\s+|original\s+)?(system\s+)?(prompt|instructions?|configuration|directives?|message|context)\b"
    ),
    (
        "system_prompt_question",
        r"\bwhat\s+(is|are)\s+(your\s+)?(system\s+prompt|hidden\s+instructions?|secret\s+directives?)\b"
    ),
    (
        "developer_mode",
        r"\b(enable|switch\s+to|activate|enter|you\s+are\s+now\s+in)\s+(developer|admin|root|dev)\s+mode\b"
    ),
    (
        "jailbreak_dan",
        r"\bDAN\b|\bdo\s+anything\s+now\b|\bjailbreak\s*(mode|activated|enabled)?\b|\byou('?re|\s+are)\s+(now\s+)?jailbroken\b"
    ),
    (
        "no_restrictions",
        r"\b(no|without)\s+(restrictions?|rules?|filters?|guidelines?|constraints?|censorship)\b|\bremove\s+all\s+(restrictions?|rules?|filters?|guidelines?|censorship)\b"
    ),
    (
        "pretend_no_rules",
        r"\b(pretend|act|behave|respond)\s+(as\s+if|like)\s+(you\s+)?(have\s+no|without|had\s+no)\s+(rules?|restrictions?|guidelines?|filters?|training|ethics?)\b"
    ),
    (
        "uncensored_mode",
        r"\buncensored(\s+mode|\s+version|\s+ai)?\b|\bwithout\s+censorship\b|\bremove\s+all\s+censorship\b"
    ),
    (
        "persona_hijack",
        r"\b(you\s+are\s+now|act\s+as|switch\s+to|become|simulate(\s+being)?)\s+(an?\s+)?(AI|GPT|model|assistant|chatbot)\s+(with\s+no|without|that\s+has\s+no)\s+(rules?|restrictions?|guidelines?|ethics?|filters?)\b"
    ),
    (
        "context_dump",
        r"\b(repeat|print|output|show|display|copy)\s+(everything|all(\s+text)?)\s*(above|before|prior|in\s+your\s+context|from\s+the\s+(beginning|start|top))\b"
    ),
    (
        "safety_disabled",
        r"\b(safety|content|security|ethical)\s*(mode|filter|policy|guidelines?|check)?\s*(is\s+)?(now\s+)?(disabled?|turned\s+off|removed|deactivated|suspended|lifted)\b"
    ),
    (
        "new_instructions",
        r"\bnew\s+(instructions?|system\s+prompt|directives?|rules?|commands?)\s*[:\-–]?\s*(ignore|forget|disregard|you\s+are|you\s+must)\b"
    ),
    (
        "all_rules_gone",
        r"\b(all\s+)?(rules?|restrictions?|guidelines?|filters?|limitations?)\s+(are\s+)?(now\s+)?(suspended|lifted|removed|gone|disabled?|no\s+longer\s+apply)\b"
    ),
    (
        "evil_persona",
        r"\b(EvilGPT|DAN|AIM|STAN|AntiGPT|BetterDAN|JailBreak)\b"
    ),
]

_compiled_rules = [
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in RULES
]


def rule_check(prompt: str) -> tuple[float, list[str]]:
    triggered = [name for name, rx in _compiled_rules if rx.search(prompt)]

    if not triggered:
        return 0.0, []

    # Give high risk only for real rule hits
    score = min(0.85 + 0.05 * (len(triggered) - 1), 0.98)
    return round(score, 4), triggered


# ─────────────────────────────────────────────────────────────
# Layer 2: ML classifier
# ─────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "model.pkl"
)

_model = None


def _load_model() -> bool:
    global _model

    if _model is not None:
        return True

    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        return True

    return False


def ml_check(prompt: str) -> tuple[str, float]:
    if not _load_model():
        return "unknown", 0.5

    pred = _model.predict([prompt])[0]
    proba = _model.predict_proba([prompt])[0]

    classes = list(_model.named_steps["clf"].classes_)
    confidence = float(proba[classes.index(pred)])

    return pred, round(confidence, 4)


# ─────────────────────────────────────────────────────────────
# Combined detection
# ─────────────────────────────────────────────────────────────

BLOCK_THRESHOLD = 0.80
SUSPICIOUS_THRESHOLD = 0.60


def detect(prompt: str) -> DetectionResult:
    prompt = prompt.strip()

    if not prompt:
        return DetectionResult(
            label="safe",
            risk_score=0.0,
            triggered_rules=[],
            ml_label="unknown",
            ml_confidence=0.0,
            blocked=False,
        )

    rule_score, triggered = rule_check(prompt)
    ml_label, ml_conf = ml_check(prompt)

    # Fusion logic
    if triggered:
        risk = rule_score
    else:
        if ml_label == "malicious":
            risk = ml_conf
        else:
            risk = 1.0 - ml_conf

    risk = round(risk, 4)

    if risk >= BLOCK_THRESHOLD:
        label = "malicious"
        blocked = True
    elif risk >= SUSPICIOUS_THRESHOLD:
        label = "suspicious"
        blocked = False
    else:
        label = "safe"
        blocked = False

    return DetectionResult(
        label=label,
        risk_score=risk,
        triggered_rules=triggered,
        ml_label=ml_label,
        ml_confidence=ml_conf,
        blocked=blocked,
    )