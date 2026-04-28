# PromptGuard v2 — Prompt Injection Detector

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train the model (from project root)
python model/train_model.py

# 3. Start API + frontend
uvicorn api.main:app --reload

# 4. Open browser
open http://localhost:8000
```

## Structure
```
promptguard/
├── data/dataset.csv        ← ~160 labelled prompts
├── model/
│   └── train_model.py      ← trains TF-IDF + LR classifier
├── api/
│   ├── detector.py         ← rule engine + ML layer
│   └── main.py             ← FastAPI
├── frontend/
│   └── index.html          ← standalone UI
└── requirements.txt
```

## Endpoints
- `GET  /`          → serves the UI
- `POST /detect`    → `{"prompt": "..."}` → detection result
- `GET  /health`    → health check
- `GET  /docs`      → Swagger UI
