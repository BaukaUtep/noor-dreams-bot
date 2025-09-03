# server.py (фрагмент сверху)
import os, importlib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Попытаться найти answer_question в популярных путях
CANDIDATES = [
    "bot", "main", "app",
    "worker.main", "telegram_bot",
    "noor_dreams_bot.main", "noor_dreams_bot.bot",
    "src.main", "src.bot",
]

answer_question = None
for modname in CANDIDATES:
    try:
        mod = importlib.import_module(modname)
        if hasattr(mod, "answer_question"):
            answer_question = getattr(mod, "answer_question")
            break
    except Exception:
        continue

if answer_question is None:
    raise RuntimeError(
        f"Не удалось импортировать answer_question. "
        f"Проверь, в каком файле она определена и скорректируй импорт. "
        f"Пробовали: {CANDIDATES}"
    )

app = FastAPI(title="DreamWisdom API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_origin_regex=r"^https://([a-z0-9-]+)\.(web|firebaseapp)\.com$",
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/interpret")
def interpret(q: Query):
    try:
        return {"answer": answer_question(q.message)}
    except Exception:
        raise HTTPException(status_code=500, detail="Internal error")
