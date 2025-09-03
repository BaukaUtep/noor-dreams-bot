# server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ⬇️ ЯВНЫЙ импорт из bot.py
from bot import answer_question   # <-- здесь ИМЯ ФАЙЛА и ФУНКЦИИ из твоего проекта

app = FastAPI(title="DreamWisdom API")

# Разрешаем запросы с локалки и всех *.web.app / *.firebaseapp.com (Firebase превью/прод)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
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
