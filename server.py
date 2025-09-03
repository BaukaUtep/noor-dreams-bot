import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ❗️Поменяй импорт на свой модуль/функцию, если нужно:
# try сначала из bot.py, если нет — из main.py
try:
    from bot import answer_question   # def answer_question(text: str) -> str
except Exception:
    from main import answer_question  # запасной вариант

app = FastAPI(title="DreamWisdom API")

# Разрешаем запросы с локалки и Firebase (включая preview-каналы)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",  # Vite
        "https://localhost",      # если запускаешь через https локально
    ],
    allow_origin_regex=r"^https://([a-z0-9-]+)\.(web|firebaseapp)\.com$",  # все *.web.app и *.firebaseapp.com
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    message: str

@app.get("/")
def root():
    return {"ok": True, "service": "dreamwisdom-api"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/interpret")
def interpret(q: Query):
    try:
        ans = answer_question(q.message)
        return {"answer": ans}
    except Exception as e:
        # чтобы не светить внутренние ошибки на фронт
        raise HTTPException(status_code=500, detail="Internal error")
