# server.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ⬇️ Импортируем функцию из bot.py
from bot import interpret  # def interpret(question: str) -> str

app = FastAPI(title="DreamWisdom API")

# Разрешаем локалку и Firebase (*.web.app / *.firebaseapp.com, включая preview channels)
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

class QueryBody(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

# POST /api/interpret  { "message": "текст сна" }
@app.post("/api/interpret")
def api_interpret(body: QueryBody):
    try:
        return {"answer": interpret(body.message)}
    except Exception as e:
        # Логи останутся в сервере, наружу — общее сообщение
        raise HTTPException(status_code=500, detail="Internal error")

# GET /api/interpret?message=текст
@app.get("/api/interpret")
def api_interpret_get(message: str = Query(..., min_length=1)):
    try:
        return {"answer": interpret(message)}
    except Exception:
        raise HTTPException(status_code=500, detail="Internal error")
