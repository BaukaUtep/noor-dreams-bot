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
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "https://dreamwisdom-firebase-46707352.web.app",
        "https://dreamwisdom-firebase-46707352.firebaseapp.com",
        "https://dream-wisdom.com",
        "https://www.dream-wisdom.com",
        "https://3000-firebase-dreamwisdom-firebase-1757095027813.cluster-edb2jv34dnhjisxuq5m7l37ccy.cloudworkstations.dev",
    ],
    allow_origin_regex=r"^https://([a-z0-9-]+)\.(web\.app|firebaseapp\.com|cloudworkstations\.dev)$",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
    max_age=86400,
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




