# server.py (IMPROVED VERSION)
import logging
from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import from bot.py
from bot import interpret

# ===== CONFIGURATION =====
MAX_MESSAGE_LENGTH = 500
MIN_MESSAGE_LENGTH = 3

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== RATE LIMITER =====
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="DreamWisdom API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ===== CORS MIDDLEWARE =====
# Only allow specific domains (tightened security)
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
    ],
    # More specific regex - only your exact Firebase project
    allow_origin_regex=r"^https://dreamwisdom-firebase-46707352-[a-z0-9]+\.(web\.app|firebaseapp\.com)$",
    allow_methods=["GET", "POST"],  # Only what we need
    allow_headers=["Content-Type", "Accept"],
    allow_credentials=False,
    max_age=86400,
)

# ===== REQUEST MODELS =====
class QueryBody(BaseModel):
    message: str

    @validator('message')
    def validate_message(cls, v):
        # Strip whitespace
        v = v.strip()

        # Check length
        if len(v) < MIN_MESSAGE_LENGTH:
            raise ValueError(f'Message too short (minimum {MIN_MESSAGE_LENGTH} characters)')
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f'Message too long (maximum {MAX_MESSAGE_LENGTH} characters)')

        # Check for suspicious patterns (optional - uncomment if needed)
        # if v.count('\n') > 50:  # Too many line breaks
        #     raise ValueError('Invalid message format')

        return v

# ===== ENDPOINTS =====
@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "DreamWisdom API",
        "version": "1.1.0"
    }

@app.post("/api/interpret")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def api_interpret_post(request: Request, body: QueryBody):
    """
    Interpret a dream using POST method

    Rate limit: 10 requests per minute per IP
    Max message length: 500 characters
    """
    client_ip = get_remote_address(request)
    logger.info(f"POST request from {client_ip}, message length: {len(body.message)}")

    try:
        answer = interpret(body.message)
        logger.info(f"Successfully processed request from {client_ip}")
        return {"answer": answer}

    except ValueError as e:
        # Client errors (bad input)
        logger.warning(f"Validation error from {client_ip}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Server errors (our fault)
        logger.error(f"Server error processing request from {client_ip}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Unable to process your dream at this time. Please try again later."
        )

@app.get("/api/interpret")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def api_interpret_get(
    request: Request,
    message: str = Query(
        ...,
        min_length=MIN_MESSAGE_LENGTH,
        max_length=MAX_MESSAGE_LENGTH,
        description="The dream text to interpret"
    )
):
    """
    Interpret a dream using GET method

    Rate limit: 10 requests per minute per IP
    Max message length: 500 characters
    """
    client_ip = get_remote_address(request)
    logger.info(f"GET request from {client_ip}, message length: {len(message)}")

    try:
        answer = interpret(message.strip())
        logger.info(f"Successfully processed request from {client_ip}")
        return {"answer": answer}

    except ValueError as e:
        logger.warning(f"Validation error from {client_ip}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Server error processing request from {client_ip}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Unable to process your dream at this time. Please try again later."
        )

# ===== STARTUP/SHUTDOWN EVENTS =====
@app.on_event("startup")
async def startup_event():
    logger.info("DreamWisdom API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("DreamWisdom API shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
