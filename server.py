# server.py (IMPROVED VERSION - Optimized for High Web Traffic)
import logging
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import from bot.py
from bot import interpret

# ===== CONFIGURATION =====
# Web API gets MORE traffic than Telegram bot
MAX_MESSAGE_LENGTH = 500
MIN_MESSAGE_LENGTH = 3

# Rate limiting - more generous since web is primary channel
WEB_RATE_LIMIT = "30/minute"  # 30 requests per minute per IP (was 10)
# This allows 1 request every 2 seconds, good for legitimate users
# Adjust if needed: "60/minute" = 1 per second, "20/minute" = 1 per 3 seconds

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
# Only allow specific domains
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
    allow_methods=["GET", "POST"],
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
            raise ValueError(f'Dream too short (minimum {MIN_MESSAGE_LENGTH} characters)')
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f'Dream too long (maximum {MAX_MESSAGE_LENGTH} characters)')

        # Check if it's only whitespace or repeated characters (spam detection)
        if len(set(v.replace(' ', ''))) < 3:
            raise ValueError('Invalid dream text')

        return v

# ===== METRICS (optional - helps you understand usage) =====
request_count = 0
error_count = 0

# ===== ENDPOINTS =====
@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "DreamWisdom API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "interpret_post": "/api/interpret (POST)",
            "interpret_get": "/api/interpret?message=... (GET)"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "DreamWisdom API",
        "version": "1.2.0",
        "requests_processed": request_count,
        "errors": error_count
    }

@app.post("/api/interpret")
@limiter.limit(WEB_RATE_LIMIT)  # 30 requests per minute (primary channel)
async def api_interpret_post(request: Request, body: QueryBody):
    """
    Interpret a dream using POST method (TEXT ONLY)

    Model: gpt-4o-mini (text-only, no images or voice)
    Rate limit: 30 requests per minute per IP
    Max message length: 500 characters

    Example:
    {
        "message": "I dreamed of flying birds over blue water"
    }
    """
    global request_count, error_count

    client_ip = get_remote_address(request)
    logger.info(f"POST /api/interpret from {client_ip}, length: {len(body.message)}")

    try:
        answer = interpret(body.message)
        request_count += 1
        logger.info(f"✓ Success: {client_ip}")
        return {"answer": answer}

    except ValueError as e:
        # Client error (invalid input)
        error_count += 1
        logger.warning(f"✗ Validation error from {client_ip}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Server error (OpenAI/Pinecone issue)
        error_count += 1
        logger.error(f"✗ Server error from {client_ip}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Unable to interpret your dream right now. Please try again in a moment."
        )

@app.get("/api/interpret")
@limiter.limit(WEB_RATE_LIMIT)  # 30 requests per minute
async def api_interpret_get(
    request: Request,
    message: str = Query(
        ...,
        min_length=MIN_MESSAGE_LENGTH,
        max_length=MAX_MESSAGE_LENGTH,
        description="The dream text to interpret (text only)"
    )
):
    """
    Interpret a dream using GET method (TEXT ONLY)

    Model: gpt-4o-mini (text-only, no images or voice)
    Rate limit: 30 requests per minute per IP
    Max message length: 500 characters

    Example:
    /api/interpret?message=I dreamed of flying birds
    """
    global request_count, error_count

    client_ip = get_remote_address(request)
    logger.info(f"GET /api/interpret from {client_ip}, length: {len(message)}")

    try:
        answer = interpret(message.strip())
        request_count += 1
        logger.info(f"✓ Success: {client_ip}")
        return {"answer": answer}

    except ValueError as e:
        error_count += 1
        logger.warning(f"✗ Validation error from {client_ip}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        error_count += 1
        logger.error(f"✗ Server error from {client_ip}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Unable to interpret your dream right now. Please try again in a moment."
        )

# ===== STARTUP/SHUTDOWN EVENTS =====
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("DreamWisdom API Started")
    logger.info("=" * 60)
    logger.info(f"Rate limit: {WEB_RATE_LIMIT} per IP")
    logger.info(f"Message length: {MIN_MESSAGE_LENGTH}-{MAX_MESSAGE_LENGTH} chars")
    logger.info(f"Model: gpt-4o-mini (text-only)")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("=" * 60)
    logger.info("DreamWisdom API Shutting Down")
    logger.info(f"Total requests processed: {request_count}")
    logger.info(f"Total errors: {error_count}")
    logger.info("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
