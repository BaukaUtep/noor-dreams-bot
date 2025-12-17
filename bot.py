# bot.py (IMPROVED VERSION - Telegram Bot, Text Only)
import os
import time
import requests
import re
import signal
import sys
import logging
from typing import List
from collections import defaultdict

from openai import OpenAI
from pinecone import Pinecone

# ===== CONFIGURATION =====
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX   = "noor-dreams"

# Model settings (TEXT ONLY)
EMBED_MODEL      = "text-embedding-3-small"
EMBED_DIM        = 1536
CHAT_MODEL       = "gpt-4o-mini"  # TEXT ONLY - no images, no voice
TOP_K            = 5

# API endpoints
TELEGRAM_API     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Limits and timeouts
MAX_MESSAGE_LENGTH    = 500
MIN_MESSAGE_LENGTH    = 3
MAX_TOKENS_OUTPUT     = 800
OPENAI_TIMEOUT        = 30.0
TELEGRAM_TIMEOUT      = 30
REQUEST_TIMEOUT       = 10

# Rate limiting (Telegram gets LESS traffic than web)
# More lenient since web is the primary channel
MIN_REQUEST_INTERVAL  = 2  # seconds between requests per user (was 3)

# Telegram limits
TELEGRAM_MAX_MESSAGE_LENGTH = 4096

# File for persisting offset
OFFSET_FILE = "bot_offset.txt"

# Retrieval settings
MIN_CONTEXTS     = 2
PER_SYMBOL_MAX   = 3
UNFILTERED_TOP_K = 20
TRANSLATE_FALLBACK = True

# Fallback messages
FALLBACK = {
    "English": "This interpretation is not present in the provided excerpts.",
    "Russian": "Этого толкования нет в предоставленных отрывках.",
    "Kazakh":  "Түсіндіру берілген үзінділерде жоқ."
}

# Allowed mild generalizations
GENERALIZATION_HINTS = """
Allowed mild generalizations (label them):
phone/mobile -> communication/message (generalized)
talking/speaking/chatting -> communication/speech (generalized)
car/driving/vehicle -> journey/travel (generalized)
computer/laptop -> record/knowledge (generalized)
"""

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== VALIDATE ENVIRONMENT =====
if not TELEGRAM_TOKEN:
    raise RuntimeError("No TELEGRAM_TOKEN set")
if not OPENAI_API_KEY:
    raise RuntimeError("No OPENAI_API_KEY set")
if not PINECONE_API_KEY:
    raise RuntimeError("No PINECONE_API_KEY set")

# ===== INIT CLIENTS =====
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV) if PINECONE_ENV else Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    logger.info("API clients initialized successfully")
    logger.info(f"Using model: {CHAT_MODEL} (text-only)")
except Exception as e:
    logger.error(f"Failed to initialize API clients: {e}")
    raise

# ===== RATE LIMITING =====
user_last_request = defaultdict(float)  # chat_id -> timestamp

def check_rate_limit(chat_id: int) -> tuple[bool, float]:
    """
    Returns (can_proceed, seconds_remaining)
    """
    now = time.time()
    last_request = user_last_request[chat_id]
    time_since_last = now - last_request

    if time_since_last < MIN_REQUEST_INTERVAL:
        remaining = MIN_REQUEST_INTERVAL - time_since_last
        return False, remaining

    user_last_request[chat_id] = now
    return True, 0.0

# ===== OFFSET PERSISTENCE =====
def save_offset(offset: int):
    """Save the last processed update offset to file"""
    try:
        with open(OFFSET_FILE, 'w') as f:
            f.write(str(offset))
    except Exception as e:
        logger.error(f"Failed to save offset: {e}")

def load_offset() -> int:
    """Load the last processed update offset from file"""
    if os.path.exists(OFFSET_FILE):
        try:
            with open(OFFSET_FILE, 'r') as f:
                return int(f.read().strip())
        except Exception as e:
            logger.error(f"Failed to load offset: {e}")
    return None

# ===== TELEGRAM HELPERS =====
def get_updates(offset=None, timeout=TELEGRAM_TIMEOUT):
    """Get updates from Telegram with improved error handling"""
    try:
        r = requests.get(
            f"{TELEGRAM_API}/getUpdates",
            params={"offset": offset, "timeout": timeout},
            timeout=timeout + 5,
        )
        r.raise_for_status()
        return r.json().get("result", [])
    except requests.exceptions.Timeout:
        logger.debug("get_updates timeout (normal for long polling)")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"get_updates request error: {e}")
        return []
    except Exception as e:
        logger.error(f"get_updates unexpected error: {e}")
        return []

def send_message(chat_id: int, text: str):
    """
    Send message to Telegram, handling the 4096 character limit
    TEXT ONLY - no images, voice, or other media
    """
    if not text:
        logger.warning(f"Attempted to send empty message to {chat_id}")
        return

    try:
        # If message fits in one telegram message
        if len(text) <= TELEGRAM_MAX_MESSAGE_LENGTH:
            requests.post(
                f"{TELEGRAM_API}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=REQUEST_TIMEOUT,
            )
        else:
            # Split into chunks if response is very long
            chunks = []
            current_chunk = ""

            for line in text.split('\n'):
                if len(current_chunk) + len(line) + 1 <= TELEGRAM_MAX_MESSAGE_LENGTH:
                    current_chunk += line + '\n'
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = line + '\n'

            if current_chunk:
                chunks.append(current_chunk)

            # Send chunks
            for i, chunk in enumerate(chunks):
                requests.post(
                    f"{TELEGRAM_API}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": chunk if i == 0 else f"(continued...)\n\n{chunk}"
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                if i < len(chunks) - 1:
                    time.sleep(0.5)  # Avoid rate limits

            logger.info(f"Sent long message to {chat_id} in {len(chunks)} parts")

    except Exception as e:
        logger.error(f"send_message error for chat {chat_id}: {e}")

# ===== LANGUAGE DETECTION =====
def detect_language(text: str) -> str:
    """Detect language from text (Kazakh, Russian, or English) - IMPROVED"""
    t = text.lower()

    # Kazakh distinctive letters (expanded with uppercase variants)
    if re.search(r"[ңғүұқәіөһӘәІіҢңҒғҮүҰұҚқӨөҺһ]", t):
        return "kk"

    # Common Kazakh words that don't contain distinctive letters
    # This helps detect words like "арыстан" (lion)
    kazakh_indicators = [
        "арыстан", "түс", "түсінде", "көру", "көрсе", "болу", "білдіреді",
        "мүмкін", "пайғамбар", "жүру", "келу", "бару", "қарау", "деген",
        "екен", "еді", "емес", "тұра"
    ]
    for indicator in kazakh_indicators:
        if indicator in t:
            return "kk"

    # Cyrillic (assume Russian if not Kazakh)
    if re.search(r"[А-Яа-яЁё]", text):
        return "ru"

    return "en"

LANG_NAME = {
    "en": "English",
    "ru": "Russian",
    "kk": "Kazakh"
}

# ===== EMBEDDINGS =====
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for texts using OpenAI"""
    try:
        resp = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
            dimensions=EMBED_DIM,
            timeout=OPENAI_TIMEOUT
        )
        return [d.embedding for d in resp.data]
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise

# ===== RETRIEVAL =====
def _build_grouped_contexts(matches):
    """
    Group sentences by symbol_id and keep up to PER_SYMBOL_MAX per symbol
    """
    buckets = {}
    for m in matches or []:
        md = m.get("metadata", {}) if isinstance(m, dict) else (getattr(m, "metadata", {}) or {})

        symbol_id    = (md.get("symbol_id") or "symbol").strip()
        symbol_label = (md.get("symbol_label") or symbol_id).strip()
        sent_idx     = md.get("sentence_index", "?")
        sent_text    = (md.get("text") or md.get("sentence_text") or md.get("content") or "").strip()
        lang         = md.get("lang")

        if not sent_text:
            continue

        b = buckets.setdefault(symbol_id, {"label": symbol_label, "items": []})
        if len(b["items"]) < PER_SYMBOL_MAX:
            b["items"].append((sent_idx, sent_text, lang))

    # Convert to prompt-ready blocks
    contexts = []
    for sid, data in buckets.items():
        label = data["label"]
        lines = [f"[{label} · {sid}]"]
        for (idx, text, lg) in data["items"]:
            lg_tag = f" ({lg})" if lg else ""
            lines.append(f"• (sent {idx}){lg_tag} {text}")
        contexts.append("\n".join(lines))

    return contexts

def _query_with_filter(emb, filter_obj, top_k):
    """Query Pinecone with optional filter"""
    res = index.query(vector=emb, top_k=top_k, include_metadata=True, filter=filter_obj)
    return res.get("matches") if isinstance(res, dict) else getattr(res, "matches", [])

def _translate_to_english(text: str) -> str:
    """Translate text to English using OpenAI"""
    try:
        chat = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Translate to English. Keep names and religious terms intact."},
                {"role": "user",   "content": text}
            ],
            max_tokens=800,
            timeout=OPENAI_TIMEOUT
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise

def retrieve_context(question: str, lang_code: str, top_k=TOP_K):
    """
    Retrieve relevant context from Pinecone with multi-pass strategy:
    1. Try same-language matches
    2. Try cross-language matches
    3. Try translating to English and searching
    """
    emb = embed_texts([question])[0]

    # Pass A: same-language
    try:
        matches = _query_with_filter(emb, {"lang": lang_code}, top_k)
        contexts = _build_grouped_contexts(matches)
        if len(contexts) >= MIN_CONTEXTS:
            logger.info(f"Found {len(contexts)} contexts (same-language)")
            return contexts
    except Exception as e:
        logger.error(f"Pinecone query error (filtered): {e}")

    # Pass B: unfiltered (cross-language)
    try:
        matches = _query_with_filter(emb, None, UNFILTERED_TOP_K)
        contexts = _build_grouped_contexts(matches)
        if contexts:
            logger.info(f"Found {len(contexts)} contexts (cross-language)")
            return contexts
    except Exception as e:
        logger.error(f"Pinecone query error (unfiltered): {e}")

    # Pass C: translate to English and search
    if TRANSLATE_FALLBACK:
        try:
            eng_q = _translate_to_english(question)
            emb_en = embed_texts([eng_q])[0]
            matches = _query_with_filter(emb_en, {"lang": "en"}, UNFILTERED_TOP_K)
            contexts = _build_grouped_contexts(matches)
            if contexts:
                logger.info(f"Found {len(contexts)} contexts (translated)")
                return contexts
        except Exception as e:
            logger.error(f"Pinecone query error (translate→en): {e}")

    logger.warning("No contexts found for question")
    return []

# ===== PROMPT BUILDING =====
def build_system_prompt(lang: str, contexts: List[str]) -> str:
    """Build system prompt with retrieved contexts"""
    lang_name = LANG_NAME.get(lang, "English")
    fb = FALLBACK.get(lang_name, FALLBACK["English"])
    joined = "\n---\n".join(contexts) if contexts else "[NO RELEVANT EXCERPTS]"

    return f"""
You are an Islamic dream interpreter analyzing visions through classical Islamic sources. Follow these guidelines:

STRICT RULES:
1) RESPOND IN USER'S LANGUAGE: Always match the language of the dream question.
2) USE ONLY CLASSICAL SOURCES: Draw only from these excerpts below. If you see exactly "[NO RELEVANT EXCERPTS]" then reply exactly: "{fb}"
3) MODERN ITEMS: Map to classical symbols as needed, and label such mappings as (generalized).

GENERALIZATION HINTS:
{GENERALIZATION_HINTS.strip()}

EXCERPTS:
{joined}

RESPONSE FORMAT:
[Interpretation in 2–3 clear sentences]

[Optional: Additional symbolic meanings if relevant]

EXAMPLE FOR "БУТЫЛКА ВО СНЕ":
Приснившаяся бутылка может символизировать скрытые эмоции или желания, которые ждут своего выражения. Также может указывать на необходимость расслабления.

Символическое значение:
• Бутылка — скрытые чувства или невыраженные желания

EXAMPLE FOR "DREAM OF PHONE":
Dreaming of a phone may represent important messages or communications coming your way.

Symbolic meaning:
• Phone — important news (interpreted from classical message symbolism)

PROHIBITED:
- Technical details (sources, confidence levels)
- Empty sections like "Uninterpreted Elements: None"
- Mixing languages in response
- Any interpretation not grounded in the excerpts above
""".strip()

def build_user_prompt(question: str, lang: str) -> str:
    """Build user prompt with question"""
    lang_name = LANG_NAME.get(lang, "English")
    return f"Interpret this dream in {lang_name} using only the excerpts provided above: {question}"

# ===== CALL OPENAI CHAT =====
def interpret(question: str) -> str:
    """
    Main interpretation function - used by both bot and web API
    TEXT ONLY - gpt-4o-mini (no images, no voice)
    """
    # Validate input
    question = question.strip()
    if len(question) < MIN_MESSAGE_LENGTH:
        raise ValueError(f"Dream too short (minimum {MIN_MESSAGE_LENGTH} characters)")
    if len(question) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"Dream too long (maximum {MAX_MESSAGE_LENGTH} characters)")

    lang = detect_language(question)
    logger.info(f"Question: '{question[:50]}...' | Detected language: {LANG_NAME.get(lang, 'unknown')} ({lang})")

    contexts = retrieve_context(question, lang)
    logger.info(f"Retrieved {len(contexts)} context blocks")
    system_prompt = build_system_prompt(lang, contexts)
    user_prompt = build_user_prompt(question, lang)

    try:
        chat = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=MAX_TOKENS_OUTPUT,
            temperature=0.3,
            timeout=OPENAI_TIMEOUT
        )
        answer = chat.choices[0].message.content.strip()
        logger.info("Successfully generated interpretation")
        return answer

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise

# ===== MAIN BOT LOOP =====
def signal_handler(sig, frame):
    """Handle graceful shutdown"""
    logger.info("Received shutdown signal, stopping bot...")
    sys.exit(0)

def main():
    """
    Main bot loop with improved error handling and rate limiting
    TEXT ONLY - handles only text messages (no images, voice, documents)
    """
    logger.info("=" * 60)
    logger.info("Dream Bot Started (Telegram)")
    logger.info("=" * 60)
    logger.info(f"Model: {CHAT_MODEL} (text-only)")
    logger.info(f"Rate limit: {MIN_REQUEST_INTERVAL}s per user")
    logger.info(f"Message length: {MIN_MESSAGE_LENGTH}-{MAX_MESSAGE_LENGTH} chars")
    logger.info(f"Note: Web API is primary channel, Telegram is secondary")
    logger.info("=" * 60)

    signal.signal(signal.SIGINT, signal_handler)

    offset = load_offset()
    if offset:
        logger.info(f"Resuming from offset {offset}")

    while True:
        try:
            updates = get_updates(offset)

            for upd in updates:
                offset = upd["update_id"] + 1

                msg = upd.get("message", {})
                chat_id = msg.get("chat", {}).get("id")
                text = msg.get("text", "")

                # Ignore non-text messages (photos, voice, documents, etc.)
                if msg.get("photo") or msg.get("voice") or msg.get("document") or msg.get("video"):
                    if chat_id:
                        send_message(
                            chat_id,
                            "⚠️ I can only interpret text descriptions of dreams.\n\n"
                            "Please describe your dream in words (English, Russian, or Kazakh)."
                        )
                    continue

                if not chat_id or not text:
                    continue

                logger.info(f"Message from {chat_id}: {text[:50]}...")

                # Handle /start command
                if text.startswith("/start"):
                    send_message(
                        chat_id,
                        "👋 Welcome to DreamWisdom Bot!\n\n"
                        "I interpret dreams using classical Islamic sources.\n\n"
                        "🌐 Languages: English, Russian, Kazakh\n"
                        f"📝 Length: {MIN_MESSAGE_LENGTH}-{MAX_MESSAGE_LENGTH} characters\n"
                        "💬 Text only (no images or voice)\n\n"
                        "Just send me your dream description and I'll interpret it!"
                    )
                    continue

                # Check rate limit
                can_proceed, remaining = check_rate_limit(chat_id)
                if not can_proceed:
                    send_message(
                        chat_id,
                        f"⏳ Please wait {int(remaining)+1} seconds before sending another dream."
                    )
                    logger.warning(f"Rate limit: chat {chat_id}, {remaining:.1f}s remaining")
                    continue

                # Validate message length
                if len(text) > MAX_MESSAGE_LENGTH:
                    send_message(
                        chat_id,
                        f"❌ Dream too long ({len(text)} characters).\n\n"
                        f"Please keep it under {MAX_MESSAGE_LENGTH} characters."
                    )
                    logger.warning(f"Message too long from {chat_id}: {len(text)} chars")
                    continue

                if len(text.strip()) < MIN_MESSAGE_LENGTH:
                    send_message(
                        chat_id,
                        f"❌ Dream too short.\n\n"
                        f"Please provide at least {MIN_MESSAGE_LENGTH} characters."
                    )
                    continue

                # Process the dream
                send_message(chat_id, "⏳ Interpreting your dream...")

                try:
                    answer = interpret(text)
                    send_message(chat_id, answer)
                    logger.info(f"✓ Success: chat {chat_id}")

                except ValueError as e:
                    # Client error (validation)
                    send_message(chat_id, f"❌ {str(e)}")
                    logger.warning(f"Validation error for {chat_id}: {e}")

                except Exception as e:
                    # Server error
                    send_message(
                        chat_id,
                        "❌ Sorry, I encountered an error while interpreting your dream.\n\n"
                        "Please try again in a moment."
                    )
                    logger.error(f"✗ Error for chat {chat_id}: {e}", exc_info=True)

            # Save offset after processing all updates
            if offset:
                save_offset(offset)

            time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            break

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    main()
