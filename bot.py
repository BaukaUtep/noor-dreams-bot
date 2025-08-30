import os
import time
import requests
import re
import signal
import sys
from typing import List

from openai import OpenAI
from pinecone import Pinecone

# ===== 1. CONFIG =====
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENVIRONMENT")    # may be None in new SDKs
PINECONE_INDEX   = "noor-dreams"

EMBED_MODEL      = "text-embedding-3-small"
EMBED_DIM        = 1536            # must match your index dimension
CHAT_MODEL       = "gpt-4o-mini"  
TOP_K            = 5
TELEGRAM_API     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Fallback messages
FALLBACK = {
    "English": "This interpretation is not present in the provided excerpts.",
    "Russian": "Этого толкования нет в предоставленных отрывках.",
    "Kazakh":  "Түсіндіру берілген үзінділерде жоқ."
}

# Allowed mild generalizations (modern -> classical symbolic family)
GENERALIZATION_HINTS = """
Allowed mild generalizations (label them):
phone/mobile -> communication/message (generalized)
talking/speaking/chatting -> communication/speech (generalized)
car/driving/vehicle -> journey/travel (generalized)
computer/laptop -> record/knowledge (generalized)
"""

# ===== 2. INIT CLIENTS =====
if not TELEGRAM_TOKEN:
    raise RuntimeError("No TELEGRAM_TOKEN set")
if not OPENAI_API_KEY:
    raise RuntimeError("No OPENAI_API_KEY set")
if not PINECONE_API_KEY:
    raise RuntimeError("No PINECONE_API_KEY set")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV) if PINECONE_ENV else Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# ===== 3. TELEGRAM HELPERS =====
def get_updates(offset=None, timeout=30):
    try:
        r = requests.get(
            f"{TELEGRAM_API}/getUpdates",
            params={"offset": offset, "timeout": timeout},
            timeout=timeout + 5,
        )
        r.raise_for_status()
        return r.json().get("result", [])
    except requests.exceptions.Timeout:
        print("get_updates timeout, returning empty list")
        return []
    except requests.exceptions.RequestException as e:
        print("get_updates request error:", e)
        return []
    except Exception as e:
        print("get_updates unexpected error:", e)
        return []

def send_message(chat_id, text):
    try:
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
    except Exception as e:
        print("send_message error:", e)

# ===== 4. LANGUAGE DETECTION =====
def detect_language(text: str) -> str:
    t = text.lower()
    # Kazakh distinctive letters
    if re.search(r"[ңғүұқәіөһ]", t):
        return "kk"
    # Cyrillic (assume Russian if not Kazakh-specific)
    if re.search(r"[А-Яа-яЁё]", text):
        return "ru"
    return "en"

LANG_NAME = {
    "en": "English",
    "ru": "Russian",
    "kk": "Kazakh"
}

# ===== 5. EMBEDDINGS =====
def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIM
    )
    return [d.embedding for d in resp.data]

# ===== 6. RETRIEVAL (robust multilingual) =====
MIN_CONTEXTS     = 2         # if we get less, try the next pass
PER_SYMBOL_MAX   = 3         # cap sentences per symbol to keep prompt tight
UNFILTERED_TOP_K = 20        # expand when we remove the lang filter
TRANSLATE_FALLBACK = True    # set False if you don't want pass (C)

def _build_grouped_contexts(matches):
    """
    Group sentences by symbol_id and keep up to PER_SYMBOL_MAX per symbol.
    Accepts metadata keys from your schema:
      symbol_id, symbol_label, sentence_index, text, lang
    """
    buckets = {}  # (symbol_id) -> {"label": str, "items": [(sent_idx, text, lang)]}
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

    # turn into prompt-ready blocks
    contexts = []
    for sid, data in buckets.items():
        label = data["label"]
        lines = [f"[{label} · {sid}]"]
        for (idx, text, lg) in data["items"]:
            # include sentence index; language tag helps when it's cross-lang
            lg_tag = f" ({lg})" if lg else ""
            lines.append(f"• (sent {idx}){lg_tag} {text}")
        contexts.append("\n".join(lines))
    return contexts

def _query_with_filter(emb, filter_obj, top_k):
    res = index.query(vector=emb, top_k=top_k, include_metadata=True, filter=filter_obj)
    return res.get("matches") if isinstance(res, dict) else getattr(res, "matches", [])

def _translate_to_english(text: str) -> str:
    # lightweight translation using your current chat model
    chat = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Translate to English. Keep names and religious terms intact."},
            {"role": "user",   "content": text}
        ],
        max_tokens=800
    )
    return chat.choices[0].message.content.strip()

def retrieve_context(question: str, lang_code: str, top_k=TOP_K):
    # Pass A: same-language
    emb = embed_texts([question])[0]
    try:
        matches = _query_with_filter(emb, {"lang": lang_code}, top_k)
        contexts = _build_grouped_contexts(matches)
        if len(contexts) >= MIN_CONTEXTS:
            return contexts
    except Exception as e:
        print("Pinecone query error (filtered):", e)

    # Pass B: unfiltered (cross-language)
    try:
        matches = _query_with_filter(emb, None, UNFILTERED_TOP_K)
        contexts = _build_grouped_contexts(matches)
        if contexts:
            return contexts
    except Exception as e:
        print("Pinecone query error (unfiltered):", e)

    # Pass C: translate query to EN and filter on lang="en"
    if TRANSLATE_FALLBACK:
        try:
            eng_q = _translate_to_english(question)
            emb_en = embed_texts([eng_q])[0]
            matches = _query_with_filter(emb_en, {"lang": "en"}, UNFILTERED_TOP_K)
            contexts = _build_grouped_contexts(matches)
            if contexts:
                return contexts
        except Exception as e:
            print("Pinecone query error (translate→en):", e)

    return []

# ===== 7. PROMPT BUILDING =====
def build_system_prompt(lang: str, contexts: List[str]) -> str:
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
    lang_name = LANG_NAME.get(lang, "English")
    return f"Interpret this dream in {lang_name} using only the excerpts provided above: {question}"

# ===== 8. CALL OPENAI CHAT =====
def interpret(question: str) -> str:
    lang = detect_language(question)
    contexts = retrieve_context(question, lang)
    system_prompt = build_system_prompt(lang, contexts)
    user_prompt = build_user_prompt(question, lang)

    chat = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=800,
        temperature=0.3
    )
    return chat.choices[0].message.content.strip()

# ===== 9. MAIN LOOP =====
def signal_handler(sig, frame):
    print("\nShutting down gracefully...")
    sys.exit(0)

def main():
    print("Dream bot started.")
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    offset = None
    while True:
        try:
            updates = get_updates(offset)
            for upd in updates:
                offset = upd["update_id"] + 1
                msg = upd.get("message", {})
                chat_id = msg.get("chat", {}).get("id")
                text = msg.get("text", "")
                if not chat_id or not text:
                    continue

                if text.startswith("/start"):
                    send_message(
                        chat_id,
                        "👋 Welcome! Send me your dream (English, Russian or Kazakh) and I will interpret it using classical Islamic sources."
                    )
                    continue

                send_message(chat_id, "⏳ Interpreting your dream...")
                try:
                    answer = interpret(text)
                except Exception as e:
                    print("Interpret error:", e)
                    answer = "❌ Error. Please try again later."
                send_message(chat_id, answer)
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            break
        except Exception as e:
            print("Main loop error:", e)
            time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    main()