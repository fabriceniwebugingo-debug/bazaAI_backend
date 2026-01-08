"""
Telecom Bundle Chat Service - single-file FastAPI app
This version is a corrected, self-contained rewrite of your provided code.
- Preserves all original endpoints and behavior.
- Adds safer, optional Kinyarwanda language support with graceful fallbacks.
- Removes hard dependencies on heavy ML libraries; uses langdetect if available, else keyword heuristics.
- Keeps mobile-money webhook endpoint and stubs for provider integration (no external provider SDKs required).
- Uses try/except imports for optional libs so static analyzers won't error at runtime.
Install requirements for full functionality:
  pip install fastapi uvicorn python-dotenv psycopg2-binary python-multipart pydantic langdetect
Optional (for Redis support):
  pip install redis
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv
import logging
import re
from datetime import datetime, timedelta
from decimal import Decimal
import json
from typing import Optional, Dict, Any, List
import shutil

# Optional Redis support for state/cache
try:
    import redis
except Exception:
    redis = None

# Optional language detection library (fasttext preferred, else langdetect)
try:
    import fasttext
except Exception:
    fasttext = None

try:
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 0
except Exception:
    detect_langs = None

# Load env
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telecom-chat")

app = FastAPI(title="Telecom Bundle Chat Service", version="1.6.1")

# CORS middleware (adjust allow_origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up uploads directory for avatar storage and serve it
UPLOAD_DIR = os.getenv("UPLOADS_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ---------------------------
# Request models
# ---------------------------
class ChatRequest(BaseModel):
    phone: str = Field(..., description="User phone number")
    message: str = Field(..., description="User message / query")


class PurchaseRequest(BaseModel):
    phone: str = Field(..., description="User phone number")
    qp_id: int = Field(..., description="QuantityPrice ID to purchase")


class RegisterRequest(BaseModel):
    name: str = Field(..., description="User name")
    phone: str = Field(..., description="User phone number")


class ProfileUpdate(BaseModel):
    phone: str = Field(..., description="User phone number")
    name: Optional[str] = None
    bio: Optional[str] = None
    language: Optional[str] = None


# ---------------------------
# Config / helpers
# ---------------------------
ALLOWED_TABLES = [
    "users", "airtime_balance", "purchased_bundles",
    "main_category", "sub_category", "period", "quantity_price",
    "conversation_state"
]

ADMIN_PHONES = [p.strip() for p in os.getenv("ADMIN_PHONES", "").split(",") if p.strip()]

# Redis client (optional)
REDIS_URL = os.getenv("REDIS_URL")
redis_client = None
if REDIS_URL and redis:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        logger.info("Connected to Redis at %s", REDIS_URL)
    except Exception:
        logger.exception("Failed to connect to Redis; falling back to DB state store")
        redis_client = None

# Database helpers
def get_db_connection():
    try:
        return psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
        )
    except Exception:
        logger.exception("Failed to connect to DB")
        raise


def fetch_one(query: str, params=()):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchone()


def fetch_all(query: str, params=()):
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall()


def execute(query: str, params=()):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            conn.commit()


def to_float(value):
    if isinstance(value, Decimal):
        return float(value)
    return value


# Normalization helpers
def normalize_phone(phone: Optional[str]) -> Optional[str]:
    if phone is None:
        return phone
    p = phone.strip()
    p = re.sub(r"[ \-\(\)]", "", p)
    return p


def normalize(text: Optional[str]) -> str:
    return re.sub(r"[\-_/]+", " ", (text or "").strip().lower())


# ---------------------------
# Period / bundle helpers
# ---------------------------
PERIOD_MAP = {"daily": "day", "weekly": "week", "monthly": "month", "day": "day", "week": "week", "month": "month"}


def parse_period(text: Optional[str]) -> Optional[str]:
    t = normalize(text)
    for k, v in PERIOD_MAP.items():
        if k in t:
            return v
    return None


def get_period_labels(sub_name: Optional[str] = None) -> List[str]:
    if sub_name:
        rows = fetch_all(
            """
            SELECT DISTINCT p.label FROM period p
            JOIN quantity_price qp ON qp.period_id = p.id
            JOIN sub_category sc ON p.sub_id = sc.id
            WHERE sc.name = %s
            """,
            (sub_name,),
        )
    else:
        rows = fetch_all("SELECT DISTINCT label FROM period;")
    return [r["label"] for r in rows]


def match_period_label(period_token_or_word: Optional[str], sub_name: Optional[str] = None) -> Optional[str]:
    if not period_token_or_word:
        return None
    token = normalize(period_token_or_word)
    candidates = get_period_labels(sub_name)
    if not candidates:
        return None
    norm_map = {normalize(c): c for c in candidates}
    if token in norm_map:
        return norm_map[token]
    for norm, orig in norm_map.items():
        if token in norm or norm in token:
            return orig
    for k, v in PERIOD_MAP.items():
        if v == token or k == token:
            for norm, orig in norm_map.items():
                if k in norm or v in norm:
                    return orig
    return None


# ---------------------------
# Conversation state helpers (Redis or DB)
# ---------------------------
def save_conversation_state(phone: str, state: dict):
    now = datetime.utcnow()
    try:
        logger.info("Saving conversation_state for %s: stage=%s lang=%s", phone, state.get("stage"), state.get("lang"))
        if redis_client:
            redis_client.hset(f"conv:{phone}", mapping={"state": json.dumps(state), "updated_at": now.isoformat()})
            redis_client.expire(f"conv:{phone}", timedelta(hours=24))
            return
        query = """
        INSERT INTO conversation_state (phone, last_type, options, updated_at)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (phone) DO UPDATE
          SET last_type = EXCLUDED.last_type,
              options = EXCLUDED.options,
              updated_at = EXCLUDED.updated_at
        """
        execute(query, (phone, state.get("stage"), json.dumps(state), now))
    except Exception:
        logger.exception("Failed saving conversation_state for %s", phone)
        raise


def get_conversation_state(phone: str) -> Optional[dict]:
    try:
        if redis_client:
            raw = redis_client.hget(f"conv:{phone}", "state")
            if not raw:
                logger.debug("No redis conversation_state for %s", phone)
                return None
            try:
                return json.loads(raw)
            except Exception:
                return raw
        row = fetch_one("SELECT last_type, options, updated_at FROM conversation_state WHERE phone=%s", (phone,))
        if not row:
            logger.debug("No conversation_state found for %s", phone)
            return None
        opts = row.get("options")
        if not opts:
            return None
        if isinstance(opts, dict):
            return opts
        try:
            return json.loads(opts)
        except Exception:
            return opts
    except Exception:
        logger.exception("Failed reading conversation_state for %s", phone)
        return None


def clear_conversation_state(phone: str):
    try:
        if redis_client:
            redis_client.delete(f"conv:{phone}")
            logger.info("Cleared redis conversation_state for %s", phone)
            return
        execute("DELETE FROM conversation_state WHERE phone=%s", (phone,))
        logger.info("Cleared conversation_state for %s", phone)
    except Exception:
        logger.exception("Failed to clear conversation_state for %s", phone)


def set_conversation_language(phone: str, lang: str):
    state = get_conversation_state(phone) or {}
    state["lang"] = lang
    save_conversation_state(phone, state)


# ---------------------------
# Domain helpers
# ---------------------------
def fetch_user(phone: str) -> Optional[dict]:
    row = fetch_one(
        """
        SELECT u.*, COALESCE(a.balance, 0) AS airtime
        FROM users u
        LEFT JOIN airtime_balance a ON u.phone_number = a.phone_number
        WHERE u.phone_number = %s
        """,
        (phone,),
    )
    if row:
        row["airtime"] = float(row.get("airtime", 0))
        row.setdefault("avatar_url", None)
        row.setdefault("bio", None)
        row.setdefault("language", None)
        row.setdefault("loyalty_points", row.get("loyalty_points", 0))
    return row


def fetch_airtime(phone: str) -> float:
    row = fetch_one("SELECT COALESCE(balance,0) AS balance FROM airtime_balance WHERE phone_number=%s", (phone,))
    return float(row["balance"]) if row else 0.0


def update_airtime(phone: str, new_balance: float):
    execute("UPDATE airtime_balance SET balance=%s WHERE phone_number=%s", (new_balance, phone))


def fetch_bundle_balances(phone: str):
    rows = fetch_all(
        """
        SELECT pb.phone_number, mc.name AS main_category, sc.name AS sub_category,
               pb.remaining, qp.quantity, qp.price, p.label AS period, pb.purchase_date
        FROM purchased_bundles pb
        JOIN quantity_price qp ON pb.quantity_price_id = qp.id
        JOIN period p ON qp.period_id = p.id
        JOIN sub_category sc ON p.sub_id = sc.id
        JOIN main_category mc ON sc.main_id = mc.id
        WHERE pb.phone_number=%s
        ORDER BY pb.purchase_date DESC
    """,
        (phone,),
    )
    return rows


def list_main_categories() -> List[str]:
    rows = fetch_all("SELECT name FROM main_category;")
    return [r["name"] for r in rows]


def list_subcategories(main_category: Optional[str] = None) -> List[str]:
    if main_category:
        rows = fetch_all(
            """
            SELECT sc.name FROM sub_category sc
            JOIN main_category mc ON sc.main_id = mc.id
            WHERE mc.name=%s
        """,
            (main_category,),
        )
    else:
        rows = fetch_all("SELECT name FROM sub_category;")
    return [r["name"] for r in rows]


def list_periods_for_sub(sub_name: str) -> List[str]:
    rows = fetch_all(
        """
        SELECT DISTINCT p.label FROM period p
        JOIN quantity_price qp ON qp.period_id = p.id
        JOIN sub_category sc ON p.sub_id = sc.id
        WHERE sc.name = %s
    """,
        (sub_name,),
    )
    return [r["label"] for r in rows]


def fetch_bundles(main: Optional[str] = None, sub: Optional[str] = None, period: Optional[str] = None, min_quantity: Optional[float] = None):
    query = """
        SELECT qp.id AS qp_id, mc.name AS main_category, sc.name AS sub_category,
               qp.quantity, qp.price, p.label AS period
        FROM quantity_price qp
        JOIN period p ON qp.period_id = p.id
        JOIN sub_category sc ON p.sub_id = sc.id
        JOIN main_category mc ON sc.main_id = mc.id
        WHERE 1=1
    """
    params = []
    if main:
        query += " AND mc.name=%s"
        params.append(main)
    if sub:
        query += " AND sc.name=%s"
        params.append(sub)
    if period:
        query += " AND p.label=%s"
        params.append(period)
    if min_quantity:
        query += " AND qp.quantity >= %s"
        params.append(min_quantity)
    return fetch_all(query, tuple(params))


def fetch_quantity_price(qp_id: int):
    return fetch_one(
        "SELECT qp.id, qp.quantity, qp.price, p.label as period, sc.name as sub_category, mc.name as main_category FROM quantity_price qp JOIN period p ON qp.period_id = p.id JOIN sub_category sc ON p.sub_id = sc.id JOIN main_category mc ON sc.main_id = mc.id WHERE qp.id=%s",
        (qp_id,),
    )


# ---------------------------
# NLP / smalltalk / translations
# ---------------------------
TRANSLATIONS = {
    "en": {
        "greeting_intro": "Hello {name} 👋\nI can help you:\n• Check airtime balance\n• View bundle balances\n• Buy data, voice, or SMS bundles\n\nJust say something like *show bundles* to begin.",
        "show_bundles": "Show bundles",
        "my_balance": "My balance",
        "help": "Help",
        "no_bundles": "{name}, there are no bundle categories available at the moment.",
        "choose_main": "{name}, choose a main category (reply with number or name):",
        "choose_sub": "{name}, under '{main}' choose a subcategory:",
        "choose_period": "{name}, choose a period for '{sub}':",
        "bundles_matching": "{name}, bundles matching your request:",
        "reply_options_prompt": "Reply with 'option <n>' for details or 'purchase <n>' to buy.",
        "details_for_option": "Details for option {idx}:",
        "purchase_prompt": "Reply 'purchase {idx}' to complete.",
        "purchase_success": "Purchase successful. {quantity} units activated. New airtime balance: {new_balance}",
        "insufficient_funds": "Insufficient airtime balance ({balance}). Please top-up or pay via mobile money and try again.",
        "restart_msg": "{name}, conversation restarted. Ask me to show bundles to begin.",
        "back_none": "{name}, there is nothing to go back from. Ask me to show bundles to begin.",
        "default_fallback": "Hello {name}, I can help you buy bundles step-by-step. Say 'show bundles' to begin.",
        "goodbye": "Goodbye {name} 👋. If you need anything else, just say 'show bundles' or 'help'.",
        "thanks": "You're welcome {name}! Glad I could help.",
        "who_are_you": "I'm BazaAI — your telecom assistant. I can help you check airtime, view and buy bundles, and assist with account info.",
        "how_are_you": "I'm a bot 🤖 — running fine and ready to help. What would you like to do today?",
        "support_prompt": "I can help with airtime, bundles, and account info. Try 'show bundles' to browse available packages.",
    },
    "kin": {
        "greeting_intro": "Bite {name} 👋\nNdagufasha:\n• Reba amafaranga ya airtime\n• Reba bundles ziri ku konti yawe\n• Gura data, voice, cyangwa SMS bundles\n\nVuga nka 'erekana bundles' kugirango dutangire.",
        "show_bundles": "Erekana bundles",
        "my_balance": "Balance yanjye",
        "help": "Ubufasha",
        "no_bundles": "{name}, nta byiciro bya bundles biboneka ubu.",
        "choose_main": "{name}, hitamo icyiciro nyamukuru (andikira nomero cyangwa izina):",
        "choose_sub": "{name}, munsi ya '{main}' hitamo subcategory:",
        "choose_period": "{name}, hitamo igihe (period) kuri '{sub}':",
        "bundles_matching": "{name}, bundles zihuye n'icyo wasabye:",
        "reply_options_prompt": "Soma kandi usubize 'option <n>' kuri details cyangwa 'purchase <n>' kugirango ugure.",
        "details_for_option": "Ibisobanuro kuri option {idx}:",
        "purchase_prompt": "Andika 'purchase {idx}' kugirango urangize kugura.",
        "purchase_success": "Kugura byagenze neza. {quantity} units zasobanuwe. Airtime usigaranye: {new_balance}",
        "insufficient_funds": "Nta mafaranga ahagije kuri airtime ({balance}). Nyamuneka wongere amafaranga cyangwa ukoreshe mobile money.",
        "restart_msg": "{name}, ikiganiro cyongeye gutangira. Vuga 'erekana bundles' kugirango utangire.",
        "back_none": "{name}, ntaho usubira inyuma. Vuga 'erekana bundles' kugirango utangire.",
        "default_fallback": "Bite {name}, ndagufasha kugura bundles buhoro buhoro. Vuga 'erekana bundles' kugirango utangire.",
        "goodbye": "Murabeho {name} 👋. Niba ukeneye ibindi, vuga 'erekana bundles' cyangwa 'ubufasha'.",
        "thanks": "Murakoze {name}! Nishimiye kugufasha.",
        "who_are_you": "Ndi BazaAI — umufasha wawe wa telecom. Ndagufasha kureba airtime, kureba no kugura bundles, no gufasha kuri konti.",
        "how_are_you": "Ndi bot 🤖 — meze neza kandi niteguye gufasha. Wifuza gukora iki uyu munsi?",
        "support_prompt": "Ndagufasha kuri airtime, bundles, na konti. Gerageza 'erekana bundles' kugirango urebe packages.",
    },
}


def tr(key: str, lang: str = "en", **kwargs) -> str:
    if not lang or lang not in TRANSLATIONS:
        lang = "en"
    template = TRANSLATIONS[lang].get(key, TRANSLATIONS["en"].get(key, ""))
    try:
        return template.format(**kwargs)
    except Exception:
        return template


# Smalltalk helpers
GREETINGS = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "what's up",
    "sup",
    "mwaramutse",
    "mwiriwe",
    "muraho",
    "bite",
    "mbega",
]
GOODBYES = ["bye", "goodbye", "see you", "see ya", "talk later", "later", "farewell", "good night", "murabeho"]
THANKS = ["thank", "thanks", "thank you", "thx", "cheers", "much appreciated", "murakoze", "urakoze", "ndabashimiye"]


def is_greeting(text: Optional[str]) -> bool:
    if not text:
        return False
    t = normalize(text)
    return any(g in t for g in GREETINGS)


def is_goodbye(text: Optional[str]) -> bool:
    if not text:
        return False
    t = normalize(text)
    return any(g in t for g in GOODBYES)


def is_thanks(text: Optional[str]) -> bool:
    if not text:
        return False
    t = normalize(text)
    return any(g in t for g in THANKS)


def extract_category(message: Optional[str]):
    msg = normalize(message)
    mains = list_main_categories()
    subs = list_subcategories()
    found_main, found_sub = None, None
    for m in mains:
        if m and m.lower() in msg:
            found_main = m
            break
    for s in subs:
        if s and s.lower() in msg:
            found_sub = s
            break
    return found_main, found_sub


def greeting_reply(user: dict, lang: str = "en"):
    name = user.get("name") or ""
    return tr("greeting_intro", lang, name=name)


def parse_quantity(text: Optional[str]):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(gb|mb)?", (text or "").lower())
    if m:
        try:
            return float(m.group(1)), (m.group(2) or "unit")
        except Exception:
            return None, None
    # Some Kinyarwanda users may omit units; fallback to detect numbers anywhere
    m2 = re.search(r"(\d+)", (text or ""))
    if m2:
        try:
            return float(m2.group(1)), "unit"
        except Exception:
            return None, None
    return None, None


# Improved bundle intent detection: include Kinyarwanda keywords
def is_bundle_intent(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    english_keywords = [
        "buy",
        "purchase",
        "bundle",
        "subscribe",
        "get",
        "need",
        "want",
        "which",
        "show",
        "packages",
        "plans",
        "bundles",
    ]
    kinyarwanda_keywords = [
        "gura",
        "erekana",
        "konti",
        "amafaranga",
        "airtime",
        "data",
        "minuta",
        "sms",
        "butumwa",
        "gukurikirana",
        "shaka",
        "nkeneye",
        "nifuza",
        "ndashaka",
        "igiciro",
        "gura",
        "gufata",
        "kubona",
    ]
    keywords = english_keywords + kinyarwanda_keywords
    if any(k in t for k in keywords):
        return True
    if re.search(r"\b(show|list|erekana)\b.*\b(bundle|bundles|plans|packages)\b", t):
        return True
    return False


ORDINALS = {
    "one": 1,
    "first": 1,
    "1st": 1,
    "two": 2,
    "second": 2,
    "2nd": 2,
    "three": 3,
    "third": 3,
    "3rd": 3,
    "four": 4,
    "fourth": 4,
    "4th": 4,
    "five": 5,
    "fifth": 5,
    "5th": 5,
    "six": 6,
    "sixth": 6,
    "6th": 6,
    "seven": 7,
    "seventh": 7,
    "7th": 7,
    "eight": 8,
    "eighth": 8,
    "8th": 8,
    "nine": 9,
    "ninth": 9,
    "9th": 9,
    "ten": 10,
    "tenth": 10,
    "10th": 10,
    # Note: avoid hardcoding possibly incorrect Kinyarwanda ordinal words here; numeric digits are commonly used
}


def interpret_choice(user_text: str, options: list):
    t = normalize(user_text)
    if not t:
        return None
    # digits preferred
    m = re.search(r"\b(\d+)\b", t)
    if m:
        idx = int(m.group(1))
        return next((o for o in options if int(o.get("index", 0)) == idx), None)
    # ordinal words (english)
    for w, idx in ORDINALS.items():
        if re.search(rf"\b{re.escape(w)}\b", t):
            return next((o for o in options if int(o.get("index", 0)) == idx), None)
    # match by name/display parts (case-insensitive)
    for o in options:
        name = normalize(o.get("name", "") or "")
        display = normalize(o.get("display", "") or "")
        if name and name in t:
            return o
        for part in display.split():
            if part and part in t:
                return o
    return None


# ---------------------------
# Conversation flows and purchase helper
# ---------------------------
def smart_start_flow(phone: str, message: str, user: dict):
    state = get_conversation_state(phone) or {}
    lang = state.get("lang", "en")
    msg = message or ""
    main, sub = extract_category(msg)
    period = parse_period(msg)
    qty_val, qty_unit = parse_quantity(msg)
    # If user provided both main+sub+period (or sub+period), go straight to bundle_list
    if (sub and period) or (main and sub and period):
        bundles = fetch_bundles(main, sub, period, min_quantity=qty_val if qty_val else None)
        if bundles:
            options = []
            lines = [tr("bundles_matching", lang, name=user.get("name") or "")]
            for i, b in enumerate(bundles[:20], start=1):
                display = f"{b['main_category']} / {b['sub_category']} — {b['quantity']} units — {b['period']} — Price: {b['price']}"
                lines.append(f"{i}. {display}")
                options.append(
                    {
                        "index": i,
                        "display": display,
                        "qp_id": int(b["qp_id"]),
                        "name": f"{b['quantity']} {b['period']}",
                        "price": float(b["price"]),
                    }
                )
            state = {
                "stage": "bundle_list",
                "selected_main": main,
                "selected_sub": sub,
                "selected_period": period,
                "options": options,
                "lang": lang,
            }
            save_conversation_state(phone, state)
            lines.append(tr("reply_options_prompt", lang))
            return {"reply": "\n".join(lines), "options": options, "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang), tr("help", lang)]}
        if sub:
            periods = list_periods_for_sub(sub)
            if periods:
                options = [{"index": i + 1, "name": p, "display": p} for i, p in enumerate(periods)]
                state = {"stage": "period_list", "selected_main": main, "selected_sub": sub, "options": options, "lang": lang}
                save_conversation_state(phone, state)
                lines = [tr("choose_period", lang, name=user.get("name") or "", sub=sub)]
                for o in options:
                    lines.append(f"{o['index']}. {o['display']}")
                return {"reply": "\n".join(lines), "options": options, "quick_replies": [tr("show_bundles", lang), tr("help", lang)]}
    if main and sub and not period:
        periods = list_periods_for_sub(sub)
        if periods:
            options = [{"index": i + 1, "name": p, "display": p} for i, p in enumerate(periods)]
            state = {"stage": "period_list", "selected_main": main, "selected_sub": sub, "options": options, "lang": lang}
            save_conversation_state(phone, state)
            lines = [tr("choose_period", lang, name=user.get("name") or "", sub=sub)]
            for o in options:
                lines.append(f"{o['index']}. {o['display']}")
            return {"reply": "\n".join(lines), "options": options, "quick_replies": [tr("show_bundles", lang)]}
    if sub and not main:
        periods = list_periods_for_sub(sub)
        if periods:
            options = [{"index": i + 1, "name": p, "display": p} for i, p in enumerate(periods)]
            state = {"stage": "period_list", "selected_main": None, "selected_sub": sub, "options": options, "lang": lang}
            save_conversation_state(phone, state)
            lines = [tr("choose_period", lang, name=user.get("name") or "", sub=sub)]
            for o in options:
                lines.append(f"{o['index']}. {o['display']}")
            return {"reply": "\n".join(lines), "options": options, "quick_replies": [tr("help", lang)]}
    if main and not sub:
        subs = list_subcategories(main)
        if subs:
            options = [{"index": i + 1, "name": s, "display": s} for i, s in enumerate(subs)]
            state = {"stage": "sub_list", "selected_main": main, "options": options, "lang": lang}
            save_conversation_state(phone, state)
            lines = [tr("choose_sub", lang, name=user.get("name") or "", main=main)]
            for o in options:
                lines.append(f"{o['index']}. {o['display']}")
            return {"reply": "\n".join(lines), "options": options, "quick_replies": [tr("help", lang)]}
    if qty_val:
        bundles = fetch_bundles(None, None, period, min_quantity=qty_val)
        if bundles:
            options = []
            lines = [tr("bundles_matching", lang, name=user.get("name") or "")]
            for i, b in enumerate(bundles[:20], start=1):
                display = f"{b['main_category']} / {b['sub_category']} — {b['quantity']} units — {b['period']} — Price: {b['price']}"
                lines.append(f"{i}. {display}")
                options.append(
                    {
                        "index": i,
                        "display": display,
                        "qp_id": int(b["qp_id"]),
                        "name": f"{b['quantity']} {b['period']}",
                        "price": float(b["price"]),
                    }
                )
            state = {"stage": "bundle_list", "options": options, "lang": lang}
            save_conversation_state(phone, state)
            lines.append(tr("reply_options_prompt", lang))
            return {"reply": "\n".join(lines), "options": options, "quick_replies": [tr("show_bundles", lang)]}
    mains = list_main_categories()
    if not mains:
        return {"reply": tr("no_bundles", lang, name=user.get("name") or "")}
    options = [{"index": i + 1, "name": m, "display": m} for i, m in enumerate(mains)]
    state = {"stage": "main_list", "options": options, "lang": lang}
    save_conversation_state(phone, state)
    lines = [tr("choose_main", lang, name=user.get("name") or "")]
    for o in options:
        lines.append(f"{o['index']}. {o['display']}")
    return {"reply": "\n".join(lines), "options": options, "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang), tr("help", lang)]}


def process_purchase(phone: str, qp_id: int, lang: str = "en") -> Dict[str, Any]:
    logger.info("Processing purchase helper for %s qp_id=%s", phone, qp_id)
    user = fetch_user(phone)
    if not user:
        return {"reply": "User not found", "status": "error", "code": 404}
    qp = fetch_quantity_price(qp_id)
    if not qp:
        return {"reply": "Bundle / option not found", "status": "error", "code": 404}
    price = float(qp.get("price") or 0.0)
    user_balance = fetch_airtime(phone)
    if user_balance < price:
        return {"reply": tr("insufficient_funds", lang, balance=user_balance), "status": "insufficient_funds"}
    try:
        new_balance = round(user_balance - price, 2)
        update_airtime(phone, new_balance)
        now = datetime.utcnow()
        try:
            execute(
                """
                INSERT INTO purchased_bundles (phone_number, quantity_price_id, remaining, purchase_date)
                VALUES (%s, %s, %s, %s)
            """,
                (phone, qp_id, qp.get("quantity", 0), now),
            )
        except Exception:
            logger.exception("Failed to insert purchased_bundles row; please ensure table schema matches")
        # Update loyalty points (best-effort)
        try:
            execute("UPDATE users SET loyalty_points = COALESCE(loyalty_points,0) + %s WHERE phone_number=%s", (int(qp.get("quantity", 0)), phone))
        except Exception:
            logger.debug("Could not update loyalty points (maybe missing column).")
        receipt = {
            "phone": phone,
            "bundle": f"{qp.get('main_category')} / {qp.get('sub_category')} {qp.get('quantity')} {qp.get('period')}",
            "price": price,
            "new_balance": new_balance,
            "timestamp": now.isoformat(),
        }
        logger.info("Purchase successful for %s: %s", phone, receipt)
        return {"reply": tr("purchase_success", lang, quantity=qp.get("quantity"), new_balance=new_balance), "status": "ok", "receipt": receipt}
    except Exception:
        logger.exception("Purchase failed for %s", phone)
        return {"reply": "Purchase failed due to server error.", "status": "error", "code": 500}


def smalltalk_reply(user: dict, message: str, lang: str = "en"):
    name = user.get("name") or ""
    t = (message or "").lower()
    if is_greeting(t):
        return {"reply": greeting_reply(user, lang), "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang), tr("help", lang)]}
    if is_goodbye(t):
        return {"reply": tr("goodbye", lang, name=name), "quick_replies": [tr("show_bundles", lang), tr("help", lang)]}
    if is_thanks(t):
        return {"reply": tr("thanks", lang, name=name), "quick_replies": [tr("show_bundles", lang), tr("help", lang)]}
    if re.search(r"\b(who are you|what are you|your name|uri nde)\b", t):
        return {"reply": tr("who_are_you", lang)}
    if re.search(r"\b(how are you|how's it going|how you doing|amere)\b", t):
        return {"reply": tr("how_are_you", lang)}
    if re.search(r"\b(help|support|assist|can you|ufasha|ubufasha)\b", t):
        return {"reply": tr("support_prompt", lang), "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang)]}
    return None


# ---------------------------
# Profile & endpoints
# ---------------------------
@app.post("/register")
def register(req: RegisterRequest):
    phone = normalize_phone(req.phone or "")
    name = (req.name or "").strip()
    logger.info("Register request: phone=%s name=%s", phone, name)
    if not phone or not name:
        raise HTTPException(status_code=400, detail="Name and phone are required")
    try:
        # Upsert user
        try:
            execute(
                """
                INSERT INTO users (phone_number, name)
                VALUES (%s, %s)
                ON CONFLICT (phone_number) DO UPDATE
                  SET name = EXCLUDED.name
            """,
                (phone, name),
            )
        except Exception:
            logger.exception("Failed upsert into users (ensure users table exists with phone_number and name columns)")

        # Ensure airtime_balance row exists (best-effort)
        try:
            row = fetch_one("SELECT phone_number FROM airtime_balance WHERE phone_number=%s", (phone,))
            if not row:
                try:
                    execute("INSERT INTO airtime_balance (phone_number, balance) VALUES (%s, %s)", (phone, 0.0))
                except Exception:
                    logger.exception("Failed to insert airtime_balance")
        except Exception:
            logger.exception("Failed checking/creating airtime_balance row")

        return {"status": "ok", "reply": f"Registered {name} with phone {phone}"}
    except Exception:
        logger.exception("Registration failed for %s", phone)
        raise HTTPException(status_code=500, detail="Registration failed")


@app.get("/profile")
def get_profile(phone: str):
    phone = normalize_phone(phone or "")
    user = fetch_user(phone)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Recent purchases
    purchases = fetch_all(
        """
        SELECT pb.id, pb.quantity_price_id AS qp_id, qp.price, qp.quantity, p.label AS period, sc.name AS sub_category, pb.purchase_date
        FROM purchased_bundles pb
        LEFT JOIN quantity_price qp ON pb.quantity_price_id = qp.id
        LEFT JOIN period p ON qp.period_id = p.id
        LEFT JOIN sub_category sc ON p.sub_id = sc.id
        WHERE pb.phone_number = %s
        ORDER BY pb.purchase_date DESC
        LIMIT 20
    """,
        (phone,),
    )
    # Simple recommendations: pick top bundles from same sub_category as last purchase
    recommendations = []
    if purchases:
        last_sub = purchases[0].get("sub_category")
        if last_sub:
            recs = fetch_bundles(None, last_sub, None)
            for r in recs[:6]:
                recommendations.append(
                    {
                        "qp_id": int(r["qp_id"]),
                        "display": f"{r['main_category']} / {r['sub_category']} — {r['quantity']} units — {r['period']} — Price: {r['price']}",
                        "price": float(r["price"]),
                    }
                )
    if not recommendations:
        top = fetch_all(
            "SELECT qp.id AS qp_id, qp.quantity, qp.price, p.label AS period, sc.name AS sub_category, mc.name AS main_category FROM quantity_price qp JOIN period p ON qp.period_id=p.id JOIN sub_category sc ON p.sub_id=sc.id JOIN main_category mc ON sc.main_id=mc.id ORDER BY qp.price ASC LIMIT 6;"
        )
        for r in top:
            recommendations.append(
                {
                    "qp_id": int(r["qp_id"]),
                    "display": f"{r['main_category']} / {r['sub_category']} — {r['quantity']} units — {r['period']} — Price: {r['price']}",
                    "price": float(r["price"]),
                }
            )
    loyalty = user.get("loyalty_points", 0)
    resp = {
        "name": user.get("name"),
        "phone": user.get("phone_number") or phone,
        "avatar_url": user.get("avatar_url"),
        "bio": user.get("bio"),
        "language": user.get("language"),
        "airtime": user.get("airtime", 0),
        "loyalty_points": loyalty,
        "recent_purchases": purchases,
        "recommendations": recommendations,
    }
    return resp


@app.patch("/profile")
def update_profile(payload: ProfileUpdate):
    phone = normalize_phone(payload.phone or "")
    user = fetch_user(phone)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    set_parts = []
    params = []
    if payload.name is not None:
        set_parts.append("name=%s")
        params.append(payload.name.strip())
    if payload.bio is not None:
        set_parts.append("bio=%s")
        params.append(payload.bio.strip())
    if payload.language is not None:
        set_parts.append("language=%s")
        params.append(payload.language)
    if not set_parts:
        return {"status": "ok", "reply": "No changes"}
    sql = f"UPDATE users SET {', '.join(set_parts)} WHERE phone_number=%s"
    params.append(phone)
    try:
        execute(sql, tuple(params))
    except Exception:
        logger.exception("Failed updating profile for %s", phone)
        raise HTTPException(status_code=500, detail="Update failed")
    return {"status": "ok", "reply": "Profile updated"}


@app.post("/profile/avatar")
async def upload_avatar(phone: str = Form(...), avatar: UploadFile = File(...)):
    phone = normalize_phone(phone or "")
    user = fetch_user(phone)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    filename = f"{phone}_{int(datetime.utcnow().timestamp())}_{avatar.filename}"
    dest = os.path.join(UPLOAD_DIR, filename)
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(avatar.file, f)
    except Exception:
        logger.exception("Failed saving uploaded avatar")
        raise HTTPException(status_code=500, detail="Failed to save file")
    avatar_url = f"/uploads/{filename}"
    try:
        execute("UPDATE users SET avatar_url=%s WHERE phone_number=%s", (avatar_url, phone))
    except Exception:
        logger.exception("Failed to save avatar_url to DB")
    return {"status": "ok", "avatar_url": avatar_url, "reply": "Avatar uploaded"}


@app.get("/purchases")
def get_purchases(phone: str, limit: int = 20):
    phone = normalize_phone(phone or "")
    user = fetch_user(phone)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    purchases = fetch_all(
        """
        SELECT pb.id, pb.quantity_price_id AS qp_id, qp.price, qp.quantity, p.label AS period, sc.name AS sub_category, pb.purchase_date
        FROM purchased_bundles pb
        LEFT JOIN quantity_price qp ON pb.quantity_price_id = qp.id
        LEFT JOIN period p ON qp.period_id = p.id
        LEFT JOIN sub_category sc ON p.sub_id = sc.id
        WHERE pb.phone_number = %s
        ORDER BY pb.purchase_date DESC
        LIMIT %s
    """,
        (phone, limit),
    )
    return {"purchases": purchases}


@app.get("/recommendations")
def get_recommendations(phone: str, limit: int = 6):
    phone = normalize_phone(phone or "")
    purchases = fetch_all(
        "SELECT pb.quantity_price_id as qp_id FROM purchased_bundles pb WHERE pb.phone_number=%s ORDER BY pb.purchase_date DESC LIMIT 1",
        (phone,),
    )
    recommendations = []
    if purchases:
        last_qp = purchases[0].get("qp_id")
        if last_qp:
            qp = fetch_quantity_price(last_qp)
            if qp:
                sub_name = qp.get("sub_category")
                recs = fetch_bundles(None, sub_name, None)
                for r in recs[:limit]:
                    recommendations.append(
                        {
                            "qp_id": int(r["qp_id"]),
                            "display": f"{r['main_category']} / {r['sub_category']} — {r['quantity']} units — {r['period']} — Price: {r['price']}",
                            "price": float(r["price"]),
                        }
                    )
    if not recommendations:
        top = fetch_all(
            "SELECT qp.id AS qp_id, qp.quantity, qp.price, p.label AS period, sc.name AS sub_category, mc.name AS main_category FROM quantity_price qp JOIN period p ON qp.period_id=p.id JOIN sub_category sc ON p.sub_id=sc.id JOIN main_category mc ON sc.main_id=mc.id ORDER BY qp.price ASC LIMIT %s;",
            (limit,),
        )
        for r in top:
            recommendations.append(
                {
                    "qp_id": int(r["qp_id"]),
                    "display": f"{r['main_category']} / {r['sub_category']} — {r['quantity']} units — {r['period']} — Price: {r['price']}",
                    "price": float(r["price"]),
                }
            )
    return {"recommendations": recommendations}


# ---------------------------
# Chat & purchase endpoints (kept intact while integrating Kinyarwanda improvements)
# ---------------------------

# Extra token-based vocab lists for lightweight detection scoring
_KIN_VOCAB = set(
    [
        "gura",
        "erekana",
        "konti",
        "amafaranga",
        "airtime",
        "data",
        "minuta",
        "sms",
        "butumwa",
        "gukurikirana",
        "shaka",
        "nkeneye",
        "nifuza",
        "ndashaka",
        "igiciro",
        "gufata",
        "kubona",
        "muraho",
        "murabeho",
        "urakoze",
        "ndabashimiye",
        "mwaramutse",
        "mwiriwe",
        "bite",
        "erekana",
        "gereza",
        "ubufasha",
        "ubutumwa",
        "konti",
        "gura",
    ]
)

_EN_VOCAB = set(
    [
        "buy",
        "purchase",
        "bundle",
        "subscribe",
        "get",
        "need",
        "want",
        "which",
        "show",
        "packages",
        "plans",
        "data",
        "minutes",
        "sms",
        "price",
        "help",
        "support",
        "account",
        "balance",
    ]
)


def detect_language_simple(text: str, user_lang_hint: Optional[str] = None) -> str:
    """
    Returns 'kin' for Kinyarwanda, 'en' otherwise.
    Uses (in order): stored user language hint (if provided), fasttext if available, langdetect fallback,
    then a token-scoring heuristic using Kinyarwanda and English small vocab lists.
    This approach is conservative and aims to prefer Kinyarwanda only when there are clear signals.
    """
    if not text or text.strip() == "":
        # No text -> prefer user hint or English
        return user_lang_hint if user_lang_hint in ("kin", "en") else "en"

    t = text.strip()

    # If caller provided explicit language hint (e.g., user's profile), respect it strongly
    if user_lang_hint and user_lang_hint in ("kin", "en"):
        return user_lang_hint

    # Fasttext (if model loaded by user; no auto-download here)
    try:
        if fasttext:
            model_path = os.getenv("FASTTEXT_LANG_MODEL_PATH")
            if model_path and os.path.exists(model_path):
                ft = fasttext.load_model(model_path)
                labels, probs = ft.predict(t, k=1)
                if labels and len(labels) > 0:
                    code = labels[0].replace("__label__", "")
                    if code.startswith("rw") or code == "kin":
                        return "kin"
                    if code.startswith("en"):
                        return "en"
    except Exception:
        logger.debug("fasttext detection failed or not configured")

    # langdetect fallback
    try:
        if detect_langs:
            langs = detect_langs(t)
            if langs:
                top = langs[0]
                lang_code = top.lang
                if lang_code in ("rw", "kin"):
                    return "kin"
                if lang_code.startswith("en"):
                    return "en"
    except Exception:
        logger.debug("langdetect failed")

    # Token scoring heuristic
    tokens = re.findall(r"\w+", t.lower())
    kin_score = sum(1 for tok in tokens if tok in _KIN_VOCAB)
    en_score = sum(1 for tok in tokens if tok in _EN_VOCAB)

    # Additionally, count presence of numeric-only messages (likely bundle queries) and rely on other flows
    # Decide using sensible thresholds: require at least 1 more kin token than en tokens, or kin_score >=2 and >= en_score
    if kin_score >= 2 and kin_score >= en_score:
        return "kin"
    if kin_score - en_score >= 1:
        return "kin"

    # Heuristic single-keyword triggers (commonly used Kinyarwanda commands)
    for kw in _KIN_VOCAB:
        if kw in t.lower():
            return "kin"

    # Default fallback to English
    return "en"


@app.post("/chat")
def chat(req: ChatRequest):
    phone = normalize_phone(req.phone or "")
    message = req.message or ""
    logger.info("Incoming chat from %s: %s", phone, message)
    user = fetch_user(phone)
    if not user:
        alt_phone = phone.lstrip("+") if phone.startswith("+") else "+" + phone
        user = fetch_user(alt_phone)
        if user:
            phone = alt_phone
        else:
            raise HTTPException(status_code=404, detail="User not found")

    # Prefer user's stored language (if any) as a hint to detection
    user_lang_hint = None
    try:
        user_lang_hint = user.get("language")
        if user_lang_hint and user_lang_hint not in ("kin", "en"):
            user_lang_hint = None
    except Exception:
        user_lang_hint = None

    # Detect user language (respect user hint if present)
    detected = detect_language_simple(message or "", user_lang_hint)
    user_lang = "kin" if detected == "kin" else "en"

    # If detected Kinyarwanda and user state not set, set it
    try:
        set_conversation_language(phone, user_lang)
    except Exception:
        logger.debug("Could not set conversation language")

    # smalltalk on raw message
    small = smalltalk_reply(user, message, user_lang)
    if small:
        return small

    lower = (message or "").lower().strip()
    state = get_conversation_state(phone) or {}
    lang = state.get("lang", user_lang)

    # Balance query
    if re.search(r"\b(airtime|account balance|my balance|balance|balance yanjye|amafaranga|balance yange)\b", lower) and "bundle" not in lower:
        return {"reply": f"{user.get('name')}, {tr('my_balance', lang)}: {fetch_airtime(phone)}", "quick_replies": [tr("show_bundles", lang), tr("help", lang)]}

    # "purchase <n>" in chat
    m_purchase = re.match(r"^\s*(?:purchase|buy)\s+(\d+)\b", lower)
    if m_purchase:
        idx = int(m_purchase.group(1))
        state = get_conversation_state(phone)
        if not state or state.get("stage") != "bundle_list":
            return {"reply": tr("no_bundles", lang, name=user.get("name") or ""), "quick_replies": [tr("show_bundles", lang)]}
        options = state.get("options", [])
        chosen = next((o for o in options if int(o.get("index", 0)) == idx), None)
        if not chosen:
            return {"reply": f"{user.get('name')}, option {idx} not found. Please choose a valid bundle number.", "options": options}
        result = process_purchase(phone, int(chosen.get("qp_id")), lang)
        if result.get("status") == "ok":
            clear_conversation_state(phone)
            return {"reply": result.get("reply"), "status": "ok", "receipt": result.get("receipt"), "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang)]}
        elif result.get("status") == "insufficient_funds":
            return {"reply": result.get("reply"), "status": "insufficient_funds", "quick_replies": [tr("help", lang)]}
        else:
            return {"reply": result.get("reply"), "status": result.get("status")}

    # Smart start or other flows
    if is_bundle_intent(lower) or extract_category(lower)[0] or extract_category(lower)[1] or parse_period(lower) or parse_quantity(lower)[0]:
        return smart_start_flow(phone, message, user)

    # Fallback
    return {"reply": f"Hello {user.get('name')}, I can help you buy bundles step-by-step. Say 'show bundles' to begin.", "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang), tr("help", lang)]}


@app.post("/purchase")
def purchase(req: PurchaseRequest):
    phone = normalize_phone(req.phone or "")
    qp_id = req.qp_id
    logger.info("Purchase request from %s for qp_id=%s", phone, qp_id)
    state = get_conversation_state(phone) or {}
    lang = state.get("lang", "en")
    result = process_purchase(phone, qp_id, lang)
    if result.get("status") == "error" and result.get("code") == 404:
        raise HTTPException(status_code=404, detail=result.get("reply"))
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("reply"))
    if result.get("status") == "ok":
        clear_conversation_state(phone)
    return result


@app.post("/payment-webhook")
def payment_webhook(payload: dict, background_tasks: BackgroundTasks):
    logger.info("Payment webhook received: %s", payload)

    def process(p):
        try:
            phone = normalize_phone(p.get("phone"))
            status = p.get("status", "unknown")
            amount = float(p.get("amount", 0))
            if status == "success":
                row = fetch_one("SELECT COALESCE(balance,0) AS balance FROM airtime_balance WHERE phone_number=%s", (phone,))
                current = float(row["balance"]) if row else 0.0
                new_balance = round(current + amount, 2)
                try:
                    execute("UPDATE airtime_balance SET balance=%s WHERE phone_number=%s", (new_balance, phone))
                except Exception:
                    try:
                        execute("INSERT INTO airtime_balance (phone_number, balance) VALUES (%s, %s)", (phone, new_balance))
                    except Exception:
                        logger.exception("Failed to update/insert airtime balance for %s", phone)
                logger.info("Payment applied to %s: +%s new_balance=%s", phone, amount, new_balance)
            else:
                logger.info("Payment status not success (%s) for %s", status, phone)
        except Exception:
            logger.exception("Failed processing payment webhook")

    background_tasks.add_task(process, payload)
    return {"status": "ok"}


@app.post("/debug/echo")
async def debug_echo(request: Request):
    payload = await request.json()
    phone = normalize_phone(payload.get("phone", ""))
    state = get_conversation_state(phone)
    logger.info("DEBUG ECHO phone=%s payload=%s state=%s", phone, payload, state)
    return {"payload": payload, "state": state}