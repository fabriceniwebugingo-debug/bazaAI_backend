from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, validator, ValidationError

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import psycopg2
import psycopg2.extras

import os

from dotenv import load_dotenv

import logging

import re

from datetime import datetime, timedelta

from decimal import Decimal

import json

from typing import Optional, Dict, Any, List, Tuple

from pathlib import Path

import shutil

import tempfile

import time

import hashlib

import random



# Whisper import

import whisper



# Additional ASR imports for Kinyarwanda support

try:

    import speech_recognition as sr

except ImportError:

    sr = None



# For custom language model loading

try:

    import torch

    import torchaudio

except ImportError:

    torch = None

    torchaudio = None



# New imports for enhancements

import nltk

from sumy.parsers.plaintext import PlaintextParser

from sumy.nlp.tokenizers import Tokenizer

from sumy.summarizers.lsa import LsaSummarizer

import sentry_sdk

from sentry_sdk.integrations.fastapi import FastApiIntegration

from slowapi import Limiter, _rate_limit_exceeded_handler

from slowapi.util import get_remote_address

from slowapi.errors import RateLimitExceeded



# Optional Redis support for state/cache

try:

    import redis

except Exception:

    redis = None



# Optional language detection libraries

try:

    import fasttext

except Exception:

    fasttext = None



try:

    from langdetect import detect_langs, DetectorFactory

    DetectorFactory.seed = 0

except Exception:

    detect_langs = None



# ----------------------------------------------------------------------

# Sentry initialization (requires SENTRY_DSN env var)

# Note: Basic logging setup first for Sentry initialization messages

load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("telecom-chat")



try:

    sentry_sdk.init(

        dsn=os.getenv("SENTRY_DSN"),

        integrations=[FastApiIntegration()],

        traces_sample_rate=0.1,

        environment=os.getenv("ENVIRONMENT", "development")

    )

    logger.info("Sentry initialized successfully")

except Exception as e:

    logger.warning(f"Failed to initialize Sentry: {e}")

# ----------------------------------------------------------------------



app = FastAPI(title="Telecom Bundle Chat Service", version="2.7.0")



# ----------------------------------------------------------------------

# Rate limiter setup

limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ----------------------------------------------------------------------



# Add validation error handler for detailed debugging
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error for {request.method} {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "details": exc.errors(),
            "message": "Request body validation failed. Check required fields and formats.",
            "required_fields": {
                "register": ["name", "phone", "pin", "confirm_pin"],
                "pin_requirements": "Exactly 5 digits (0-9)",
                "example": {
                    "name": "John Doe",
                    "phone": "0781234567",
                    "pin": "12345",
                    "confirm_pin": "12345"
                }
            }
        }
    )

app.add_middleware(

    CORSMiddleware,

    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),

    allow_credentials=True,

    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],

    allow_headers=["*"],

)



UPLOAD_DIR = os.getenv("UPLOADS_DIR", "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")



@app.on_event("startup")

async def startup_event():

    """Initialize database tables on startup"""

    try:

        initialize_database()

        logger.info("Application startup completed successfully")

    except Exception as e:

        logger.exception("Failed to initialize application")

        raise



# Load Whisper models once at startup (lazy loading to prevent memory issues)

whisper_models = {}

kinyarwanda_models = {}

model_config = None



def load_model_config():

    """Load model configuration from file"""

    global model_config

    config_path = Path("models/kinyarwanda_trained/model_config.json")

    if config_path.exists():

        with open(config_path, 'r') as f:

            model_config = json.load(f)

        logger.info(f"Loaded model config with {len(model_config.get('available_models', []))} models")

    else:

        model_config = {"available_models": [], "model_paths": {}}

        logger.info("No model config found, using fallback")



def get_kinyarwanda_trained_model(model_name: str = None):

    """

    Load a trained Kinyarwanda model from downloaded models

    """

    global kinyarwanda_models

    

    if model_config is None:

        load_model_config()

    

    # Use default model if none specified

    if model_name is None:

        model_name = model_config.get("default_model", "whisper-kinyarwanda-base")

    

    # Check if model is available

    if model_name not in model_config.get("available_models", []):

        logger.warning(f"Trained Kinyarwanda model '{model_name}' not available")

        return None

    

    # Load model if not cached

    if model_name not in kinyarwanda_models:

        try:

            model_path = model_config["model_paths"][model_name]

            logger.info(f"Loading trained Kinyarwanda model: {model_name}")

            

            # Load based on model type

            if model_name.startswith("whisper"):

                # Whisper model

                kinyarwanda_models[model_name] = whisper.load_model(model_path)

            elif model_name.startswith("wav2vec2"):

                # Wav2Vec2 model

                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

                processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xls-r-300m-kinyarwanda")

                model = Wav2Vec2ForCTC.from_pretrained(model_path)

                kinyarwanda_models[model_name] = {"model": model, "processor": processor}

            else:

                # Generic model loading

                import torch

                kinyarwanda_models[model_name] = torch.load(model_path)

            

            logger.info(f"Successfully loaded trained Kinyarwanda model: {model_name}")

            

        except Exception as e:

            logger.error(f"Failed to load trained Kinyarwanda model '{model_name}': {e}")

            return None

    

    return kinyarwanda_models[model_name]



def get_whisper_model(model_size="base"):
    """Get Whisper model with caching for different sizes"""
    global whisper_models

    if model_size not in whisper_models:
        try:
            logger.info(f"Loading Whisper model '{model_size}' - this may take a moment...")
            whisper_models[model_size] = whisper.load_model(model_size)
            logger.info(f"Whisper model '{model_size}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_size}': {e}")
            # Fallback to tiny model if base fails
            if model_size != "tiny":
                logger.info("Attempting fallback to tiny model...")
                try:
                    whisper_models[model_size] = whisper.load_model("tiny")
                    logger.info("Fallback tiny model loaded successfully")
                except Exception as fallback_error:
                    logger.error(f"Failed to load fallback model: {fallback_error}")
                    raise HTTPException(status_code=500, detail="Voice transcription unavailable - model loading failed")
            else:
                raise HTTPException(status_code=500, detail="Voice transcription unavailable - model loading failed")
    return whisper_models[model_size]



# Kinyarwanda-specific model loading (enhanced with trained models)

def get_kinyarwanda_asr_model():

    """

    Get the best available Kinyarwanda ASR model

    Tries trained models first, falls back to standard Whisper

    """

    # Try trained models first

    if model_config is None:

        load_model_config()

    

    available_models = model_config.get("available_models", [])

    if available_models:

        # Try to load the default model

        model = get_kinyarwanda_trained_model()

        if model is not None:

            return model

        

        # Try other available models

        for model_name in available_models:

            model = get_kinyarwanda_trained_model(model_name)

            if model is not None:

                return model

    

    # Fall back to standard Whisper with Kinyarwanda prompting

    logger.warning("No trained Kinyarwanda models available, using standard Whisper")

    return get_whisper_model("base")



# Ensure nltk punkt data is available (for summarizer)

try:

    nltk.data.find('tokenizers/punkt')

    nltk.data.find('tokenizers/punkt_tab')

except LookupError:

    try:

        nltk.download('punkt')

        nltk.download('punkt_tab')

        logger.info("NLTK punkt data downloaded successfully")

    except Exception as e:

        logger.warning(f"Failed to download nltk punkt data: {e}. Summarizer may fail.")



# ---------------------------

# Kinyarwanda vocabulary (loaded from file)

# ---------------------------

_KIN_VOCAB_FILE = "kin_vocab.txt"

_KIN_VOCAB = set()

if os.path.exists(_KIN_VOCAB_FILE):

    with open(_KIN_VOCAB_FILE, "r", encoding="utf-8") as f:

        for line in f:

            word = line.strip().lower()

            if word:

                _KIN_VOCAB.add(word)

    logger.info(f"Loaded {len(_KIN_VOCAB)} Kinyarwanda words from {_KIN_VOCAB_FILE}")

else:

    # Minimal fallback set (original)

    _KIN_VOCAB = set([

        "gura", "erekana", "ohereza", "koherereza", "tanga", "shaka", "nkeneye", "nifuza", "ndashaka",

        "amafaranga", "airtime", "konti", "data", "minuta", "sms", "murakoze", "muraho", "mwaramutse",

        "mwiriwe", "bite", "murabeho", "ndabashimiye", "ubufasha", "irekure", "ntagipimo", "udasobanutse",

        "kuri buri munsi", "buri munsi", "icyumweru", "ukwezi", "heza", "byinshi", "yaba", "nki", "guhitamo",

        "ibiciro", "ubwoko"

    ])

    logger.warning(f"Vocabulary file {_KIN_VOCAB_FILE} not found, using fallback set of {len(_KIN_VOCAB)} words.")



# ---------------------------

# Request models

# ---------------------------

class ChatRequest(BaseModel):

    phone: str = Field(..., description="User phone number")

    message: str = Field(..., description="User message / query")





class PurchaseRequest(BaseModel):

    phone: str = Field(..., description="User phone number")

    qp_id: int = Field(..., description="QuantityPrice ID to purchase")





class ProfileUpdate(BaseModel):

    phone: str = Field(..., description="User phone number")

    name: Optional[str] = None

    bio: Optional[str] = None

    language: Optional[str] = None

    

    @validator('name')

    def validate_name(cls, v):

        if v is not None:

            return v.strip()[:100]  # Limit length

        return v

    

    @validator('bio')

    def validate_bio(cls, v):

        if v is not None:

            return v.strip()[:500]  # Limit length

        return v

    

    @validator('language')

    def validate_language(cls, v):

        if v is not None and v not in ['en', 'kin']:

            raise ValueError('Language must be "en" or "kin"')

        return v





class TransferRequest(BaseModel):

    from_phone: str = Field(..., pattern=r"^\+?\d{7,15}$", description="Sender phone")

    to_phone: str = Field(..., pattern=r"^\+?\d{7,15}$", description="Recipient phone")

    amount: float = Field(..., gt=0, le=50000, description="Amount of airtime to send")



# New models for enhancements

class SummarizeRequest(BaseModel):

    messages: str = Field(..., description="Conversation text to summarize")

    language: Optional[str] = "en"



class RedeemRequest(BaseModel):

    phone: str = Field(..., pattern=r"^\+?\d{7,15}$", description="User phone number")

    points: int = Field(..., gt=0, le=100000, description="Points to redeem")



class ActionRequest(BaseModel):

    phone: str = Field(..., description="User phone number")

    action: str = Field(..., description="Action type: purchase_bundle, recharge, check_balance, etc.")

    action_type: Optional[str] = Field(None, description="Action subtype: data, calling, airtime, info")

    bundle_id: Optional[int] = Field(None, description="Bundle ID for purchase actions")

    bundle_name: Optional[str] = Field(None, description="Bundle name for display")

    price: Optional[float] = Field(None, description="Price for recharge/purchase actions")

    amount: Optional[float] = Field(None, description="Amount for recharge actions")


# Enhanced transcription request model

class TranscribeRequest(BaseModel):

    phone: str = Field(..., description="User phone number")

    language: Optional[str] = "auto"  # auto, en, kin, mixed

    model_size: Optional[str] = "base"  # tiny, base, small, medium, large

    enable_code_mixing: Optional[bool] = True

    confidence_threshold: Optional[float] = 0.5


# Mobile Money Request Models
class MobileMoneyDepositRequest(BaseModel):
    phone: str = Field(..., description="User phone number")
    provider: str = Field(..., description="Mobile money provider (mtn, airtel, tigo)")
    amount: float = Field(..., gt=0, le=1000000, description="Amount to deposit")
    transaction_reference: Optional[str] = Field(None, description="External transaction reference")


class MobileMoneyWithdrawRequest(BaseModel):
    phone: str = Field(..., description="User phone number")
    provider: str = Field(..., description="Mobile money provider (mtn, airtel, tigo)")
    amount: float = Field(..., gt=0, le=500000, description="Amount to withdraw")
    description: Optional[str] = Field(None, description="Withdrawal description")


class MobileMoneyTransferRequest(BaseModel):
    from_phone: str = Field(..., pattern=r"^\+?\d{7,15}$", description="Sender phone")
    to_phone: str = Field(..., pattern=r"^\+?\d{7,15}$", description="Recipient phone")
    provider: str = Field(..., description="Mobile money provider")
    amount: float = Field(..., gt=0, le=200000, description="Amount to transfer")
    description: Optional[str] = Field(None, description="Transfer description")


class MobileMoneyBalanceRequest(BaseModel):
    phone: str = Field(..., description="User phone number")
    provider: Optional[str] = Field(None, description="Specific provider (optional)")


# Electricity Request Models
class AddMeterRequest(BaseModel):
    phone: str = Field(..., description="User phone number")
    meter_number: str = Field(..., description="Electricity meter number")
    provider_code: str = Field(..., description="Electricity provider code (eucl, reg, energo, redi)")
    customer_name: Optional[str] = Field(None, description="Customer name on the meter")


class BuyElectricityRequest(BaseModel):
    phone: str = Field(..., description="User phone number")
    meter_number: str = Field(..., description="Electricity meter number")
    amount: float = Field(..., gt=100, le=500000, description="Amount to pay (minimum 100 RWF)")
    payment_method: str = Field(default="mobile_money", description="Payment method")


class ElectricityMetersRequest(BaseModel):
    phone: str = Field(..., description="User phone number")


class ElectricityProvidersRequest(BaseModel):
    pass


# Payment Methods Request Models
class AddPaymentMethodRequest(BaseModel):
    phone: str = Field(..., description="User phone number")
    method_type: str = Field(..., description="Payment method type (mobile_money, airtime, bank)")
    provider: Optional[str] = Field(None, description="Provider for mobile money")
    account_number: Optional[str] = Field(None, description="Account number")
    is_default: Optional[bool] = Field(False, description="Set as default payment method")


class PaymentMethodsRequest(BaseModel):
    phone: str = Field(..., description="User phone number")


# Authentication Request Models
class LoginRequest(BaseModel):
    phone: str = Field(..., pattern=r"^\+?\d{7,15}$", description="User phone number")
    pin: str = Field(..., pattern=r"^\d{5}$", description="5-digit PIN")
    device_info: Optional[str] = Field(None, description="Device information")


class RegisterRequest(BaseModel):
    name: str = Field(..., description="User name")
    phone: str = Field(..., pattern=r"^\+?\d{7,15}$", description="User phone number")
    pin: str = Field(..., pattern=r"^\d{5}$", description="5-digit PIN")
    confirm_pin: str = Field(..., pattern=r"^\d{5}$", description="Confirm 5-digit PIN")


class ChangePinRequest(BaseModel):
    phone: str = Field(..., pattern=r"^\+?\d{7,15}$", description="User phone number")
    current_pin: str = Field(..., pattern=r"^\d{5}$", description="Current 5-digit PIN")
    new_pin: str = Field(..., pattern=r"^\d{5}$", description="New 5-digit PIN")
    confirm_pin: str = Field(..., pattern=r"^\d{5}$", description="Confirm new PIN")


class LogoutRequest(BaseModel):
    session_token: str = Field(..., description="Session token")


class ValidateSessionRequest(BaseModel):
    session_token: str = Field(..., description="Session token")


# ---------------------------

ALLOWED_TABLES = [

    "users", "airtime_balance", "purchased_bundles",

    "main_category", "sub_category", "period", "quantity_price",

    "conversation_state", "airtime_transfers"

]



ADMIN_PHONES = [p.strip() for p in os.getenv("ADMIN_PHONES", "").split(",") if p.strip()]



REDIS_URL = os.getenv("REDIS_URL")

redis_client = None

if REDIS_URL and redis:

    try:

        redis_client = redis.from_url(REDIS_URL, decode_responses=True)

        logger.info("Connected to Redis at %s", REDIS_URL)

    except Exception:

        logger.exception("Failed to connect to Redis; falling back to DB state store")

        redis_client = None





def get_db_connection():
    try:
        return psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            sslmode="require",
            sslcert=None,
            sslkey=None,
            sslrootcert=None,
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





def initialize_database():

    """Initialize required database tables if they don't exist"""

    try:

        # Create conversation_state table

        execute("""

            CREATE TABLE IF NOT EXISTS conversation_state (

                phone VARCHAR(20) PRIMARY KEY,

                last_type VARCHAR(50),

                options TEXT,

                updated_at TIMESTAMP,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

            )

        """)

        

        # Create index for conversation_state

        execute("""

            CREATE INDEX IF NOT EXISTS idx_conversation_state_updated_at 

            ON conversation_state(updated_at)

        """)

        

        # Create airtime_transfers table

        execute("""

            CREATE TABLE IF NOT EXISTS airtime_transfers (

                id SERIAL PRIMARY KEY,

                from_phone VARCHAR(20) NOT NULL,

                to_phone VARCHAR(20) NOT NULL,

                amount DECIMAL(10,2) NOT NULL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                status VARCHAR(20) DEFAULT 'completed'

            )

        """)

        

        # Create indexes for airtime_transfers

        execute("""

            CREATE INDEX IF NOT EXISTS idx_airtime_transfers_from_phone 

            ON airtime_transfers(from_phone)

        """)

        

        execute("""

            CREATE INDEX IF NOT EXISTS idx_airtime_transfers_created_at 

            ON airtime_transfers(created_at)

        """)

        

        # Add authentication columns to users table
        execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS pin_hash VARCHAR(255)")
        execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true")
        execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login TIMESTAMP")
        execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS failed_login_attempts INTEGER DEFAULT 0")
        execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS account_locked_until TIMESTAMP")

        # Create user_sessions table
        execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id SERIAL PRIMARY KEY,
                phone_number VARCHAR(20) NOT NULL,
                session_token VARCHAR(255) UNIQUE NOT NULL,
                device_info TEXT,
                ip_address INET,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT true,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (phone_number) REFERENCES users(phone_number) ON DELETE CASCADE
            )
        """)

        # Create login_attempts table
        execute("""
            CREATE TABLE IF NOT EXISTS login_attempts (
                id SERIAL PRIMARY KEY,
                phone_number VARCHAR(20) NOT NULL,
                ip_address INET,
                user_agent TEXT,
                success BOOLEAN NOT NULL,
                failure_reason VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (phone_number) REFERENCES users(phone_number) ON DELETE CASCADE
            )
        """)

        # Create indexes for authentication tables
        execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_phone ON user_sessions(phone_number)")
        execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token)")
        execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions(expires_at)")
        execute("CREATE INDEX IF NOT EXISTS idx_login_attempts_phone ON login_attempts(phone_number)")
        execute("CREATE INDEX IF NOT EXISTS idx_login_attempts_created ON login_attempts(created_at)")

        logger.info("Database tables initialized successfully")

    except Exception as e:

        logger.exception("Failed to initialize database tables")

        raise


def hash_pin(pin: str) -> str:
    """Hash a 5-digit PIN using bcrypt"""
    import bcrypt
    return bcrypt.hashpw(pin.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_pin(pin: str, hashed_pin: str) -> bool:
    """Verify a 5-digit PIN against its hash"""
    import bcrypt
    if not hashed_pin:
        return False
    return bcrypt.checkpw(pin.encode('utf-8'), hashed_pin.encode('utf-8'))


def generate_session_token() -> str:
    """Generate a secure session token"""
    import secrets
    return secrets.token_urlsafe(32)


def create_user_session(phone: str, device_info: str = None, ip_address: str = None) -> str:
    """Create a new user session"""
    session_token = generate_session_token()
    expires_at = datetime.utcnow() + timedelta(hours=24)  # 24-hour session
    
    execute("""
        INSERT INTO user_sessions (phone_number, session_token, device_info, ip_address, expires_at)
        VALUES (%s, %s, %s, %s, %s)
    """, (phone, session_token, device_info, ip_address, expires_at))
    
    return session_token


def validate_session(session_token: str) -> Optional[dict]:
    """Validate a session token and return user info"""
    session = fetch_one("""
        SELECT phone_number, device_info, ip_address, created_at, expires_at, last_activity
        FROM user_sessions 
        WHERE session_token = %s AND is_active = true AND expires_at > CURRENT_TIMESTAMP
    """, (session_token,))
    
    if session:
        # Update last activity
        execute("UPDATE user_sessions SET last_activity = CURRENT_TIMESTAMP WHERE session_token = %s", (session_token,))
        return session
    return None


def invalidate_session(session_token: str) -> bool:
    """Invalidate a session token"""
    result = execute("UPDATE user_sessions SET is_active = false WHERE session_token = %s", (session_token,))
    return True


def is_account_locked(phone: str) -> bool:
    """Check if an account is locked"""
    user = fetch_one("""
        SELECT account_locked_until FROM users 
        WHERE phone_number = %s AND account_locked_until > CURRENT_TIMESTAMP
    """, (phone,))
    return user is not None


def record_login_attempt(phone: str, success: bool, ip_address: str = None, user_agent: str = None, failure_reason: str = None):
    """Record a login attempt"""
    execute("""
        INSERT INTO login_attempts (phone_number, ip_address, user_agent, success, failure_reason)
        VALUES (%s, %s, %s, %s, %s)
    """, (phone, ip_address, user_agent, success, failure_reason))
    
    if not success:
        # Increment failed login attempts
        execute("""
            UPDATE users 
            SET failed_login_attempts = failed_login_attempts + 1,
                account_locked_until = CASE 
                    WHEN failed_login_attempts + 1 >= 3 THEN CURRENT_TIMESTAMP + INTERVAL '30 minutes'
                    ELSE NULL
                END
            WHERE phone_number = %s
        """, (phone,))
    else:
        # Reset failed login attempts
        execute("""
            UPDATE users 
            SET failed_login_attempts = 0, 
                account_locked_until = NULL,
                last_login = CURRENT_TIMESTAMP
            WHERE phone_number = %s
        """, (phone,))



def to_float(value):

    if isinstance(value, Decimal):

        return float(value)

    return value





def normalize_phone(phone: Optional[str]) -> Optional[str]:

    if phone is None:

        return phone

    p = phone.strip()

    p = re.sub(r"[ \-\(\)]", "", p)

    return p





def normalize(text: Optional[str]) -> str:

    return re.sub(r"[\-_/]+", " ", (text or "").strip().lower())



# ---------------------------

# Period / bundle expiry helpers

# ---------------------------

PERIOD_MAP = {"daily": "day", "weekly": "week", "monthly": "month", "day": "day", "week": "week", "month": "month"}



def parse_period(text: Optional[str]) -> Optional[str]:

    if not text:

        return None

    t = normalize(text)

    # English / general patterns

    for k, v in PERIOD_MAP.items():

        if k in t:

            return v

    # Kinyarwanda period phrases

    kin_periods = {

        "buri munsi": "day",

        "kuri buri munsi": "day",

        "icyumweru": "week",

        "ukwezi": "month",

    }

    for phrase, period in kin_periods.items():

        if phrase in t:

            return period

    # numeric patterns (e.g., '7 days', '14-day')

    m = re.search(r"(\d+)\s*-\s*day|\b(\d+)\s*days?\b", text or "")

    if m:

        return "day"

    m = re.search(r"(\d+)\s*-\s*week|\b(\d+)\s*weeks?\b", text or "")

    if m:

        return "week"

    m = re.search(r"(\d+)\s*-\s*month|\b(\d+)\s*months?\b", text or "")

    if m:

        return "month"

    return None





def period_label_to_timedelta(label: Optional[str]) -> Optional[timedelta]:

    if not label:

        return None

    L = (label or "").lower()

    # simple mappings

    if "day" in L or "daily" in L:

        return timedelta(days=1)

    if "week" in L or "weekly" in L:

        return timedelta(weeks=1)

    if "month" in L or "monthly" in L:

        return timedelta(days=30)

    # parse "7 days", "14-day", "2 weeks", "3 months"

    m = re.search(r"(\d+)\s*day", L)

    if m:

        return timedelta(days=int(m.group(1)))

    m = re.search(r"(\d+)\s*week", L)

    if m:

        return timedelta(weeks=int(m.group(1)))

    m = re.search(r"(\d+)\s*month", L)

    if m:

        return timedelta(days=30 * int(m.group(1)))

    return None





def expire_bundles_for_phone(phone: str) -> None:

    """

    Mark purchased_bundles as expired (remaining = 0) when purchase_date + period <= now.

    This function is non-destructive and idempotent; used lazily when returning balances/purchases.

    Optimized to use bulk operations instead of processing one by one.

    """

    try:

        now = datetime.utcnow()

        

        # Get all bundles that need to be expired in one query

        # Using a simpler approach without relying on validity_hours column

        rows_to_expire = fetch_all(

            """

            SELECT pb.id, pb.purchase_date, p.label AS period_label,

                   CASE 

                     WHEN p.label = '1 day' THEN pb.purchase_date + INTERVAL '1 day'

                     WHEN p.label = '7 days' THEN pb.purchase_date + INTERVAL '7 days'

                     WHEN p.label = '30 days' THEN pb.purchase_date + INTERVAL '30 days'

                     ELSE pb.purchase_date + INTERVAL '30 days'

                   END as expiry_date

            FROM purchased_bundles pb

            JOIN quantity_price qp ON pb.quantity_price_id = qp.id

            JOIN period p ON qp.period_id = p.id

            WHERE pb.phone_number = %s 

              AND COALESCE(pb.remaining,0) > 0

              AND CASE 

                     WHEN p.label = '1 day' THEN pb.purchase_date + INTERVAL '1 day'

                     WHEN p.label = '7 days' THEN pb.purchase_date + INTERVAL '7 days'

                     WHEN p.label = '30 days' THEN pb.purchase_date + INTERVAL '30 days'

                     ELSE pb.purchase_date + INTERVAL '30 days'

                   END <= %s

            """,

            (phone, now),

        )

        

        if not rows_to_expire:

            return

            

        # Bulk update all expired bundles at once

        expired_ids = [str(row["id"]) for row in rows_to_expire]

        if expired_ids:

            execute(

                f"UPDATE purchased_bundles SET remaining = 0 WHERE id IN ({','.join(expired_ids)})"

            )

            logger.info(f"Expired {len(expired_ids)} purchased_bundles for phone={phone}")

            

    except Exception as e:

        logger.exception(f"expire_bundles_for_phone failed for {phone}: {e}")

        # Fallback to original method if bulk operation fails

        try:

            rows = fetch_all(

                """

                SELECT pb.id, pb.purchase_date, p.label AS period_label

                FROM purchased_bundles pb

                JOIN quantity_price qp ON pb.quantity_price_id = qp.id

                JOIN period p ON qp.period_id = p.id

                WHERE pb.phone_number = %s AND COALESCE(pb.remaining,0) > 0

                """,

                (phone,),

            )

            if not rows:

                return

                

            to_expire = []

            for r in rows:

                pd = r.get("purchase_date")

                label = r.get("period_label")

                if not pd or not label:

                    continue

                if not isinstance(pd, datetime):

                    try:

                        pd = datetime.fromisoformat(str(pd))

                    except Exception:

                        continue

                delta = period_label_to_timedelta(label)

                if not delta:

                    continue

                if pd + delta <= now:

                    to_expire.append(r["id"])

                    

            for bid in to_expire:

                try:

                    execute("UPDATE purchased_bundles SET remaining = 0 WHERE id = %s", (bid,))

                    logger.info("Expired purchased_bundles id=%s for phone=%s", bid, phone)

                except Exception:

                    logger.exception("Failed to mark purchased_bundles id %s expired", bid)

        except Exception:

            logger.exception("Fallback expiry also failed for %s", phone)





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


def fetch_user(phone: str) -> Optional[dict]:
    try:
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
    except Exception:
        logger.exception("Failed to fetch user for %s", phone)


def _digits_only(phone: Optional[str]) -> str:

    if not phone:

        return ""

    return re.sub(r"\D", "", phone)





def find_user_by_phone_variants(phone: Optional[str]) -> Optional[dict]:

    """

    Try common phone variants and return the first matched user dict or None.

    Variants tried:

      - exact phone

      - without leading '+'

      - with leading '+'

      - replace leading 0 with DEFAULT_COUNTRY_CODE (if set)

      - digits-only equality search (regexp_replace)

    """

    if not phone:

        return None

    tried = set()

    candidates = []

    p = phone.strip()

    candidates.append(p)

    if p.startswith("+"):

        candidates.append(p.lstrip("+"))

    else:

        candidates.append("+" + p)

    default_cc = os.getenv("DEFAULT_COUNTRY_CODE")

    if p.startswith("0") and default_cc:

        candidates.append("+" + default_cc + p.lstrip("0"))

        candidates.append(default_cc + p.lstrip("0"))

    for c in candidates:

        if not c or c in tried:

            continue

        tried.add(c)

        user = None

        try:

            user = fetch_user(c)

        except Exception:

            logger.debug("fetch_user raised for candidate %s", c)

            user = None

        if user:

            logger.info("Found user by phone variant: %s -> candidate %s", phone, c)

            return user

    digits = _digits_only(phone)

    if digits:

        try:

            row = fetch_one(

                "SELECT u.*, COALESCE(a.balance,0) AS airtime FROM users u LEFT JOIN airtime_balance a ON u.phone_number = a.phone_number WHERE regexp_replace(u.phone_number, '\\\\D', '', 'g') = %s",

                (digits,),

            )

            if row:

                row["airtime"] = float(row.get("airtime", 0))

                return row

        except Exception:

            logger.debug("Digits-only fallback failed for %s", phone)

    return None

def fetch_airtime(phone: str) -> float:

    row = fetch_one("SELECT COALESCE(balance,0) AS balance FROM airtime_balance WHERE phone_number=%s", (phone,))

    return float(row["balance"]) if row else 0.0





def update_airtime(phone: str, new_balance: float):

    try:

        execute("UPDATE airtime_balance SET balance=%s WHERE phone_number=%s", (new_balance, phone))

    except Exception:

        try:

            execute("INSERT INTO airtime_balance (phone_number, balance) VALUES (%s, %s)", (phone, new_balance))

        except Exception:

            logger.exception("Failed to update/insert airtime balance for %s", phone)





def fetch_bundle_balances(phone: str):

    # expire bundles first (lazy enforcement)

    try:

        expire_bundles_for_phone(phone)

    except Exception:

        logger.debug("Failed to expire bundles before fetching balances for %s", phone)

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

# Enhanced ASR and Code-Mixing Functions

# ---------------------------



def detect_code_mixing(text: str) -> Dict[str, Any]:

    """

    Detect code-mixing between Kinyarwanda and English in transcribed text.

    More strict analysis to prevent false positives from other languages.

    Returns statistics about language distribution.

    """

    if not text:

        return {"has_mixing": False, "kin_ratio": 0, "en_ratio": 0, "mixed_words": []}

    

    words = re.findall(r"\b\w+\b", text.lower())

    total_words = len(words)

    

    if total_words == 0:

        return {"has_mixing": False, "kin_ratio": 0, "en_ratio": 0, "mixed_words": []}

    

    # Count Kinyarwanda words (using our vocabulary)

    kin_words = [w for w in words if w in _KIN_VOCAB]

    

    # Count English words (only common English words that are definitely English)

    # More restrictive to avoid counting other languages

    common_en_words = {

        # Most common English words

        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',

        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',

        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',

        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',

        'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',

        'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',

        'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',

        'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',

        'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',

        'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',

        'give', 'day', 'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had',

        'were', 'said', 'did', 'having', 'may', 'am'

    }

    

    # Only count as English if NOT in Kinyarwanda vocabulary

    en_words = [w for w in words if w in common_en_words and w not in _KIN_VOCAB]

    

    kin_ratio = len(kin_words) / total_words

    en_ratio = len(en_words) / total_words

    

    # Detect mixed words (words that appear in both vocabularies)

    mixed_words = list(set(kin_words) & set(common_en_words))

    

    # More strict mixing detection - need significant presence of both

    # At least 15% of each language AND at least 2 words from each

    kin_threshold = len(kin_words) >= 2 and kin_ratio >= 0.15

    en_threshold = len(en_words) >= 2 and en_ratio >= 0.15

    

    has_mixing = kin_threshold and en_threshold

    

    return {

        "has_mixing": has_mixing,

        "kin_ratio": kin_ratio,

        "en_ratio": en_ratio,

        "kin_words": kin_words,

        "en_words": en_words,

        "mixed_words": mixed_words,

        "total_words": total_words

    }



def transcribe_with_code_mixing(audio_path: str, language: str = "auto", model_size: str = "base") -> Dict[str, Any]:

    """

    Transcribe audio with enhanced Kinyarwanda support and code-mixing detection.

    """

    results = []

    

    if language == "auto" or language == "mixed":

        # Try multiple approaches for best accuracy

        

        # 1. Kinyarwanda-specific model if available

        try:

            kin_model = get_kinyarwanda_asr_model()

            # Enhanced Kinyarwanda prompt with more context

            kin_prompt = (

                "murakoze bite mbega yego oya ndabizi mbega "

                "data bundles airtime konti phone money "

                "show buy get help please thank you"

            )

            kin_result = kin_model.transcribe(

                audio_path,

                language=None,  # Auto-detect but guided

                fp16=False,

                task="transcribe",

                initial_prompt=kin_prompt,

                temperature=0.0,  # Lower temperature for more deterministic output

                best_of=3,  # Try multiple times and pick best

                beam_size=5  # Wider search for better accuracy

            )

            results.append({

                "model": "kinyarwanda_specific",

                "text": kin_result["text"].strip(),

                "language": "kin",

                "confidence": 0.85  # Higher confidence for specific model

            })

        except Exception as e:

            logger.warning(f"Kinyarwanda model failed: {e}")

        

        # 2. Multilingual Whisper with auto-detection (restricted to en/kin)

        try:

            multi_model = get_whisper_model(model_size)

            # Enhanced prompt for Kinyarwanda/English code-mixing

            mixed_prompt = (

                "murakoze bite mbega hello thank you data bundles "

                "airtime phone money show buy get help please "

                "yego oya ndabizi mbega kuri konti"

            )

            multi_result = multi_model.transcribe(

                audio_path,

                language=None,  # Auto-detect

                fp16=False,

                task="transcribe",

                initial_prompt=mixed_prompt,

                temperature=0.1,  # Slightly higher for mixed content

                best_of=5,  # More attempts for mixed content

                beam_size=7,  # Wider search

                language_detection_threshold=0.1  # Lower threshold for detection

            )

            detected_lang = multi_result.get("language", "unknown")

            

            # Validate and filter language

            validated_lang = validate_detected_language(detected_lang)

            if validated_lang:

                results.append({

                    "model": "multilingual_whisper",

                    "text": multi_result["text"].strip(),

                    "language": validated_lang,

                    "confidence": 0.75

                })

            else:

                logger.info(f"Detected unsupported language '{detected_lang}', filtering out")

        except Exception as e:

            logger.warning(f"Multilingual model failed: {e}")

        

        # 3. English transcription (for comparison)

        try:

            en_model = get_whisper_model(model_size)

            # English prompt with telecom context

            en_prompt = (

                "hello thank you please help me show buy get "

                "data bundles airtime phone money account "

                "balance transfer send receive"

            )

            en_result = en_model.transcribe(

                audio_path,

                language="en",

                fp16=False,

                task="transcribe",

                initial_prompt=en_prompt,

                temperature=0.0,  # Deterministic for English

                best_of=3,

                beam_size=5

            )

            results.append({

                "model": "english_whisper",

                "text": en_result["text"].strip(),

                "language": "en",

                "confidence": 0.70

            })

        except Exception as e:

            logger.warning(f"English model failed: {e}")

    

    else:

        # Specific language transcription

        try:

            model = get_whisper_model(model_size)

            # Whisper doesn't support 'rw' directly, use auto-detection with prompt

            whisper_lang = None if language == "kin" else language

            initial_prompt = "murakoze bite mbega" if language == "kin" else None

            

            result = model.transcribe(

                audio_path,

                language=whisper_lang,

                fp16=False,

                task="transcribe",

                initial_prompt=initial_prompt

            )

            results.append({

                "model": f"whisper_{model_size}",

                "text": result["text"].strip(),

                "language": language,

                "confidence": 0.80

            })

        except Exception as e:

            logger.error(f"Transcription failed: {e}")

            raise

    

    # Select best result based on confidence and code-mixing analysis

    if not results:

        raise Exception("All transcription attempts failed")

    

    best_result = max(results, key=lambda x: x["confidence"])

    

    # Analyze code-mixing in the best result

    mixing_analysis = detect_code_mixing(best_result["text"])

    

    # If code-mixing detected and we have multiple results, try to improve

    if mixing_analysis["has_mixing"] and len(results) > 1:

        # Combine results from different models for better accuracy

        combined_text = best_result["text"]

        

        # Look for words that might be better transcribed by other models

        for result in results:

            if result["text"] != best_result["text"]:

                # Simple heuristic: if another result has more Kinyarwanda words, prefer it

                other_mixing = detect_code_mixing(result["text"])

                if other_mixing["kin_ratio"] > mixing_analysis["kin_ratio"]:

                    combined_text = result["text"]

                    mixing_analysis = other_mixing

                    best_result = result

                    break

    

    return {

        "text": best_result["text"],

        "detected_language": best_result["language"],

        "model_used": best_result["model"],

        "confidence": best_result["confidence"],

        "code_mixing": mixing_analysis,

        "all_results": results if len(results) > 1 else None

    }



def validate_detected_language(detected_lang: str) -> str:
    """
    Validate and normalize detected language to only allow Kinyarwanda and English.
    Strict filtering - no other languages supported.
    Returns 'kin' for Kinyarwanda, 'en' for English, or None for unsupported.
    """
    if not detected_lang:
        return None
    
    # Normalize to lowercase and remove variants
    detected_lang = detected_lang.lower().strip()
    
    # Strict English variants only
    if detected_lang in ["en", "english"] or detected_lang.startswith("en"):
        return "en"
    
    # Strict Kinyarwanda variants only  
    if detected_lang in ["rw", "kin", "kinyarwanda"]:
        return "kin"
    
    # Log and filter ALL other languages
    logger.info(f"Filtering out unsupported language: {detected_lang}")
    return None



def clean_non_target_languages(text: str, detected_lang: str) -> str:

    if not text:

        return text

    

    # Common words from other languages that might appear

    other_lang_words = {

        # French common words

        'bonjour', 'merci', 'aujourd', 'oui', 'non', 's il', 'vous', 'pour',

        'avec', 'dans', 'sur', 'par', 'une', 'le', 'la', 'les',

        'tout', 'plus', 'bien', 'très', 'comme', 'entre', 'sans',

        

        # German common words  

        'hallo', 'danke', 'heute', 'ja', 'nein', 'ich', 'sie', 'nicht',

        'mit', 'auf', 'für', 'der', 'die', 'das', 'ein', 'eine',

        

        # Spanish common words

        'hola', 'gracias', 'hoy', 'sí', 'no', 'por favor', 'con', 'para',

        'una', 'el', 'la', 'los', 'muy', 'bien', 'grande',

        

        # Chinese/Japanese common words

        'nǐ hǎo', 'xièxie', 'arigatou', 'sumimasen', 'hai', 'iie',

        'konnichiwa', 'sayonara', 'ohayou', 'dōmo', 'watashi',

        

        # Arabic common words

        'marhaban', 'shukran', 'ahlan', 'naam', 'la', 'fi', 'min',

        'sabah', 'maa', 'salaam', 'yalla', 'inshallah'

    }

    

    words = text.split()

    cleaned_words = []

    

    for word in words:

        word_lower = word.lower().strip('.,!?')

        # Keep word if it's not clearly from other languages

        if word_lower not in other_lang_words:

            cleaned_words.append(word)

        else:

            logger.debug(f"Filtered out non-target word: '{word}' (detected: {detected_lang})")

    

    return ' '.join(cleaned_words)



def detect_voice_intent(text: str, user_context: dict = None) -> dict:
    """Detect user intent from voice transcription"""
    text = text.lower().strip()
    
    # Intent patterns
    intents = {
        "balance": {
            "keywords": ["balance", "money", "how much", "account", "amafaranga", "balance yange", "mfite"],
            "confidence": 0.0
        },
        "bundles": {
            "keywords": ["bundle", "bundles", "data", "internet", "calling", "amabundle", "yamunika"],
            "confidence": 0.0
        },
        "recharge": {
            "keywords": ["top up", "recharge", "add money", "tanga", "kongera", "airtime"],
            "confidence": 0.0
        },
        "send_money": {
            "keywords": ["send", "transfer", "tunga", "ohereza", "kohereza"],
            "confidence": 0.0
        },
        "help": {
            "keywords": ["help", "support", "what can", "ufasha", "ubufasha", "nsabira"],
            "confidence": 0.0
        }
    }
    
    # Calculate confidence for each intent
    best_intent = "unknown"
    best_confidence = 0.0
    
    for intent_name, intent_data in intents.items():
        matches = 0
        total_keywords = len(intent_data["keywords"])
        
        for keyword in intent_data["keywords"]:
            if keyword in text:
                matches += 1
        
        confidence = matches / total_keywords if total_keywords > 0 else 0
        intent_data["confidence"] = confidence
        
        if confidence > best_confidence:
            best_confidence = confidence
            best_intent = intent_name
    
    return {
        "intent": best_intent,
        "confidence": best_confidence,
        "text": text,
        "all_intents": intents
    }

def post_process_transcription(text: str, code_mixing_info: Dict[str, Any]) -> str:

    """

    Post-process transcribed text to improve Kinyarwanda accuracy and handle code-mixing.

    """

    if not text:
        return "No speech detected. Please try again."
    
    # Check if text contains only silence/filler sounds
    text = text.strip()
    if not text or len(text) < 2:
        return "No speech detected. Please try again."
    
    # Common transcription corrections for Kinyarwanda

    corrections = {

        # Common misrecognitions

        "muri ko": "murakoze",

        "mira koze": "murakoze",

        "mira": "murakoze",

        "bite": "bite",  # Already correct

        "mwaramutse": "mwaramutse",

        "muraho": "muraho",

        "yego": "yego",

        "oya": "oya",

        "ndabwi": "ndabizi",

        "sinzi": "ndabizi",

        "mbega": "mbega",

        "niba": "niba",

        "kuri": "kuri",

        "kuko": "kuko",

        "kubera": "kubera",

        "twe": "twe",

        "mwe": "mwe",

        "buri": "buri",

        "byinshi": "byinshi",

        "bike": "bicyo",

        "bikora": "bikora",

        "bikomeye": "bikomeye",

        

        # Telecom-specific corrections

        "konti": "konti",

        "account": "konti",

        "balance": "balance",

        "data": "data",

        "bundles": "bundles",

        "airtime": "airtime",

        "money": "amafaranga",

        "phone": "telefoni",

        "show": "erekana",

        "buy": "gura",

        "get": "kubona",

        "help": "ufasha",

        "please": "nyamuneka",

        "thank you": "murakoze",

        "thank": "murakoze",

        

        # Code-mixing corrections

        "show me": "erekana",

        "buy data": "gura data",

        "get balance": "kubona balance",

        "send money": "ohereza amafaranga",

        "airtime balance": "balance y'amafaranga",

        "check my": "check my",
        "how much": "how much",
        "what is": "what is",
        "where can": "where can",
        "I need to": "I need to",
        "top up": "top up",
        "credit": "credit",
        "amount": "amount",
        "available": "available",
        "balance": "balance",
        "account": "account",
        "phone": "phone",
        "money": "money",
        "data": "data",
        "bundles": "bundles",
        "airtime": "airtime",
        "transfer": "transfer",
        "recharge": "recharge",
        "payment": "payment",
        "purchase": "purchase",
        "price": "price",
        "cost": "cost",
        "costs": "costs",

        # Common English words that might be misrecognized

        "hello": "hello",

        "please": "please",

        "thank you": "thank you",

        "help me": "help me",

        "show me": "show me",

        "get": "get",

        "buy some": "buy some",

        "send to": "send to",

        "receive from": "receive from"

    }

    

    # Apply corrections while preserving case

    corrected_text = text

    for wrong, correct in corrections.items():

        # Case-insensitive replacement

        pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)

        def replace_func(match):

            word = match.group()

            # Preserve original case pattern

            if word.isupper():

                return correct.upper()

            elif word[0].isupper():

                return correct.capitalize()

            else:

                return correct.lower()

        corrected_text = pattern.sub(replace_func, corrected_text)

    

    # Add spacing corrections for common patterns

    corrected_text = re.sub(r"\b(ng')\s+", r"\1", corrected_text)  # Fix ng' spacing

    corrected_text = re.sub(r"\b(n')\s+", r"\1", corrected_text)  # Fix n' spacing

    

    return corrected_text.strip()

# ---------------------------

# FRIENDLY TRANSLATIONS with emojis and warmth

# ---------------------------

TRANSLATIONS = {

    "en": {

        "greeting_intro": "Hello {name}! 👋\nI'm here to help you with:\n• Checking airtime balance 💰\n• Viewing your bundle balances 📊\n• Buying data, voice, or SMS bundles 📱\n\nJust say something like *show bundles* and we'll get started!",

        "show_bundles": "Let's see what bundles we have, {name}!",

        "my_balance": "Your current airtime balance is:",

        "help": "I can help with bundles, balance, transfers – what do you need?",

        "no_bundles": "Oops {name}, it looks like there are no bundle categories available right now. Would you like to check your balance instead?",

        "choose_main": "{name}, please choose a main category (reply with number or name):",

        "choose_sub": "Great! Under '{main}' choose a subcategory:",

        "choose_period": "Awesome, for '{sub}' which period would you like?",

        "bundles_matching": "{name}, I found some bundles matching your request:",

        "reply_options_prompt": "Just reply with the number to see details or buy – easy!",

        "details_for_option": "Here are the details for option {idx}:",

        "purchase_prompt": "Ready to buy? Reply 'purchase {idx}' to complete.",

        "purchase_success": "🎉 Success! {quantity} units activated. Your new airtime balance is {new_balance}. Enjoy!",

        "insufficient_funds": "😕 Sorry {name}, you don't have enough airtime ({balance}). Would you like to top up now? 💳",

        "restart_msg": "{name}, let's start fresh! Ask me to show bundles when you're ready.",

        "back_none": "There's nothing to go back to, {name}. Just say 'show bundles' to begin.",

        "default_fallback": "😊 Hi {name}, I'm here to help! You can ask me to show bundles, check balance, or send airtime. What would you like?",

        "goodbye": "Goodbye {name} 👋 Come back anytime! If you need anything else, just say 'show bundles' or 'help'.",

        "thanks": "You're very welcome, {name}! Glad I could help. 😊",

        "who_are_you": "I'm BazaAI – your friendly telecom assistant! I can help with airtime, bundles, account info, and more.",

        "how_are_you": "I'm doing great, thanks for asking! 🤖 Ready to assist you with anything telecom. How can I help you today?",

        "support_prompt": "I can help with airtime, bundles, and account info. Try saying 'show bundles' to see what's available.",

        "bundle_balances_header": "{name}, here are your active bundles and remaining balances:",

        "bundle_item_line": "{idx}. {main} / {sub} — remaining: {remaining} — purchased: {date}",

        "no_bundles_found": "{name}, you have no active bundles right now. Want to buy one?",

        "send_airtime_success": "✅ Sent {amount} to {to_phone}. Your new balance: {new_balance}",

        "send_airtime_failed": "❌ Airtime transfer failed. Please try again later.",

        "send_airtime_insufficient": "😕 Sorry {name}, you don't have enough airtime ({balance}) to send {amount}. Would you like to recharge?",

        "send_airtime_unknown": "Hmm, the recipient {to_phone} isn't in our system. Please check the number and try again.",

        "bundle_display": "{main} / {sub} — {quantity} units — {period} — Price: {price}",

        "found_bundles": "Found {count} bundles. Here are the top {limit} by price:",

        "no_exact_match": "Sorry {name}, I couldn't find any {main} {sub} {period} bundles. But here are other {sub} {period} bundles you might like:",

        "here_are_bundles": "{name}, here are {sub} {period} bundles for you:",

        "you_asked_about": "{name}, you asked about {sub}. Which period interests you?",

        "here_are_period_bundles": "{name}, here are {period} bundles:",

        "bundles_with_quantity": "{name}, bundles with at least {quantity} {unit}:",

        "show_all_option": "Show all",

        "feedback_question": "By the way, are you happy with this conversation so far? (yes/no)",

    },

    "kin": {

        "greeting_intro": "Muraho {name}! 👋\nNdagufasha:\n• Kureba amafaranga ya airtime 💰\n• Kureba bundles ziri ku konti 📊\n• Kugura data, voice, cyangwa SMS bundles 📱\n\nVuga nka 'erekana bundles' kugirango dutangire!",

        "show_bundles": "Reka turebe bundles, {name}!",

        "my_balance": "Airtime usigaranye:",

        "help": "Ndagufasha kuri bundles, balance, kohereza – ukeneye iki?",

        "no_bundles": "Urababari {name}, nta byiciro bya bundles biboneka ubu. Wifuza kureba balance yawe?",

        "choose_main": "{name}, hitamo icyiciro nyamukuru (andika nomero cyangwa izina):",

        "choose_sub": "Ni byiza! Munsi ya '{main}' hitamo subcategory:",

        "choose_period": "Nibyiza, kuri '{sub}' ni ikihe gihe ushaka?",

        "bundles_matching": "{name}, nabonye bundles zihuye n'icyo wasabye:",

        "reply_options_prompt": "Subiza numero kugirango ubone ibisobanuro cyangwa ugure – byoroshye!",

        "details_for_option": "Ibi ni ibisobanuro kuri option {idx}:",

        "purchase_prompt": "Uteguriye kugura? Subiza 'purchase {idx}' kugirango urangize.",

        "purchase_success": "🎉 Byaciyemo! {quantity} units byongewe. Airtime usigaranye: {new_balance}. Murakoze!",

        "insufficient_funds": "😕 Urababari {name}, nta mafaranga ahagije kuri airtime ({balance}). Wifuza kwiyongerera ubu? 💳",

        "restart_msg": "{name}, reka dutangire bundi bushya! Vuga 'erekana bundles' iyo ugiye gutangira.",

        "back_none": "Nta ho usubira inyuma, {name}. Vuga 'erekana bundles' kugirango utangire.",

        "default_fallback": "😊 Muraho {name}, ndi hano nkugufasha! Urashobora kuvuga ngo 'erekana bundles', kureba balance, cyangwa kohereza airtime. Wifuza iki?",

        "goodbye": "Murabeho {name} 👋 Garuka igihe icyo aricyo cyose! Niba ukeneye ikindi, vuga 'erekana bundles' cyangwa 'ubufasha'.",

        "thanks": "Murakoze cyane, {name}! Nishimiye kugufasha. 😊",

        "who_are_you": "Ndi BazaAI – umufasha wawe wa telecom wigicuti! Ndagufasha kuri airtime, bundles, amakuru ya konti, n'ibindi.",

        "how_are_you": "Mfite neza, murakoze kubaza! 🤖 Niteguye gufasha. Ndagufasha ngo dukore iki uyu munsi?",

        "support_prompt": "Ndagufasha kuri airtime, bundles, na konti. Gerageza kuvuga 'erekana bundles' kugirango urebe ibihari.",

        "bundle_balances_header": "{name}, izi ni bundles zawe hamwe n'ibisigaye:",

        "bundle_item_line": "{idx}. {main} / {sub} — ibisigaye: {remaining} — waguriye: {date}",

        "no_bundles_found": "{name}, nta bundles ziriho ubu. Wifuza kugura zimwe?",

        "send_airtime_success": "✅ Wohereje {amount} kuri {to_phone}. Airtime usigaranye: {new_balance}",

        "send_airtime_failed": "❌ Koherezwa airtime byanze. Ongera ugerageze nyuma.",

        "send_airtime_insufficient": "😕 Urababari {name}, nta mafaranga ahagije ({balance}) yo kohereza {amount}. Wifuza kwiyongerera?",

        "send_airtime_unknown": "Uwakira {to_phone} ntabwo tubonye mu buryo bwacu. Nyamuneka ugerageze nomero ikora.",

        "bundle_display": "{main} / {sub} — {quantity} units — {period} — Igiciro: {price}",

        "found_bundles": "Habonetse {count} za bundle. Dore iza mbere {limit} kuri giciro:",

        "no_exact_match": "Urababari {name}, nta bundle za {main} {sub} {period} zabonetse. Ariko dore izindi bundle za {sub} {period} ushobora gukunda:",

        "here_are_bundles": "{name}, dore bundle za {sub} {period}:",

        "you_asked_about": "{name}, wabajije kuri {sub}. Hitamo igihe:",

        "here_are_period_bundles": "{name}, dore bundle z'igihe cya {period}:",

        "bundles_with_quantity": "{name}, bundle zifite byibuze {quantity} {unit}:",

        "show_all_option": "Reba zose",

        "feedback_question": "Mu buryo, urishimye n'ikiganiro kugeza ubu? (yego/oya)",

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





# ---------------------------

# Smalltalk & NLP helpers

# ---------------------------

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



# Variants for natural responses (these are now only used for English; for Kinyarwanda we rely on tr())

GREETING_VARIANTS = [

    "Hi {name}! What can I do for you today?",

    "Hello {name}! Ready to help with bundles, balance, or transfers.",

    "Hey {name}! How can I assist?",

    "Good to see you {name}! Need help with something?",

]

BALANCE_VARIANTS = [

    "{name}, your current airtime balance is {balance}.",

    "You have {balance} airtime, {name}.",

    "{name}, your balance is {balance}.",

]

FALLBACK_VARIANTS = [

    "Sorry {name}, I didn't quite catch that. I can help with:\n• Buying bundles (e.g., 'show data bundles')\n• Checking balance ('my balance')\n• Sending airtime ('send 100 to 078...')\n\nWhat would you like?",

    "Hmm, I'm not sure I understood. You can ask me to show bundles, check your balance, or send airtime. Try something like 'show daily bundles'!",

    "I didn't get that, {name}. Here are some things I can do:\n• 'show bundles'\n• 'my balance'\n• 'send 100 to 078...'",

]



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

        "send",

        "transfer",

    ]

)



def rank_bundles(bundles: List[Dict], limit: int = 5) -> List[Dict]:

    """Simple ranking: sort by price ascending (cheapest first)."""

    return sorted(bundles, key=lambda b: b['price'])[:limit]





def is_send_airtime_intent(text: Optional[str]) -> bool:

    if not text:

        return False

    t = text.lower()



    # explicit keywords

    send_keywords = [

        "send airtime", "transfer airtime", "ohereza airtime", "ohereza amafaranga",

        "koherereza airtime", "tanga airtime", "send money", "transfer", "ohereza"

    ]

    if any(k in t for k in send_keywords):

        return True



    # Patterns: "send 100 to +2507..." or "send 100 to 0787..." (English)

    if re.search(r"\bsend\s+\d+(?:\.\d+)?\s+to\s+(\+?\d{7,15}|\b0\d{6,12}\b)", t):

        return True



    # Kinyarwanda patterns: "ohereza 100 kuri 0787..." or "kohereza 100 kuri +2507..."

    if re.search(r"\b(ohereza|kohereza|tanga)\s+\d+(?:\.\d+)?\s+(kuri|ku)\s+(\+?\d{7,15}|\b0\d{6,12}\b)", t):

        return True



    # Some users may say "send 100 0787..." (amount followed by phone) — detect amount+phone nearby

    if re.search(r"(\+?\d{7,15}|\b0\d{6,12}\b).*?\b\d+(?:\.\d+)?\b", t) or re.search(r"\b\d+(?:\.\d+)?\b.*?(\+?\d{7,15}|\b0\d{6,12}\b)", t):

        # require 'send' or kinyarwanda verbs nearby to reduce false positives

        if re.search(r"\b(send|ohereza|kohereza|tanga|transfer|send)\b", t):

            return True



    return False





def detect_language_simple(text: str, user_lang_hint: Optional[str] = None) -> str:
    if not text or text.strip() == "":
        return user_lang_hint if user_lang_hint in ("kin", "en") else "en"

    t = text.strip().lower()
    tokens = re.findall(r"\w+", t)
    kin_score = sum(1 for tok in tokens if tok in _KIN_VOCAB)
    en_score = sum(1 for tok in tokens if tok in _EN_VOCAB)

    logger.debug(f"kin_score={kin_score}, en_score={en_score} for '{text}'")

    # If any Kinyarwanda word found, prefer Kinyarwanda
    if kin_score > 0:
        return "kin"
    if en_score > 0:
        return "en"

    # Fallback to user hint or English
    return user_lang_hint if user_lang_hint in ("kin", "en") else "en"





def parse_quantity(text: Optional[str]):

    m = re.search(r"(\d+(?:\.\d+)?)\s*(gb|mb|rwf|frw|rwf)?", (text or "").lower())

    if m:

        try:

            return float(m.group(1)), (m.group(2) or "unit")

        except Exception:

            return None, None

    m2 = re.search(r"(\d+)", (text or ""))

    if m2:

        try:

            return float(m2.group(1)), "unit"

        except Exception:

            return None, None

    return None, None





def extract_category(message: Optional[str]) -> Tuple[Optional[str], Optional[str]]:

    if not message:

        return None, None



    msg_norm = normalize(message)

    msg_lower = message.lower()



    # Define synonyms for main categories (you can expand these as needed)

    main_synonyms = {

        "internet": ["data", "internet", "interineti", "intaneti", "web"],

        "voice": ["voice", "voix", "ijwi", "minute", "iminota", "appel"],

        "sms": ["sms", "text", "ubutumwa"],

        # add other mains from your DB

    }



    # Define synonyms for sub-categories (e.g., unlimited, daily, etc.)

    sub_synonyms = {

        "unlimited": ["irekure", "ntagipimo", "udasobanutse", "all", "illimité"],

        "daily": ["daily", "kuri buri munsi", "buri munsi", "quotidien"],

        "weekly": ["weekly", "icyumweru", "hebdomadaire"],

        "monthly": ["monthly", "ukwezi", "mensuel"],

        # add others as needed

    }



    # Fetch actual categories from DB

    try:

        mains = list_main_categories() or []

        subs = list_subcategories() or []

    except Exception:

        mains, subs = [], []



    # First try exact matches against DB names

    found_main = None

    for m in mains:

        if m and (normalize(m) in msg_norm or m.lower() in msg_lower):

            found_main = m

            break



    # If not found, try synonyms

    if not found_main:

        for syn_key, syn_list in main_synonyms.items():

            if any(s in msg_lower for s in syn_list):

                # Find actual main category that matches syn_key

                for m in mains:

                    if normalize(m) == syn_key or m.lower() == syn_key:

                        found_main = m

                        break

                if found_main:

                    break



    # Similar for sub-category

    found_sub = None

    for s in subs:

        if s and (normalize(s) in msg_norm or s.lower() in msg_lower):

            found_sub = s

            break



    if not found_sub:

        for syn_key, syn_list in sub_synonyms.items():

            if any(s in msg_lower for s in syn_list):

                for s in subs:

                    if normalize(s) == syn_key or s.lower() == syn_key:

                        found_sub = s

                        break

                if found_sub:

                    break



    return found_main, found_sub





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





def is_bundle_intent(text: Optional[str]) -> bool:

    if not text:

        return False

    t = text.lower()

    english_keywords = [

        "buy", "purchase", "bundle", "subscribe", "get", "need", "want",

        "which", "show", "packages", "plans", "bundles"

    ]

    kinyarwanda_keywords = [

        "gura", "erekana", "konti", "amafaranga", "airtime", "data",

        "minuta", "sms", "butumwa", "shaka", "nkeneye", "nifuza",

        "irekure", "ntagipimo", "byinshi", "nki", "guhitamo", "ubwoko"

    ]

    keywords = english_keywords + kinyarwanda_keywords

    if any(k in t for k in keywords):

        return True

    if re.search(r"\b(show|list|erekana)\b.*\b(bundle|bundles|plans|packages|ubwoko)\b", t):

        return True

    # Detect questions about "which"

    if re.search(r"\b(which|ni ikihe|ni uwuhe|nki)\b", t):

        return True

    return False





def is_bundle_balance_intent(text: Optional[str]) -> bool:

    if not text:

        return False

    t = text.lower()

    english_phrases = ["my bundles", "bundle balance", "remaining bundles", "remaining data", "remaining minutes", "remaining sms", "bundle balances"]

    kin_phrases = ["bundles zisigaye", "ibisigaye", "ibisigaye kuri bundle", "bundles zanjye", "ibisigaye kuri", "ibisigaye ku"]

    return any(p in t for p in english_phrases + kin_phrases)





def interpret_transfer(text: str):

    t = text.lower()

    phone_match = re.search(r"(\+?\d{7,15}|\b0\d{6,12}\b)", t)

    amount_match = re.search(r"(\d+(?:\.\d+)?)\s*(rwf|frw|frw|rwf)?", t)

    m2 = re.search(r"send\s+(\d+(?:\.\d+)?)\s+to\s+(\+?\d{7,15}|\b0\d{6,12}\b)", t)

    if m2:

        amt = float(m2.group(1))

        to_phone = normalize_phone(m2.group(2))

        return amt, to_phone

    m3 = re.search(r"(ohereza|kohereza|tanga)\s+(\d+(?:\.\d+)?)\s+(kuri|ku)\s+(\+?\d{7,15}|\b0\d{6,12}\b)", t)

    if m3:

        amt = float(m3.group(2))

        to_phone = normalize_phone(m3.group(4))

        return amt, to_phone

    if phone_match and amount_match:

        to_phone = normalize_phone(phone_match.group(1))

        amt = float(amount_match.group(1))

        return amt, to_phone

    return None, None





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

}





def interpret_choice(user_text: str, options: list):

    t = normalize(user_text)

    if not t:

        return None

    m = re.search(r"\b(\d+)\b", t)

    if m:

        idx = int(m.group(1))

        return next((o for o in options if int(o.get("index", 0)) == idx), None)

    for w, idx in ORDINALS.items():

        if re.search(rf"\b{re.escape(w)}\b", t):

            return next((o for o in options if int(o.get("index", 0)) == idx), None)

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



    # ----- DIRECT BUNDLE MATCHING (most specific) -----

    if main and sub and period:

        bundles = fetch_bundles(main, sub, period, min_quantity=qty_val if qty_val else None)

        if bundles:

            options = []

            lines = [tr("bundles_matching", lang, name=user.get("name") or "")]

            # If many bundles, show top ranked

            if len(bundles) > 5:

                top_bundles = rank_bundles(bundles, 5)

                lines.append(tr("found_bundles", lang, count=len(bundles), limit=5))

                display_bundles = top_bundles

            else:

                display_bundles = bundles

            for i, b in enumerate(display_bundles, start=1):

                display = tr("bundle_display", lang,

                             main=b['main_category'],

                             sub=b['sub_category'],

                             quantity=b['quantity'],

                             period=b['period'],

                             price=b['price'])

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

            if len(bundles) > 5:

                options.append({"index": 0, "display": tr("show_all_option", lang), "action": "show_all"})

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

            reply_text = "\n".join(lines)

            return {"reply": reply_text, "options": options, "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang), tr("help", lang)], "action_buttons": [

                {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"},

                {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"}

            ]}

        else:

            # No exact match, try sub+period

            bundles = fetch_bundles(None, sub, period)

            if bundles:

                options = []

                lines = [tr("no_exact_match", lang, name=user.get('name'), main=main, sub=sub, period=period)]

                top_bundles = rank_bundles(bundles, 5)

                for i, b in enumerate(top_bundles, start=1):

                    display = tr("bundle_display", lang,

                                 main=b['main_category'],

                                 sub=b['sub_category'],

                                 quantity=b['quantity'],

                                 period=b['period'],

                                 price=b['price'])

                    lines.append(f"{i}. {display}")

                    options.append({ "index": i, "display": display, "qp_id": int(b["qp_id"]), "name": f"{b['quantity']} {b['period']}", "price": float(b["price"]) })

                state = {"stage": "bundle_list", "options": options, "lang": lang}

                save_conversation_state(phone, state)

                lines.append(tr("reply_options_prompt", lang))

                reply_text = "\n".join(lines)

                return {"reply": reply_text, "options": options, "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang), tr("help", lang)], "action_buttons": [

                {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"},

                {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"}

            ]}

            # If still nothing, fall through to main/sub selection



    # ----- If only sub+period (or main+sub without period) -----

    if sub and period:

        bundles = fetch_bundles(None, sub, period)

        if bundles:

            options = []

            lines = [tr("here_are_bundles", lang, name=user.get('name'), sub=sub, period=period)]

            top_bundles = rank_bundles(bundles, 5)

            for i, b in enumerate(top_bundles, start=1):

                display = tr("bundle_display", lang,

                             main=b['main_category'],

                             sub=b['sub_category'],

                             quantity=b['quantity'],

                             period=b['period'],

                             price=b['price'])

                lines.append(f"{i}. {display}")

                options.append({ "index": i, "display": display, "qp_id": int(b["qp_id"]), "name": f"{b['quantity']} {b['period']}", "price": float(b["price"]) })

            if len(bundles) > 5:

                options.append({"index": 0, "display": tr("show_all_option", lang), "action": "show_all"})

            state = {"stage": "bundle_list", "options": options, "lang": lang}

            save_conversation_state(phone, state)

            lines.append(tr("reply_options_prompt", lang))

            reply_text = "\n".join(lines)

            return {"reply": reply_text, "options": options, "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang), tr("help", lang)], "action_buttons": [

                {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"},

                {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"}

            ]}

        # else fall through



    # ----- If only main+sub (no period) -----

    if main and sub:

        periods = list_periods_for_sub(sub)

        if periods:

            options = [{"index": i + 1, "name": p, "display": p} for i, p in enumerate(periods)]

            state = {"stage": "period_list", "selected_main": main, "selected_sub": sub, "options": options, "lang": lang}

            save_conversation_state(phone, state)

            lines = [tr("choose_period", lang, name=user.get("name") or "", sub=sub)]

            for o in options:

                lines.append(f"{o['index']}. {o['display']}")

            return {"reply": "\n".join(lines), "options": options, "quick_replies": [tr("show_bundles", lang)], "action_buttons": [

                {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"},

                {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"}

            ]}



    # ----- If only sub (no main, no period) -----

    if sub and not main and not period:

        periods = list_periods_for_sub(sub)

        if periods:

            options = [{"index": i + 1, "name": p, "display": p} for i, p in enumerate(periods)]

            state = {"stage": "period_list", "selected_sub": sub, "options": options, "lang": lang}

            save_conversation_state(phone, state)

            lines = [tr("you_asked_about", lang, name=user.get('name'), sub=sub)]

            for o in options:

                lines.append(f"{o['index']}. {o['display']}")

            return {"reply": "\n".join(lines), "options": options, "quick_replies": [tr("help", lang)], "action_buttons": [

                {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"},

                {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"}

            ]}



    # ----- If only main (no sub, no period) -----

    if main and not sub and not period:

        subs = list_subcategories(main)

        if subs:

            options = [{"index": i + 1, "name": s, "display": s} for i, s in enumerate(subs)]

            state = {"stage": "sub_list", "selected_main": main, "options": options, "lang": lang}

            save_conversation_state(phone, state)

            lines = [tr("choose_sub", lang, name=user.get("name") or "", main=main)]

            for o in options:

                lines.append(f"{o['index']}. {o['display']}")

            return {"reply": "\n".join(lines), "options": options, "quick_replies": [tr("help", lang)], "action_buttons": [

                {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"},

                {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"}

            ]}



    # ----- If only period -----

    if period and not main and not sub:

        bundles = fetch_bundles(None, None, period)

        if bundles:

            options = []

            lines = [tr("here_are_period_bundles", lang, name=user.get('name'), period=period)]

            top_bundles = rank_bundles(bundles, 5)

            for i, b in enumerate(top_bundles, start=1):

                display = tr("bundle_display", lang,

                             main=b['main_category'],

                             sub=b['sub_category'],

                             quantity=b['quantity'],

                             period=b['period'],

                             price=b['price'])

                lines.append(f"{i}. {display}")

                options.append({ "index": i, "display": display, "qp_id": int(b["qp_id"]), "name": f"{b['quantity']} {b['period']}", "price": float(b["price"]) })

            if len(bundles) > 5:

                options.append({"index": 0, "display": tr("show_all_option", lang), "action": "show_all"})

            state = {"stage": "bundle_list", "options": options, "lang": lang}

            save_conversation_state(phone, state)

            lines.append("Reply with the number to buy, or say 'show all' to see everything.")

            reply_text = "\n".join(lines)

            return {"reply": reply_text, "options": options, "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang), tr("help", lang)], "action_buttons": [

                {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"},

                {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"}

            ]}

        else:

            # No bundles for this period – maybe show all categories?

            pass



    # ----- If quantity only -----

    if qty_val and not main and not sub and not period:

        bundles = fetch_bundles(None, None, None, min_quantity=qty_val)

        if bundles:

            options = []

            lines = [tr("bundles_with_quantity", lang, name=user.get('name'), quantity=qty_val, unit=qty_unit)]

            top_bundles = rank_bundles(bundles, 5)

            for i, b in enumerate(top_bundles, start=1):

                display = tr("bundle_display", lang,

                             main=b['main_category'],

                             sub=b['sub_category'],

                             quantity=b['quantity'],

                             period=b['period'],

                             price=b['price'])

                lines.append(f"{i}. {display}")

                options.append({ "index": i, "display": display, "qp_id": int(b["qp_id"]), "name": f"{b['quantity']} {b['period']}", "price": float(b["price"]) })

            state = {"stage": "bundle_list", "options": options, "lang": lang}

            save_conversation_state(phone, state)

            lines.append(tr("reply_options_prompt", lang))

            reply_text = "\n".join(lines)

            return {"reply": reply_text, "options": options, "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang), tr("help", lang)], "action_buttons": [

                {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"},

                {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"}

            ]}



    # ----- Default: show main categories -----

    mains = list_main_categories()

    if not mains:

        return {

            "reply": tr("no_bundles", lang, name=user.get("name") or ""),

            "action_buttons": [

                {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},

                {"text": "💰 Check Balance", "action": "check_balance", "type": "info"},

                {"text": "📞 Contact Support", "action": "help", "type": "info"}

            ]

        }

    options = [{"index": i + 1, "name": m, "display": m} for i, m in enumerate(mains)]

    state = {"stage": "main_list", "options": options, "lang": lang}

    save_conversation_state(phone, state)

    lines = [tr("choose_main", lang, name=user.get("name") or "")]

    for o in options:

        lines.append(f"{o['index']}. {o['display']}")

    return {"reply": "\n".join(lines), "options": options, "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang), tr("help", lang)], "action_buttons": [

                {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"},

                {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"}

            ]}





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

        # FRIENDLY: Suggest recharge

        return {"reply": tr("insufficient_funds", lang, balance=user_balance), "status": "insufficient_funds", "suggest_recharge": True}

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

        # FRIENDLY: Add follow-up suggestion

        follow_up = " Would you like to check your bundle balance or buy another one?"

        return {"reply": tr("purchase_success", lang, quantity=qp.get("quantity"), new_balance=new_balance) + follow_up, "status": "ok", "receipt": receipt}

    except Exception:

        logger.exception("Purchase failed for %s", phone)

        return {"reply": "Purchase failed due to server error.", "status": "error", "code": 500}





# ---------------------------

# Transfer: atomic implementation + backward-compatible wrapper

# ---------------------------



# Fraud detection limits

def check_transfer_limits_atomic(cursor, from_phone: str, amount: float, lang: str = "en") -> Tuple[bool, Optional[str]]:

    # Daily total limit (e.g., 50,000)

    today = datetime.utcnow().date().isoformat()

    # Hourly count limit (e.g., 5 transfers per hour)

    hour_key = datetime.utcnow().strftime("%Y-%m-%d-%H")



    # Daily total

    cursor.execute(

        "SELECT total_amount FROM transfer_limits_daily WHERE phone_number=%s AND date=%s FOR UPDATE",

        (from_phone, today)

    )

    row = cursor.fetchone()

    daily_total = float(row[0]) if row else 0.0

    if daily_total + amount > 50000:

        return False, tr("send_airtime_failed", lang) + " Daily transfer limit of 50,000 exceeded. You can try again tomorrow."



    # Hourly count

    cursor.execute(

        "SELECT transfer_count FROM transfer_limits_hourly WHERE phone_number=%s AND hour_key=%s FOR UPDATE",

        (from_phone, hour_key)

    )

    row = cursor.fetchone()

    hourly_count = int(row[0]) if row else 0

    if hourly_count >= 5:

        return False, tr("send_airtime_failed", lang) + " You've sent too many transfers this hour. Please wait a bit."



    # Update counters (insert or update)

    if row:

        cursor.execute(

            "UPDATE transfer_limits_hourly SET transfer_count = %s WHERE phone_number=%s AND hour_key=%s",

            (hourly_count + 1, from_phone, hour_key)

        )

    else:

        cursor.execute(

            "INSERT INTO transfer_limits_hourly (phone_number, hour_key, transfer_count) VALUES (%s, %s, %s)",

            (from_phone, hour_key, 1)

        )



    if daily_total == 0:

        cursor.execute(

            "INSERT INTO transfer_limits_daily (phone_number, date, total_amount) VALUES (%s, %s, %s)",

            (from_phone, today, amount)

        )

    else:

        cursor.execute(

            "UPDATE transfer_limits_daily SET total_amount = %s WHERE phone_number=%s AND date=%s",

            (daily_total + amount, from_phone, today)

        )



    return True, None





def process_transfer_atomic(from_phone: str, to_phone: str, amount: float, lang: str = "en") -> Dict[str, Any]:

    """

    Atomic transfer using DB transaction and SELECT ... FOR UPDATE.

    Returns dict with status/reply/receipt similar to previous process_transfer.

    """

    conn = None

    cur = None

    try:

        from_phone = normalize_phone(from_phone)

        to_phone = normalize_phone(to_phone)

        if not from_phone or not to_phone:

            return {"status": "error", "reply": tr("send_airtime_failed", lang)}

        if amount <= 0:

            return {"status": "error", "reply": "Invalid amount. Please enter a positive number."}



        conn = get_db_connection()

        cur = conn.cursor()

        try:

            # --- Fraud detection (if Redis not available, do inside transaction) ---

            if redis_client:

                # Use Redis pipeline for atomic operations

                daily_key = f"transfer_daily:{from_phone}:{datetime.utcnow().date().isoformat()}"

                hourly_key = f"transfer_hourly:{from_phone}:{datetime.utcnow().strftime('%Y-%m-%d-%H')}"

                

                # Check current limits

                daily_total = float(redis_client.get(daily_key) or 0)

                hourly_count = int(redis_client.get(hourly_key) or 0)

                if daily_total + amount > 50000:

                    return {"status": "error", "reply": tr("send_airtime_failed", lang) + " Daily transfer limit of 50,000 exceeded. You can try again tomorrow."}

                if hourly_count >= 5:

                    return {"status": "error", "reply": tr("send_airtime_failed", lang) + " You've sent too many transfers this hour. Please wait a bit."}

                

                # Prepare atomic increment

                pipe = redis_client.pipeline()

                pipe.incrbyfloat(daily_key, amount)

                pipe.expire(daily_key, 86400)

                pipe.incr(hourly_key)

                pipe.expire(hourly_key, 3600)

                # We'll execute after successful DB commit

                redis_pipe = pipe

            else:

                # Check and update limits inside transaction

                allowed, msg = check_transfer_limits_atomic(cur, from_phone, amount, lang)

                if not allowed:

                    conn.rollback()

                    return {"status": "error", "reply": msg}

                redis_pipe = None

            # ----------------------



            # Lock sender balance row

            cur.execute("SELECT balance FROM airtime_balance WHERE phone_number=%s FOR UPDATE", (from_phone,))

            row = cur.fetchone()

            if not row:

                conn.rollback()

                return {"status": "error", "reply": "Sender account not found", "code": 404}

            sender_balance = float(row[0] or 0.0)

            if sender_balance < amount:

                conn.rollback()

                # FRIENDLY: Suggest recharge

                return {"status": "insufficient_funds", "reply": tr("send_airtime_insufficient", lang, balance=sender_balance, amount=amount), "suggest_recharge": True}



            # Debit sender

            new_sender_balance = round(sender_balance - amount, 2)

            cur.execute("UPDATE airtime_balance SET balance=%s WHERE phone_number=%s", (new_sender_balance, from_phone))



            # Lock recipient row or create

            cur.execute("SELECT balance FROM airtime_balance WHERE phone_number=%s FOR UPDATE", (to_phone,))

            rrow = cur.fetchone()

            if rrow:

                recipient_balance = float(rrow[0] or 0.0) + amount

                cur.execute("UPDATE airtime_balance SET balance=%s WHERE phone_number=%s", (round(recipient_balance, 2), to_phone))

            else:

                recipient_balance = round(amount, 2)

                cur.execute("INSERT INTO airtime_balance (phone_number, balance) VALUES (%s, %s)", (to_phone, recipient_balance))



            # record transfer best-effort

            try:

                now = datetime.utcnow()

                cur.execute("INSERT INTO airtime_transfers (from_phone, to_phone, amount, created_at) VALUES (%s, %s, %s, %s)",

                            (from_phone, to_phone, amount, now))

            except Exception:

                logger.debug("airtime_transfers table missing or insert failed - continuing")



            conn.commit()



            # Execute Redis pipeline if using Redis

            if redis_pipe:

                try:

                    redis_pipe.execute()

                    logger.info("Redis limits updated successfully")

                except Exception as e:

                    logger.error(f"Failed to update Redis limits: {e}")

                    # Transfer succeeded but limits not tracked - log for manual review



            receipt = {"from": from_phone, "to": to_phone, "amount": amount, "timestamp": datetime.utcnow().isoformat()}

            logger.info("Transfer successful: %s -> %s amount=%s", from_phone, to_phone, amount)

            # FRIENDLY: Add follow-up

            follow_up = " Need anything else? You can check balance or buy a bundle."

            return {"status": "ok", "reply": tr("send_airtime_success", lang, amount=amount, to_phone=to_phone, new_balance=new_sender_balance) + follow_up, "receipt": receipt}

        except Exception:

            if conn:

                conn.rollback()

            logger.exception("Transfer DB failure")

            return {"status": "error", "reply": tr("send_airtime_failed", lang)}

        finally:

            try:

                if cur:

                    cur.close()

            except Exception:

                pass

    except Exception:

        logger.exception("Unexpected transfer failure")

        return {"status": "error", "reply": tr("send_airtime_failed", lang)}

    finally:

        try:

            if conn:

                conn.close()

        except Exception:

            pass





def process_transfer(from_phone: str, to_phone: str, amount: float, lang: str = "en"):



    return process_transfer_atomic(from_phone, to_phone, amount, lang)





# ---------------------------

# Smalltalk reply (uses tr() for all responses) - ENHANCED with more patterns

# ---------------------------

def smalltalk_reply(user: dict, message: str, lang: str = "en"):

    name = user.get("name") or ""

    t = (message or "").lower()

    reply = None

    if is_greeting(t):

        reply = tr("greeting_intro", lang, name=name)

    elif is_goodbye(t):

        reply = tr("goodbye", lang, name=name)

    elif is_thanks(t):

        reply = tr("thanks", lang, name=name)

    elif re.search(r"\b(who are you|what are you|your name|uri nde)\b", t):

        reply = tr("who_are_you", lang)

    elif re.search(r"\b(how are you|how's it going|how you doing|amere)\b", t):

        reply = tr("how_are_you", lang)

    elif re.search(r"\b(help|support|assist|can you|ufasha|ubufasha)\b", t):

        reply = tr("support_prompt", lang)

    # FRIENDLY: More small talk patterns

    elif re.search(r"\b(what'?s up|how are things|how is it going)\b", t):

        reply = f"All good here, {name}! Ready to help you with anything telecom. How about you? 😊"

    elif re.search(r"\b(can you|would you|will you)\b.*\b(help|assist)\b", t):

        reply = f"Of course, {name}! That's what I'm here for. What can I do for you today? 😄"

    elif re.search(r"\b(thanks|thank you)\b.*\b(help|assist)\b", t):

        reply = f"You're very welcome, {name}! Happy to assist. Let me know if you need anything else."

    elif re.search(r"\b(what can you do|what do you do)\b", t):

        reply = f"I can help you check airtime balance, view bundles, buy data/calling bundles, send airtime, and more! Just ask me. 😊"



    if reply:

        quick = [tr("show_bundles", lang), tr("my_balance", lang), tr("help", lang)]

        return {

            "reply": reply, 

            "quick_replies": quick,

            "action_buttons": [

                {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"},

                {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},

                {"text": "💰 Check Balance", "action": "check_balance", "type": "info"}

            ]

        }

    return None





# ---------------------------

# Endpoints (preserved & extended)

# ---------------------------

class LoginRequest(BaseModel):
    phone: str = Field(..., description="User phone number")
    pin: str = Field(..., pattern=r"^\d{5}$", description="5-digit PIN")

@app.post("/login")
@limiter.limit("10/minute")
def login(req: LoginRequest, request: Request):
    """Login endpoint - authenticate user with PIN"""
    phone = normalize_phone(req.phone or "")
    pin = req.pin
    logger.info("Login request: phone=%s", phone)
    
    # Validation
    if not phone or not pin:
        logger.warning("Login failed: missing phone or PIN")
        raise HTTPException(status_code=400, detail="Phone and PIN are required")
    
    if len(pin) != 5 or not pin.isdigit():
        logger.warning("Login failed: invalid PIN format for %s", phone)
        raise HTTPException(status_code=400, detail="PIN must be exactly 5 digits")
    
    try:
        # Find user and verify PIN
        user = fetch_user(phone)
        if not user:
            logger.warning("Login failed: user not found %s", phone)
            raise HTTPException(status_code=404, detail="User not found")
        
        # Verify PIN hash
        if not verify_pin(pin, user.get('pin_hash')):
            logger.warning("Login failed: invalid PIN for %s", phone)
            raise HTTPException(status_code=401, detail="Invalid PIN")
        
        # Create session
        ip_address = request.client.host
        session_token = create_user_session(phone, None, ip_address)
        
        logger.info("Login successful for %s", phone)
        
        return {
            "status": "success",
            "message": "Login successful",
            "session_token": session_token,
            "user": {
                "phone": phone,
                "name": user.get('name')
            }
        }
        
    except HTTPException:
        raise
    except Exception:
        logger.exception("Login failed for %s", phone)
        raise HTTPException(status_code=500, detail="Login failed due to server error")

class LegacyRegisterRequest(BaseModel):
    name: str = Field(..., description="User name")
    phone: str = Field(..., description="User phone number")
    pin: str = Field(..., pattern=r"^\d{5}$", description="5-digit PIN is required")
    confirm_pin: str = Field(..., pattern=r"^\d{5}$", description="Confirm 5-digit PIN")


@app.post("/register-debug")
@limiter.limit("10/minute")
async def register_debug(request: Request):
    """Debug endpoint to show exactly what's received"""
    import json
    try:
        body = await request.body()
        data = json.loads(body.decode())
        logger.info(f"DEBUG - Raw request body: {data}")
        logger.info(f"DEBUG - Required fields present: name={data.get('name')}, phone={data.get('phone')}, pin={data.get('pin')}, confirm_pin={data.get('confirm_pin')}")
        
        return {
            "received": data,
            "required_fields": {
                "name": data.get('name'),
                "phone": data.get('phone'), 
                "pin": data.get('pin'),
                "confirm_pin": data.get('confirm_pin')
            },
            "missing_fields": [field for field in ['name', 'phone', 'pin', 'confirm_pin'] if not data.get(field)]
        }
    except Exception as e:
        logger.error(f"DEBUG - Error parsing body: {e}")
        return {"error": str(e), "raw_body": body.decode()}

@app.post("/register")
@limiter.limit("10/minute")  # rate limit
def register(req: LegacyRegisterRequest, request: Request):
    """Register endpoint - now requires PIN for security"""
    logger.info("Register request received: phone=%s name=%s", req.phone, req.name)
    
    phone = normalize_phone(req.phone or "")
    name = (req.name or "").strip()
    pin = req.pin
    confirm_pin = req.confirm_pin
    
    # Validation
    if not phone or not name:
        logger.warning("Registration failed: missing phone or name")
        raise HTTPException(status_code=400, detail="Name and phone are required")
    
    if pin != confirm_pin:
        logger.warning("Registration failed: PINs do not match for %s", phone)
        raise HTTPException(status_code=400, detail="PINs do not match")
    
    if len(pin) != 5 or not pin.isdigit():
        logger.warning("Registration failed: invalid PIN format for %s", phone)
        raise HTTPException(status_code=400, detail="PIN must be exactly 5 digits")
    
    try:
        # Check if user already exists
        existing_user = fetch_one("SELECT phone_number FROM users WHERE phone_number = %s", (phone,))
        if existing_user:
            raise HTTPException(status_code=409, detail="Phone number already registered")
        
        # Hash PIN and create user
        pin_hash = hash_pin(pin)
        
        execute("""
            INSERT INTO users (phone_number, name, pin_hash, is_active)
            VALUES (%s, %s, %s, %s)
        """, (phone, name, pin_hash, True))
        
        # Create airtime balance
        execute("INSERT INTO airtime_balance (phone_number, balance) VALUES (%s, %s)", (phone, 0.0))
        
        # Create session
        ip_address = request.client.host
        session_token = create_user_session(phone, None, ip_address)
        
        logger.info("Registration successful for %s", phone)
        
        return {
            "status": "success",
            "message": "Registration successful with secure PIN",
            "session_token": session_token,
            "user": {
                "phone": phone,
                "name": name
            },
            "note": "Your account is now secured with a 5-digit PIN"
        }
        
    except HTTPException:
        raise
    except Exception:
        logger.exception("Registration failed for %s", phone)
        raise HTTPException(status_code=500, detail="Registration failed due to server error")


@app.get("/profile")

@limiter.limit("30/minute")

def get_profile(phone: str, request: Request):

    phone = normalize_phone(phone or "")

    # tolerant lookup

    user = find_user_by_phone_variants(phone) or fetch_user(phone)

    if not user:

        raise HTTPException(status_code=404, detail="User not found")



    # expire bundles for this user before showing

    try:

        expire_bundles_for_phone(user.get("phone_number") or phone)

    except Exception:

        logger.debug("Failed to expire bundles for profile view %s", phone)



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

        (user.get("phone_number") or phone,),

    )

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

@limiter.limit("10/minute")

def update_profile(payload: ProfileUpdate, request: Request):

    phone = normalize_phone(payload.phone or "")

    user = fetch_user(phone)

    if not user:

        raise HTTPException(status_code=404, detail="User not found")

    

    # Whitelist of allowed fields for SQL injection prevention

    ALLOWED_PROFILE_FIELDS = {"name", "bio", "language"}

    set_parts = []

    params = []

    

    if payload.name is not None and "name" in ALLOWED_PROFILE_FIELDS:

        set_parts.append("name=%s")

        params.append(payload.name.strip())

    if payload.bio is not None and "bio" in ALLOWED_PROFILE_FIELDS:

        set_parts.append("bio=%s")

        params.append(payload.bio.strip())

    if payload.language is not None and "language" in ALLOWED_PROFILE_FIELDS:

        set_parts.append("language=%s")

        params.append(payload.language)

    

    if not set_parts:

        return {"status": "ok", "reply": "No changes made."}

    

    # Safe SQL construction with validated field names

    sql = f"UPDATE users SET {', '.join(set_parts)} WHERE phone_number=%s"

    params.append(phone)

    try:

        execute(sql, tuple(params))

    except Exception:

        logger.exception("Failed updating profile for %s", phone)

        raise HTTPException(status_code=500, detail="Update failed")

    return {"status": "ok", "reply": "Profile updated successfully!"}





@app.post("/profile/avatar")

@limiter.limit("5/minute")

async def upload_avatar(phone: str = Form(...), avatar: UploadFile = File(...), request: Request = None):

    phone = normalize_phone(phone or "")

    user = find_user_by_phone_variants(phone) or fetch_user(phone)

    if not user:

        raise HTTPException(status_code=404, detail="User not found")

    

    # File validation

    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}

    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

    

    if not avatar.filename:

        raise HTTPException(status_code=400, detail="No file provided")

    

    # Check file extension

    ext = os.path.splitext(avatar.filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:

        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    

    # Check file size

    content = await avatar.read()

    if len(content) > MAX_FILE_SIZE:

        raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB")

    

    # Reset file pointer

    await avatar.seek(0)

    

    # Generate safe filename

    timestamp = int(datetime.utcnow().timestamp())

    safe_filename = f"{phone}_{timestamp}_{ext.lstrip('.')}"

    dest = os.path.join(UPLOAD_DIR, safe_filename)

    

    try:

        with open(dest, "wb") as f:

            f.write(content)

    except Exception:

        logger.exception("Failed saving uploaded avatar")

        raise HTTPException(status_code=500, detail="Failed to save file")

    

    avatar_url = f"/uploads/{safe_filename}"

    try:

        execute("UPDATE users SET avatar_url=%s WHERE phone_number=%s", (avatar_url, phone))

    except Exception:

        logger.exception("Failed to save avatar_url to DB")

    return {"status": "ok", "avatar_url": avatar_url, "reply": "Avatar uploaded successfully!"}





@app.get("/purchases")

@limiter.limit("30/minute")

def get_purchases(phone: str, request: Request, limit: int = 20):

    phone = normalize_phone(phone or "")

    user = find_user_by_phone_variants(phone) or fetch_user(phone)

    if not user:

        raise HTTPException(status_code=404, detail="User not found")



    # expire bundles before showing

    try:

        expire_bundles_for_phone(user.get("phone_number") or phone)

    except Exception:

        logger.debug("Failed to expire bundles for purchases view %s", phone)



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

        (user.get("phone_number") or phone, limit),

    )

    return {"purchases": purchases}





@app.get("/recommendations")

@limiter.limit("30/minute")

def get_recommendations(phone: str, request: Request, limit: int = 6):

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





@app.post("/chat")
# @limiter.limit("30/minute")  # Temporarily disabled to debug
def chat(req: ChatRequest, request: Request):

    phone = normalize_phone(req.phone or "")

    message = req.message or ""

    logger.info("Incoming chat from %s: %s", phone, message)



    # --- A/B test assignment (logging only) ---

    user_hash = int(hashlib.md5(phone.encode()).hexdigest(), 16) % 100

    variant = "B" if user_hash < 50 else "A"

    logger.info(f"A/B test: user {phone} assigned to variant {variant} for experiment 'chat_flow_v2'")

    # -------------------------------------------



    user = find_user_by_phone_variants(phone) or fetch_user(phone)

    if not user:

        alt_phone = phone.lstrip("+") if phone.startswith("+") else "+" + phone

        user = find_user_by_phone_variants(alt_phone) or fetch_user(alt_phone)

        if user:

            phone = alt_phone

        else:

            raise HTTPException(status_code=404, detail="User not found")



    user_lang_hint = None

    try:

        user_lang_hint = user.get("language")

        if user_lang_hint and user_lang_hint not in ("kin", "en"):

            user_lang_hint = None

    except Exception:

        user_lang_hint = None



    detected = detect_language_simple(message or "", user_lang_hint)
    
    user_lang = "kin" if detected == "kin" else "en"

    logger.info(f"Detected language for {phone}: {user_lang}")



    try:

        set_conversation_language(phone, user_lang)

    except Exception:

        logger.debug("Could not set conversation language")



    # smalltalk on raw message

    small = smalltalk_reply(user, message, user_lang)

    if small:

        return small



    # Check if this is a voice transcription with intent

    if message.startswith("voice:"):
        voice_text = message[6:]  # Remove "voice:" prefix

        intent_result = detect_voice_intent(voice_text)
        
        logger.info(f"Voice intent detected for {phone}: {intent_result['intent']} (confidence: {intent_result['confidence']})")
        
        # Handle high-confidence voice intents

        if intent_result["confidence"] > 0.6:
            intent = intent_result["intent"]
            
            if intent == "balance":
                balance = fetch_airtime(phone)
                reply = tr("my_balance", "en", name=user.get('name')) + f" {balance}"
                reply += " Need to top up or buy a bundle?"
                return {
                    "reply": reply, 
                    "quick_replies": ["Top up", "Buy bundle", "Not now"],
                    "action_buttons": [
                        {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},
                        {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},
                        {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"}
                    ]
                }
            
            elif intent == "bundles":
                bundles_result = show_bundles(phone, "en")
                return bundles_result
            
            elif intent == "recharge":
                return {
                    "reply": tr("recharge_prompt", "en"),
                    "quick_replies": ["MTN", "Airtel", "Tigo"],
                    "action_buttons": [
                        {"text": "💳 MTN Mobile Money", "action": "recharge", "type": "momo", "provider": "mtn"},
                        {"text": "💳 Airtel Money", "action": "recharge", "type": "momo", "provider": "airtel"},
                        {"text": "💳 Tigo Cash", "action": "recharge", "type": "momo", "provider": "tigo"}
                    ]
                }
            
            elif intent == "help":
                return {
                    "reply": tr("support_prompt", "en"),
                    "quick_replies": ["Check balance", "Buy bundles", "Send money"],
                    "action_buttons": [
                        {"text": "💰 Check Balance", "action": "check_balance", "type": "info"},
                        {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},
                        {"text": "💸 Send Money", "action": "send_money", "type": "transfer"}
                    ]
                }

    lower = (message or "").lower().strip()

    state = get_conversation_state(phone) or {}

    lang = state.get("lang", user_lang)



    balance_keywords = r"\b(airtime|account balance|my balance|balance|balance yanjye|amafaranga|balance yange|money|amafaranga)\b"
    bundle_keywords = r"\b(bundle|bundles|data|calling|internet|yamunika|amafaranga)\b"
    
    # Handle combined request for both balance and bundles
    if re.search(balance_keywords, lower) and re.search(bundle_keywords, lower):
        # Get both balance and bundles
        balance = fetch_airtime(phone)
        balance_reply = tr("my_balance", lang, name=user.get('name')) + f" {balance}"
        
        # Get bundles
        bundles_result = show_bundles(phone, lang)
        bundle_reply = bundles_result.get("reply", "")
        
        reply = f"{balance_reply}\n\n{bundle_reply}"
        # FRIENDLY: Add follow-up

        reply += " Need to top up or buy a bundle?"

        return {

            "reply": reply, 

            "quick_replies": ["Top up", "Buy bundle", "Not now"],

            "action_buttons": [

                {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},

                {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"}

            ]

        }



    # Balance query only – using tr()
    elif re.search(balance_keywords, lower) and not re.search(bundle_keywords, lower):
        balance = fetch_airtime(phone)
        reply = tr("my_balance", lang, name=user.get('name')) + f" {balance}"
        # FRIENDLY: Add follow-up
        reply += " Need to top up or buy a bundle?"
        return {
            "reply": reply, 
            "quick_replies": ["Top up", "Buy bundle", "Not now"],
            "action_buttons": [
                {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},
                {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},
                {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"}
            ]
        }

    # Handle numeric selection when in bundle_list stage

    state = get_conversation_state(phone)

    if state and state.get("stage") == "bundle_list":

        # Check if message is just a number (bundle selection)

        m_number = re.match(r"^\s*(\d+)\s*$", message.strip())

        if m_number:

            idx = int(m_number.group(1))

            options = state.get("options", [])

            chosen = next((o for o in options if int(o.get("index", 0)) == idx), None)

            if not chosen:

                return {

                    "reply": f"{user.get('name')}, option {idx} not found. Please choose a valid bundle number.", 

                    "options": options,

                    "action_buttons": [

                        {"text": "📱 Show Data Bundles", "action": "buy_bundle", "type": "data"},

                        {"text": "📞 Show Calling Bundles", "action": "buy_bundle", "type": "calling"},

                        {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"}

                    ]

                }

            result = process_purchase(phone, int(chosen.get("qp_id")), lang)

            if result.get("status") == "ok":

                clear_conversation_state(phone)

                # FRIENDLY: Add feedback question occasionally

                reply_text = result.get("reply")

                if random.random() < 0.2:  # 20% chance

                    reply_text += " " + tr("feedback_question", lang)

                return {

                    "reply": reply_text, 

                    "status": "ok", 

                    "receipt": result.get("receipt"), 

                    "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang)],

                    "action_buttons": [

                        {"text": "💰 Check Balance", "action": "check_balance", "type": "info"},

                        {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                        {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"}

                    ]

                }

            elif result.get("status") == "insufficient_funds":

                return {

                    "reply": result.get("reply"), 

                    "status": "insufficient_funds", 

                    "quick_replies": [tr("help", lang)],

                    "action_buttons": [

                        {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},

                        {"text": "💰 Check Balance", "action": "check_balance", "type": "info"}

                    ]

                }

            else:

                return {

                    "reply": result.get("reply"), 

                    "status": result.get("status"),

                    "action_buttons": [

                        {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},

                        {"text": "💰 Check Balance", "action": "check_balance", "type": "info"}

                    ]

                }



    # "purchase <n>" in chat

    m_purchase = re.match(r"^\s*(?:purchase|buy)\s+(\d+)\b", lower)

    if m_purchase:

        idx = int(m_purchase.group(1))

        state = get_conversation_state(phone)

        if not state or state.get("stage") not in ["bundle_list", "main_list", "sub_list", "period_list"]:

            return {

                "reply": tr("no_bundles", lang, name=user.get("name") or ""), 

                "quick_replies": [tr("show_bundles", lang)],

                "action_buttons": [

                    {"text": "📱 Show Data Bundles", "action": "buy_bundle", "type": "data"},

                    {"text": "📞 Show Calling Bundles", "action": "buy_bundle", "type": "calling"},

                    {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"}

                ]

            }

        options = state.get("options", [])

        

        # If we're not in bundle_list stage, we need to get the bundles from the state

        if state.get("stage") != "bundle_list" and options:

            # Try to find the selected option by index

            chosen_option = next((o for o in options if int(o.get("index", 0)) == idx), None)

            if chosen_option and chosen_option.get("qp_id"):

                # Direct purchase from main/sub/period list

                result = process_purchase(phone, int(chosen_option.get("qp_id")), lang)

                if result.get("status") == "ok":

                    clear_conversation_state(phone)

                    reply_text = result.get("reply")

                    if random.random() < 0.2:

                        reply_text += " " + tr("feedback_question", lang)

                    return {

                        "reply": reply_text, 

                        "status": "ok", 

                        "receipt": result.get("receipt"), 

                        "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang)],

                        "action_buttons": [

                            {"text": "💰 Check Balance", "action": "check_balance", "type": "info"},

                            {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                            {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"}

                        ]

                    }

                elif result.get("status") == "insufficient_funds":

                    return {

                        "reply": result.get("reply"), 

                        "status": "insufficient_funds", 

                        "quick_replies": [tr("help", lang)],

                        "action_buttons": [

                            {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},

                            {"text": "💰 Check Balance", "action": "check_balance", "type": "info"}

                        ]

                    }

                else:

                    return {

                        "reply": result.get("reply"), 

                        "status": result.get("status"),

                        "action_buttons": [

                            {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},

                            {"text": "💰 Check Balance", "action": "check_balance", "type": "info"}

                        ]

                    }

        # Special "Show all" option

        if idx == 0:

            # fetch all bundles for current filters

            main = state.get("selected_main")

            sub = state.get("selected_sub")

            period = state.get("selected_period")

            bundles = fetch_bundles(main, sub, period)

            if bundles:

                options = []

                lines = [tr("bundles_matching", lang, name=user.get("name") or "")]

                for i, b in enumerate(bundles[:20], start=1):

                    display = tr("bundle_display", lang,

                                 main=b['main_category'],

                                 sub=b['sub_category'],

                                 quantity=b['quantity'],

                                 period=b['period'],

                                 price=b['price'])

                    lines.append(f"{i}. {display}")

                    options.append({ "index": i, "display": display, "qp_id": int(b["qp_id"]), "name": f"{b['quantity']} {b['period']}", "price": float(b["price"]) })

                state["options"] = options

                save_conversation_state(phone, state)

                lines.append(tr("reply_options_prompt", lang))

                reply_text = "\n".join(lines)

                

                # Add action buttons for each bundle

                action_buttons = []

                for option in options[:5]:  # Limit to first 5 bundles

                    action_buttons.append({

                        "text": f"🛒 {option['name']} - {option['price']} RWF",

                        "action": "purchase_bundle",

                        "bundle_id": option["qp_id"],

                        "bundle_name": option["name"],

                        "price": option["price"]

                    })

                

                return {

                    "reply": reply_text, 

                    "options": options, 

                    "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang)],

                    "action_buttons": action_buttons

                }

        chosen = next((o for o in options if int(o.get("index", 0)) == idx), None)

        if not chosen:

            return {

                "reply": f"{user.get('name')}, option {idx} not found. Please choose a valid bundle number.", 

                "options": options,

                "action_buttons": [

                    {"text": "📱 Show Data Bundles", "action": "buy_bundle", "type": "data"},

                    {"text": "📞 Show Calling Bundles", "action": "buy_bundle", "type": "calling"},

                    {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"}

                ]

            }

        result = process_purchase(phone, int(chosen.get("qp_id")), lang)

        if result.get("status") == "ok":

            clear_conversation_state(phone)

            reply_text = result.get("reply")

            if random.random() < 0.2:

                reply_text += " " + tr("feedback_question", lang)

            return {

                "reply": reply_text, 

                "status": "ok", 

                "receipt": result.get("receipt"), 

                "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang)],

                "action_buttons": [

                    {"text": "💰 Check Balance", "action": "check_balance", "type": "info"},

                    {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                    {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"}

                ]

            }

        elif result.get("status") == "insufficient_funds":

            return {

                "reply": result.get("reply"), 

                "status": "insufficient_funds", 

                "quick_replies": [tr("help", lang)],

                "action_buttons": [

                    {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},

                    {"text": "💰 Check Balance", "action": "check_balance", "type": "info"}

                ]

            }

        else:

            return {

                "reply": result.get("reply"), 

                "status": result.get("status"),

                "action_buttons": [

                    {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},

                    {"text": "💰 Check Balance", "action": "check_balance", "type": "info"}

                ]

            }



    # Bundle balances intent

    if is_bundle_balance_intent(lower):

        try:

            expire_bundles_for_phone(phone)

        except Exception:

            logger.debug("Failed to expire bundles on bundle_balance intent for %s", phone)

        bundles = fetch_bundle_balances(phone)

        if not bundles:

            return {

                "reply": tr("no_bundles_found", lang, name=user.get("name") or ""), 

                "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang)],

                "action_buttons": [

                    {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                    {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"},

                    {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"}

                ]

            }

        lines = [tr("bundle_balances_header", lang, name=user.get("name") or "")]

        for i, b in enumerate(bundles[:20], start=1):

            purchase_date = b.get("purchase_date")

            purchase_date_str = purchase_date.isoformat() if isinstance(purchase_date, datetime) else str(purchase_date)

            lines.append(tr("bundle_item_line", lang,

                            idx=i,

                            main=b.get("main_category"),

                            sub=b.get("sub_category"),

                            remaining=b.get("remaining"),

                            date=purchase_date_str))

        reply_text = "\n".join(lines)

        # FRIENDLY: Add follow-up

        reply_text += "\n\nWant to buy another bundle or check airtime balance?"

        return {

            "reply": reply_text, 

            "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang)],

            "action_buttons": [

                {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"},

                {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"}

            ]

        }



    # Send airtime intent (chat)

    if is_send_airtime_intent(lower):

        amt, to_phone = interpret_transfer(lower)

        if not amt or not to_phone:

            return {

                "reply": tr("transfer_parse_error", lang) if lang == "kin" else "Could not parse transfer command. Try 'send 100 to +2507...' or 'ohereza 100 kuri 0788...'", 

                "quick_replies": [tr("help", lang)],

                "action_buttons": [

                    {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},

                    {"text": "💰 Check Balance", "action": "check_balance", "type": "info"},

                    {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"}

                ]

            }

        transfer_result = process_transfer_atomic(phone, to_phone, float(amt), lang)

        if transfer_result.get("status") == "ok":

            reply_text = transfer_result.get("reply")

            if random.random() < 0.2:

                reply_text += " " + tr("feedback_question", lang)

            return {

                "reply": reply_text, 

                "receipt": transfer_result.get("receipt"), 

                "quick_replies": [tr("my_balance", lang), tr("show_bundles", lang)],

                "action_buttons": [

                    {"text": "💰 Check Balance", "action": "check_balance", "type": "info"},

                    {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                    {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"}

                ]

            }

        if transfer_result.get("status") == "insufficient_funds":

            return {

                "reply": transfer_result.get("reply"), 

                "quick_replies": [tr("help", lang)],

                "action_buttons": [

                    {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},

                    {"text": "💰 Check Balance", "action": "check_balance", "type": "info"}

                ]

            }

        if transfer_result.get("status") == "error":

            return {

                "reply": transfer_result.get("reply"), 

                "quick_replies": [tr("help", lang)],

                "action_buttons": [

                    {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},

                    {"text": "💰 Check Balance", "action": "check_balance", "type": "info"}

                ]

            }



    # Smart start or other flows

    if is_bundle_intent(lower) or extract_category(lower)[0] or extract_category(lower)[1] or parse_period(lower) or parse_quantity(lower)[0]:

        return smart_start_flow(phone, message, user)



    # Enhanced fallback with suggestions (translated)

    fallback = tr("default_fallback", lang, name=user.get('name'))

    return {

        "reply": fallback, 

        "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang), tr("help", lang)],

        "action_buttons": [

            {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

            {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"},

            {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},

            {"text": "💰 Check Balance", "action": "check_balance", "type": "info"}

        ]

    }





@app.post("/action")

@limiter.limit("20/minute")

def handle_action(req: ActionRequest, request: Request):

    """Handle action button clicks from chat interface"""

    phone = normalize_phone(req.phone or "")

    action = req.action or ""

    action_type = req.action_type or ""

    bundle_id = req.bundle_id

    bundle_name = req.bundle_name or ""

    price = req.price or 0

    

    logger.info("Action request from %s: %s (%s)", phone, action, action_type)

    

    user = find_user_by_phone_variants(phone) or fetch_user(phone)

    if not user:

        raise HTTPException(status_code=404, detail="User not found")

    

    state = get_conversation_state(phone) or {}

    lang = state.get("lang", "en")

    

    if action == "purchase_bundle" and bundle_id:

        # Handle bundle purchase

        result = process_purchase(phone, int(bundle_id), lang)

        if result.get("status") == "ok":

            clear_conversation_state(phone)

            reply_text = f"✅ Successfully purchased {bundle_name} for {price} RWF!"

            if random.random() < 0.2:

                reply_text += " " + tr("feedback_question", lang)

            return {

                "reply": reply_text,

                "status": "ok", 

                "receipt": result.get("receipt"),

                "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang)],

                "action_buttons": [

                    {"text": "💰 Check Balance", "action": "check_balance", "type": "info"},

                    {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                    {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"}

                ]

            }

        elif result.get("status") == "insufficient_funds":

            return {

                "reply": result.get("reply"), 

                "status": "insufficient_funds", 

                "quick_replies": [tr("help", lang)],

                "action_buttons": [

                    {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},

                    {"text": "💰 Check Balance", "action": "check_balance", "type": "info"}

                ]

            }

        else:

            return {"reply": result.get("reply"), "status": result.get("status")}

    

    elif action == "recharge":

        # Handle airtime recharge

        return {

            "reply": "💳 Please enter the amount to recharge (e.g., 'recharge 500')",

            "quick_replies": ["100 RWF", "500 RWF", "1000 RWF", "2000 RWF"],

            "action_buttons": [

                {"text": "💳 100 RWF", "action": "recharge_amount", "amount": 100},

                {"text": "💳 500 RWF", "action": "recharge_amount", "amount": 500},

                {"text": "💳 1000 RWF", "action": "recharge_amount", "amount": 1000},

                {"text": "💳 2000 RWF", "action": "recharge_amount", "amount": 2000}

            ]

        }

    

    elif action == "recharge_amount" and price > 0:

        # Handle specific recharge amount

        return {

            "reply": f"💳 Processing recharge of {price} RWF. Please complete payment through your mobile provider.",

            "quick_replies": [tr("my_balance", lang), tr("show_bundles", lang)],

            "action_buttons": [

                {"text": "💰 Check Balance", "action": "check_balance", "type": "info"},

                {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"}

            ]

        }

    

    elif action == "check_balance":

        # Check balance

        balance = fetch_airtime(phone)

        reply = tr("my_balance", lang, name=user.get('name')) + f" {balance}"

        reply += " Need to top up or buy a bundle?"

        return {

            "reply": reply,

            "quick_replies": ["Top up", "Buy bundle", "Not now"],

            "action_buttons": [

                {"text": "💳 Recharge Airtime", "action": "recharge", "type": "airtime"},

                {"text": "📱 Buy Data Bundle", "action": "buy_bundle", "type": "data"},

                {"text": "📞 Buy Calling Bundle", "action": "buy_bundle", "type": "calling"}

            ]

        }

    

    elif action == "buy_bundle":

        # Show bundles by type

        if action_type == "data":

            return smart_start_flow(phone, "show data bundles", user)

        elif action_type == "calling":

            return smart_start_flow(phone, "show calling bundles", user)

    

    else:

        # Default fallback

        return {

            "reply": "Please choose an action from the buttons above.",

            "quick_replies": [tr("show_bundles", lang), tr("my_balance", lang), tr("help", lang)]

        }





@app.post("/purchase")

@limiter.limit("10/minute")

def purchase(req: PurchaseRequest, request: Request):

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





@app.post("/transfer")

@limiter.limit("5/minute")

def transfer(req: TransferRequest, request: Request):

    from_phone = normalize_phone(req.from_phone or "")

    to_phone = normalize_phone(req.to_phone or "")

    amount = float(req.amount or 0)

    logger.info("Transfer requested: %s -> %s amount=%s", from_phone, to_phone, amount)

    from_user = find_user_by_phone_variants(from_phone) or fetch_user(from_phone)

    if not from_user:

        raise HTTPException(status_code=404, detail="Sender not found")

    result = process_transfer_atomic(from_user.get("phone_number") or from_phone, to_phone, amount, from_user.get("language") or "en")

    if result.get("status") == "error" and result.get("code") == 404:

        raise HTTPException(status_code=404, detail=result.get("reply"))

    if result.get("status") == "error":

        raise HTTPException(status_code=500, detail=result.get("reply"))

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





@app.post("/transcribe/test")
@limiter.limit("10/minute")
async def test_transcription(request: Request):
    """Test endpoint to verify transcription system is working"""
    try:
        # Test if Whisper model loads
        model = get_whisper_model("base")
        logger.info("Whisper model loaded successfully for test")
        
        return {
            "status": "ready",
            "message": "Transcription system is working",
            "model": "whisper_base",
            "test": "Try sending audio to /transcribe endpoint"
        }
    except Exception as e:
        logger.exception("Transcription test failed")
        return {
            "status": "error", 
            "message": f"Transcription system error: {str(e)}"
        }

@app.post("/debug/echo")

async def debug_echo(request: Request):

    payload = await request.json()

    phone = normalize_phone(payload.get("phone", ""))

    state = get_conversation_state(phone)

    logger.info("DEBUG ECHO phone=%s payload=%s state=%s", phone, payload, state)

    return {"payload": payload, "state": state}


# ---------------------------
# ENHANCED ENDPOINT: Voice Transcription - English Only
# ---------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring"""
    try:
        # Test database connection
        conn = get_db_connection()
        conn.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    # Check Whisper model availability
    try:
        # Try to load tiny model (smallest, fastest)
        model = get_whisper_model("tiny")
        whisper_status = "ready"
    except Exception as e:
        whisper_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": db_status,
            "whisper": whisper_status,
            "environment": os.getenv("ENVIRONMENT", "development")
        },
        "endpoints": {
            "chat": "/chat",
            "transcribe": "/transcribe",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/transcribe")
@limiter.limit("10/minute")
async def transcribe_audio(
    phone: str = Form(...),
    language: str = Form("en"),  # Force English as default
    model_size: str = Form("base"),
    enable_code_mixing: bool = Form(False),  # Disable code-mixing for English-only
    confidence_threshold: float = Form(0.5),
    audio: UploadFile = File(...),
    request: Request = None
):



    logger.info(f"Enhanced transcription request: phone={phone}, language={language}, model_size={model_size}, enable_mixing={enable_code_mixing}")



    # Validate inputs - English only
    if language not in ("en"):
        language = "en"  # Force English if not explicitly en

    if model_size not in ("tiny", "base", "small", "medium", "large"):

        model_size = "base"

    if not 0.0 <= confidence_threshold <= 1.0:

        confidence_threshold = 0.5



    tmp_path = None

    try:

        # File validation for transcription

        ALLOWED_AUDIO_EXTENSIONS = {'.m4a', '.mp3', '.wav', '.ogg', '.flac', '.webm'}

        MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25MB

        

        if not audio.filename:

            raise HTTPException(status_code=400, detail="No audio file provided")

        

        # Check file extension

        ext = os.path.splitext(audio.filename)[1].lower()

        if ext not in ALLOWED_AUDIO_EXTENSIONS:

            raise HTTPException(status_code=400, detail=f"Invalid audio format. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}")

        

        # Create temporary file

        suffix = ext if ext else ".m4a"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:

            content = await audio.read()
            
            # Log actual audio file size for debugging
            logger.info(f"Audio file received: {len(content)} bytes, filename: {audio.filename}")
            
            # Check if audio is empty or too small (reduced threshold)
            if len(content) < 100:  # Less than 100 bytes (truly empty)
                logger.error(f"Audio file appears empty: {len(content)} bytes")
                return {
                    "text": "Audio file appears to be empty. Please check your microphone and try recording again.",
                    "detected_language": "unknown",
                    "confidence": 0.0,
                    "model_used": "validation",
                    "error": "audio_empty"
                }
            
            # Log warning for very small audio but still process
            if len(content) < 5000:  # Less than 5KB
                logger.warning(f"Audio file quite small: {len(content)} bytes - may be short recording")

            if len(content) > MAX_AUDIO_SIZE:
                raise HTTPException(status_code=400, detail="Audio file too large. Maximum size is 25MB")

            tmp.write(content)

            tmp_path = tmp.name

            # Log audio file info for debugging

            logger.info(f"Audio file saved: {tmp_path}, size: {len(content)} bytes")



        # Enhanced transcription with code-mixing support

        if enable_code_mixing:

            logger.info("Starting enhanced transcription with code-mixing")

            transcription_result = transcribe_with_code_mixing(tmp_path, language, model_size)

            logger.info(f"Code-mixing result: {transcription_result}")

            # Post-process for better Kinyarwanda accuracy

            processed_text = post_process_transcription(

                transcription_result["text"], 

                transcription_result["code_mixing"]

            )

            logger.info(f"Post-processed text: {processed_text}")

            # Additional cleanup for any remaining non-target language words

            final_text = clean_non_target_languages(processed_text, transcription_result["detected_language"])

            logger.info(f"Final text after cleanup: {final_text}")
        
        # English-only transcription with telecom-specific context
        model = get_whisper_model(model_size)
        whisper_lang = "en"
        
        # Enhanced English prompt for better accuracy
        initial_prompt = (
            "hello thank you please help me show buy get "
            "data bundles airtime phone money account "
            "balance transfer send receive check my "
            "show bundles buy data recharge airtime "
            "how much what is where can I need to "
            "top up credit amount available please"
        )
        # Optimize for English transcription
        temperature = 0.0  # Lower for more deterministic
        best_of = 5  # More attempts for better accuracy
        beam_size = 7  # Wider beam search
        
        result = model.transcribe(
            tmp_path,
            language=whisper_lang,
            fp16=False,
            task="transcribe",
            initial_prompt=initial_prompt,
            temperature=temperature,
            best_of=best_of,
            beam_size=beam_size
        )
        
        transcribed_text = result["text"].strip()
        detected_lang = result.get("language", "en")
        
        logger.info(f"English transcription completed: {transcribed_text[:100]}...")
        
        # Add intent detection for better understanding
        intent_result = detect_voice_intent(transcribed_text)
        
        return {
            "text": transcribed_text,
            "detected_language": detected_lang,
            "model_used": f"whisper_{model_size}",
            "confidence": 0.85,  # High confidence for English-only
            "intent": intent_result["intent"],
            "intent_confidence": intent_result["confidence"]
        }



    except Exception as e:

        logger.exception("Enhanced transcription failed")

        raise HTTPException(status_code=500, detail="Transcription service error")

    finally:

        # Improved cleanup with logging

        if tmp_path and os.path.exists(tmp_path):

            try:

                os.unlink(tmp_path)

                logger.debug(f"Cleaned up temporary file: {tmp_path}")

            except Exception as e:

                logger.error(f"Failed to cleanup temporary file {tmp_path}: {e}")





# ---------------------------

# ADDITIONAL ENDPOINT: Batch Transcription for Multiple Files

# ---------------------------

@app.post("/transcribe/batch")

@limiter.limit("5/minute")

async def transcribe_batch(

    phone: str = Form(...),

    language: str = Form("auto"),

    model_size: str = Form("base"),

    enable_code_mixing: bool = Form(True),

    audio_files: List[UploadFile] = File(...),

    request: Request = None

):

    """

    Transcribe multiple audio files in a batch.

    Useful for processing multiple voice messages.

    """

    if len(audio_files) > 10:  # Limit batch size

        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")

    

    results = []

    for i, audio_file in enumerate(audio_files):

        try:

            # Create temporary file for each audio

            tmp_path = None

            try:

                ALLOWED_AUDIO_EXTENSIONS = {'.m4a', '.mp3', '.wav', '.ogg', '.flac', '.webm'}

                ext = os.path.splitext(audio_file.filename)[1].lower()

                if ext not in ALLOWED_AUDIO_EXTENSIONS:

                    results.append({

                        "filename": audio_file.filename,

                        "error": "Invalid file format",

                        "index": i

                    })

                    continue

                

                suffix = ext if ext else ".m4a"

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:

                    content = await audio_file.read()

                    if len(content) > 25 * 1024 * 1024:  # 25MB limit

                        results.append({

                            "filename": audio_file.filename,

                            "error": "File too large",

                            "index": i

                        })

                        continue

                    tmp.write(content)

                    tmp_path = tmp.name

                

                # Transcribe using enhanced function

                transcription_result = transcribe_with_code_mixing(tmp_path, language, model_size)

                processed_text = post_process_transcription(

                    transcription_result["text"], 

                    transcription_result["code_mixing"]

                )

                

                results.append({

                    "filename": audio_file.filename,

                    "text": processed_text,

                    "detected_language": transcription_result["detected_language"],

                    "confidence": transcription_result["confidence"],

                    "model_used": transcription_result["model_used"],

                    "code_mixing": transcription_result["code_mixing"],

                    "index": i

                })

                

            except Exception as e:

                logger.error(f"Failed to transcribe {audio_file.filename}: {e}")

                results.append({

                    "filename": audio_file.filename,

                    "error": str(e),

                    "index": i

                })

            finally:

                if tmp_path and os.path.exists(tmp_path):

                    try:

                        os.unlink(tmp_path)

                    except Exception:

                        pass

                        

        except Exception as e:

            logger.error(f"Error processing file {audio_file.filename}: {e}")

            results.append({

                "filename": audio_file.filename,

                "error": "Processing error",

                "index": i

            })

    

    return {

        "phone": phone,

        "language": language,

        "model_size": model_size,

        "total_files": len(audio_files),

        "successful": len([r for r in results if "error" not in r]),

        "failed": len([r for r in results if "error" in r]),

        "results": results

    }





# ---------------------------

# NEW ENDPOINT: Conversation Summarization (using sumy) - FIXED

# ---------------------------

@app.post("/summarize")

@limiter.limit("20/minute")

def summarize_chat(payload: SummarizeRequest, request: Request):

    messages = payload.messages

    language = payload.language

    if not messages:

        raise HTTPException(status_code=400, detail="No messages provided")



    # Always use english tokenizer – it works fine for sentence splitting in any language

    tokenizer_lang = "english"

    try:

        parser = PlaintextParser.from_string(messages, Tokenizer(tokenizer_lang))

        summarizer = LsaSummarizer()

        summary_sentences = summarizer(parser.document, sentences_count=3)

        summary_text = " ".join(str(sentence) for sentence in summary_sentences)

        # Note: summarizer produces English; translation not applied here because summary is based on user messages.

        return {"summary": summary_text if summary_text else "Summary not available."}

    except Exception as e:

        logger.error(f"Summarization failed: {e}")

        return {"summary": "Unable to generate summary at this time."}





# ---------------------------

# NEW ENDPOINT: Loyalty Points Redemption

# ---------------------------

@app.post("/loyalty/redeem")

@limiter.limit("5/minute")

def redeem_loyalty(req: RedeemRequest, request: Request):

    phone = normalize_phone(req.phone or "")

    points = req.points

    user = fetch_user(phone)

    if not user:

        raise HTTPException(status_code=404, detail="User not found")

    current_points = user.get("loyalty_points", 0)

    if current_points < points:

        raise HTTPException(status_code=400, detail="Insufficient points")



    # Convert points to airtime with configurable conversion rate

    POINTS_TO_AIRTIME_RATE = float(os.getenv("POINTS_TO_AIRTIME_RATE", "0.01"))  # Default: 100 points = 1 RWF

    airtime_value = points * POINTS_TO_AIRTIME_RATE

    

    # Validate conversion result

    if airtime_value <= 0:

        raise HTTPException(status_code=400, detail="Invalid conversion rate")

    

    new_airtime = fetch_airtime(phone) + airtime_value

    update_airtime(phone, new_airtime)



    # Deduct points

    execute("UPDATE users SET loyalty_points = loyalty_points - %s WHERE phone_number=%s", (points, phone))



    return {

        "status": "ok",

        "new_airtime": new_airtime,

        "remaining_points": current_points - points,

        "conversion_rate": POINTS_TO_AIRTIME_RATE

    }



# ----------------------------------------------------------------------

# SMART Q&A ASR SYSTEM - Understands Questions & Provides Real Answers

# ----------------------------------------------------------------------

SERVICE_DATABASE = {

    "internet_bundles": [

        {

            "name": "Daily 100MB",

            "price": 50,

            "currency": "RWF",

            "data": "100MB",

            "validity": "1 day",

            "description": "Perfect for daily WhatsApp and social media",

            "suitable_for": "Light daily use, messaging apps"

        },

        {

            "name": "Daily 500MB", 

            "price": 100,

            "currency": "RWF",

            "data": "500MB",

            "validity": "1 day",

            "description": "Great for daily browsing and social media",

            "suitable_for": "Daily browsing, social media, light streaming"

        },

        {

            "name": "Daily 1GB",

            "price": 200,

            "currency": "RWF",

            "data": "1GB", 

            "validity": "1 day",

            "description": "Perfect for heavy daily internet use",

            "suitable_for": "Heavy browsing, streaming, video calls"

        },

        {

            "name": "Daily 2GB",

            "price": 350,

            "currency": "RWF",

            "data": "2GB",

            "validity": "1 day", 

            "description": "Ultimate daily package for power users",

            "suitable_for": "Power users, HD streaming, downloads"

        }

    ],

    "weekly_bundles": [

        {

            "name": "Weekly 1GB",

            "price": 500,

            "currency": "RWF",

            "data": "1GB",

            "validity": "7 days",

            "description": "Good for weekly moderate usage",

            "suitable_for": "Weekly browsing and messaging"

        },

        {

            "name": "Weekly 3GB",

            "price": 1200,

            "currency": "RWF",

            "data": "3GB",

            "validity": "7 days",

            "description": "Great for weekly heavy usage",

            "suitable_for": "Weekly streaming and browsing"

        },

        {

            "name": "Weekly 5GB",

            "price": 2000,

            "currency": "RWF",

            "data": "5GB",

            "validity": "7 days",

            "description": "Perfect for weekly power users",

            "suitable_for": "Power users, HD content weekly"

        }

    ],

    "monthly_bundles": [

        {

            "name": "Monthly 5GB",

            "price": 3000,

            "currency": "RWF",

            "data": "5GB",

            "validity": "30 days",

            "description": "Basic monthly package",

            "suitable_for": "Light monthly users"

        },

        {

            "name": "Monthly 10GB",

            "price": 5000,

            "currency": "RWF",

            "data": "10GB",

            "validity": "30 days",

            "description": "Popular monthly choice",

            "suitable_for": "Regular monthly users"

        },

        {

            "name": "Monthly 20GB",

            "price": 8000,

            "currency": "RWF",

            "data": "20GB",

            "validity": "30 days",

            "description": "Great for heavy monthly users",

            "suitable_for": "Heavy users, streaming, downloads"

        },

        {

            "name": "Monthly 50GB",

            "price": 15000,

            "currency": "RWF",

            "data": "50GB",

            "validity": "30 days",

            "description": "Ultimate monthly package",

            "suitable_for": "Power users, 4K streaming, gaming"

        }

    ],

    "calling_bundles": [

        {

            "name": "Daily 50 min",

            "price": 300,

            "currency": "RWF",

            "minutes": "50 min",

            "validity": "1 day",

            "description": "Perfect for daily calls to all networks",

            "suitable_for": "Daily calling needs"

        },

        {

            "name": "Daily 100 min",

            "price": 500,

            "currency": "RWF",

            "minutes": "100 min",

            "validity": "1 day",

            "description": "Great for heavy daily calling",

            "suitable_for": "Business calls, family calls"

        },

        {

            "name": "Pay As You Go 30 sec",

            "price": 50,

            "currency": "RWF",

            "seconds": "30 sec",

            "validity": "Per call",

            "description": "Pay per second for short calls",

            "suitable_for": "Quick calls, emergency calls"

        },

        {

            "name": "Pay As You Go 60 sec",

            "price": 100,

            "currency": "RWF",

            "seconds": "60 sec",

            "validity": "Per call",

            "description": "Pay per second for medium calls",

            "suitable_for": "Standard calls, business calls"

        },

        {

            "name": "Weekly 200 min",

            "price": 1500,

            "currency": "RWF",

            "minutes": "200 min",

            "validity": "7 days",

            "description": "Good for weekly moderate usage",

            "suitable_for": "Weekly calling needs"

        },

        {

            "name": "Weekly 500 min",

            "price": 3000,

            "currency": "RWF",

            "minutes": "500 min",

            "validity": "7 days",

            "description": "Great for weekly heavy usage",

            "suitable_for": "Business users, heavy callers"

        },

        {

            "name": "Monthly 1000 min",

            "price": 8000,

            "currency": "RWF",

            "minutes": "1000 min",

            "validity": "30 days",

            "description": "Perfect for monthly calling needs",

            "suitable_for": "Regular monthly callers"

        },

        {

            "name": "Monthly Unlimited",

            "price": 15000,

            "currency": "RWF",

            "minutes": "Unlimited",

            "validity": "30 days",

            "description": "Ultimate monthly calling package",

            "suitable_for": "Heavy users, business calls"

        }

    ],

    "airtime_plans": [

        {

            "name": "Airtime 100 RWF",

            "price": 100,

            "currency": "RWF",

            "airtime": "100 RWF",

            "bonus": "0 RWF",

            "description": "Basic airtime recharge"

        },

        {

            "name": "Airtime 500 RWF",

            "price": 500,

            "currency": "RWF",

            "airtime": "500 RWF",

            "bonus": "50 RWF",

            "description": "Popular airtime with bonus"

        },

        {

            "name": "Airtime 1000 RWF",

            "price": 1000,

            "currency": "RWF",

            "airtime": "1000 RWF",

            "bonus": "150 RWF",

            "description": "Best value airtime package"

        },

        {

            "name": "Airtime 2000 RWF",

            "price": 2000,

            "currency": "RWF",

            "airtime": "2000 RWF",

            "bonus": "400 RWF",

            "description": "Premium airtime with big bonus"

        }

    ],

    "services": [

        {

            "name": "Mobile Banking",

            "description": "Bank from your phone anytime, anywhere",

            "features": ["Transfer money", "Pay bills", "Check balance", "Buy airtime"],

            "cost": "Free to use"

        },

        {

            "name": "Customer Care",

            "description": "24/7 customer support",

            "features": ["Live chat", "Phone support", "Email support", "Social media"],

            "cost": "Free",

            "contact": "Call 123 or email support@provider.com"

        },

        {

            "name": "International Calling",

            "description": "Call internationally at low rates",

            "features": ["USA: 50 RWF/min", "UK: 60 RWF/min", "Canada: 55 RWF/min"],

            "cost": "Per minute rates apply"

        }

    ]

}



class SmartQnAEngine:

    """Smart Q&A engine that understands questions and provides answers"""

    

    def __init__(self):

        self.intent_patterns = {

            "daily_bundles": {

                "keywords": ["daily", "today", "day", "internet bundle", "data bundle", "daily data"],

                "patterns": [

                    r"daily.*bundle",

                    r"today.*data", 

                    r"day.*internet",

                    r"what.*daily.*bundles",

                    r"daily.*data.*packages"

                ]

            },

            "weekly_bundles": {

                "keywords": ["weekly", "week", "7 days", "seven days"],

                "patterns": [

                    r"weekly.*bundle",

                    r"week.*data",

                    r"seven.*day.*bundle"

                ]

            },

            "monthly_bundles": {

                "keywords": ["monthly", "month", "30 days", "thirty days"],

                "patterns": [

                    r"monthly.*bundle",

                    r"month.*data",

                    r"thirty.*day.*bundle"

                ]

            },

            "calling_bundles": {

                "keywords": ["calling", "minutes", "mins", "seconds", "secs", "call", "voice", "talk"],

                "patterns": [

                    r"calling.*bundle",

                    r"minutes.*bundle",

                    r"seconds.*bundle",

                    r"voice.*package",

                    r"call.*bundle"

                ]

            },

            "airtime": {

                "keywords": ["airtime", "credit", "recharge", "top up", "load"],

                "patterns": [

                    r"airtime.*plans?",

                    r"recharge.*options?",

                    r"top.*up.*packages?"

                ]

            },

            "services": {

                "keywords": ["services", "features", "what do you offer", "available services"],

                "patterns": [

                    r"what.*services?",

                    r"available.*features?",

                    r"what.*do.*you.*offer?"

                ]

            },

            "prices": {

                "keywords": ["price", "cost", "how much", "rates", "fees"],

                "patterns": [

                    r"how.*much.*bundle",

                    r"price.*of.*data",

                    r"cost.*of.*airtime"

                ]

            },

            "recommendations": {

                "keywords": ["recommend", "suggest", "best", "which should", "what should"],

                "patterns": [

                    r"recommend.*bundle",

                    r"best.*data.*plan",

                    r"which.*bundle.*should"

                ]

            }

        }

    

    def analyze_question(self, text: str) -> Dict[str, Any]:

        """Analyze user question and determine intent"""

        text_lower = text.lower()

        

        # Score each intent

        intent_scores = {}

        

        for intent_name, intent_data in self.intent_patterns.items():

            score = 0

            

            # Keyword matching

            for keyword in intent_data["keywords"]:

                if keyword in text_lower:

                    score += 2

            

            # Pattern matching

            for pattern in intent_data["patterns"]:

                if re.search(pattern, text_lower):

                    score += 3

            

            intent_scores[intent_name] = score

        

        # Get best intent

        if not intent_scores or max(intent_scores.values()) == 0:

            return {

                "intent": "general",

                "confidence": 0.0,

                "text": text,

                "answer": self.get_general_answer()

            }

        

        best_intent = max(intent_scores, key=intent_scores.get)

        confidence = min(1.0, intent_scores[best_intent] / 10.0)

        

        return {

            "intent": best_intent,

            "confidence": confidence,

            "text": text,

            "answer": self.get_answer_for_intent(best_intent, text)

        }

    

    def get_answer_for_intent(self, intent: str, original_text: str) -> Dict[str, Any]:

        """Get specific answer based on intent"""

        

        if intent == "daily_bundles":

            return self.get_daily_bundles_answer()

        elif intent == "weekly_bundles":

            return self.get_weekly_bundles_answer()

        elif intent == "monthly_bundles":

            return self.get_monthly_bundles_answer()

        elif intent == "calling_bundles":

            return self.get_calling_bundles_answer()

        elif intent == "airtime":

            return self.get_airtime_answer()

        elif intent == "services":

            return self.get_services_answer()

        elif intent == "prices":

            return self.get_prices_answer(original_text)

        elif intent == "recommendations":

            return self.get_recommendations_answer(original_text)

        else:

            return self.get_general_answer()

    

    def get_daily_bundles_answer(self) -> Dict[str, Any]:

        """Get daily bundles information"""

        bundles = SERVICE_DATABASE["internet_bundles"]

        

        return {

            "type": "daily_bundles",

            "question": "Daily Internet Bundles",

            "answer": "Here are our daily internet bundles:",

            "bundles": bundles,

            "summary": f"We have {len(bundles)} daily bundles ranging from {bundles[0]['price']} RWF for {bundles[0]['data']} to {bundles[-1]['price']} RWF for {bundles[-1]['data']}.",

            "recommendation": "For daily use, I recommend the Daily 500MB package at 100 RWF - it's great value for social media and browsing!"

        }

    

    def get_weekly_bundles_answer(self) -> Dict[str, Any]:

        """Get weekly bundles information"""

        bundles = SERVICE_DATABASE["weekly_bundles"]

        

        return {

            "type": "weekly_bundles",

            "question": "Weekly Internet Bundles",

            "answer": "Here are our weekly internet bundles:",

            "bundles": bundles,

            "summary": f"We offer {len(bundles)} weekly bundles from {bundles[0]['price']} RWF for {bundles[0]['data']} to {bundles[-1]['price']} RWF for {bundles[-1]['data']}.",

            "recommendation": "The Weekly 3GB package at 1200 RWF is our most popular weekly choice!"

        }

    

    def get_monthly_bundles_answer(self) -> Dict[str, Any]:

        """Get monthly bundles information"""

        bundles = SERVICE_DATABASE["monthly_bundles"]

        

        return {

            "type": "monthly_bundles",

            "question": "Monthly Internet Bundles",

            "answer": "Here are our monthly internet bundles:",

            "bundles": bundles,

            "summary": f"We have {len(bundles)} monthly bundles ranging from {bundles[0]['price']} RWF for {bundles[0]['data']} to {bundles[-1]['price']} RWF for {bundles[-1]['data']}.",

            "recommendation": "For most users, the Monthly 10GB package at 5000 RWF offers the best value!"

        }

    

    def get_calling_bundles_answer(self) -> Dict[str, Any]:

        """Get calling bundles information"""

        bundles = SERVICE_DATABASE["calling_bundles"]

        

        return {

            "type": "calling_bundles",

            "question": "Calling Bundles",

            "answer": "Here are our calling bundles with minutes and seconds:",

            "bundles": bundles,

            "summary": f"We offer {len(bundles)} calling bundles from {bundles[0]['price']} RWF for {bundles[0].get('minutes', bundles[0].get('seconds'))} to {bundles[-1]['price']} RWF for {bundles[-1].get('minutes', bundles[-1].get('seconds'))}.",

            "recommendation": "For regular calling, I recommend the Weekly 200 Mins package at 1500 RWF - great value for weekly calls! For quick calls, try Pay As You Go 30 Secs at 50 RWF."

        }

    

    def get_airtime_answer(self) -> Dict[str, Any]:

        """Get airtime plans information"""

        plans = SERVICE_DATABASE["airtime_plans"]

        

        return {

            "type": "airtime",

            "question": "Airtime Plans",

            "answer": "Here are our airtime recharge options:",

            "plans": plans,

            "summary": f"We offer {len(plans)} airtime plans from {plans[0]['price']} RWF to {plans[-1]['price']} RWF, with bonuses on larger amounts.",

            "recommendation": "The Airtime 1000 RWF package gives you 1000 RWF airtime plus 150 RWF bonus - great value!"

        }

    

    def get_services_answer(self) -> Dict[str, Any]:

        """Get available services"""

        services = SERVICE_DATABASE["services"]

        

        return {

            "type": "services",

            "question": "Available Services",

            "answer": "Here are the services we offer:",

            "services": services,

            "summary": f"We provide {len(services)} main services including mobile banking, customer care, and international calling.",

            "recommendation": "Our mobile banking service is free and lets you manage everything from your phone!"

        }

    

    def get_prices_answer(self, original_text: str) -> Dict[str, Any]:

        """Get pricing information"""

        text_lower = original_text.lower()

        

        if "daily" in text_lower and "internet" in text_lower:

            return self.get_daily_bundles_answer()

        elif "weekly" in text_lower and "internet" in text_lower:

            return self.get_weekly_bundles_answer()

        elif "monthly" in text_lower and "internet" in text_lower:

            return self.get_monthly_bundles_answer()

        elif "calling" in text_lower or "minutes" in text_lower or "mins" in text_lower or "seconds" in text_lower or "secs" in text_lower:

            return self.get_calling_bundles_answer()

        elif "airtime" in text_lower:

            return self.get_airtime_answer()

        else:

            return {

                "type": "prices",

                "question": "Pricing Information",

                "answer": "Here's our pricing overview:",

                "summary": "Daily internet bundles start from 50 RWF, weekly from 500 RWF, monthly from 3000 RWF. Calling bundles from 300 RWF. Airtime from 100 RWF.",

                "recommendation": "Would you like specific pricing for daily, weekly, monthly internet bundles, calling bundles, or airtime?"

            }

    

    def get_recommendations_answer(self, original_text: str) -> Dict[str, Any]:

        """Get personalized recommendations"""

        text_lower = original_text.lower()

        

        if "light" in text_lower or "basic" in text_lower or "social media" in text_lower:

            return {

                "type": "recommendation",

                "question": "Bundle Recommendation",

                "answer": "For light usage and social media:",

                "recommendation": "I recommend the Daily 500MB bundle for 100 RWF - perfect for WhatsApp, Facebook, and light browsing!",

                "alternatives": ["Daily 100MB (50 RWF) for minimal use", "Weekly 1GB (500 RWF) for consistent light use"]

            }

        elif "heavy" in text_lower or "streaming" in text_lower or "video" in text_lower:

            return {

                "type": "recommendation",

                "question": "Bundle Recommendation",

                "answer": "For heavy usage and streaming:",

                "recommendation": "I recommend the Monthly 20GB bundle for 8000 RWF - great for HD streaming and downloads!",

                "alternatives": ["Daily 2GB (350 RWF) for occasional heavy use", "Weekly 5GB (2000 RWF) for weekly heavy use"]

            }

        elif "calling" in text_lower or "minutes" in text_lower or "talk" in text_lower:

            return {

                "type": "recommendation",

                "question": "Calling Bundle Recommendation",

                "answer": "For calling needs:",

                "recommendation": "I recommend the Weekly 200 Mins bundle for 1500 RWF - great value for regular weekly calls!",

                "alternatives": ["Daily 50 Mins (300 RWF) for light daily calling", "Pay As You Go 30 Secs (50 RWF) for quick calls", "Monthly 1000 Mins (8000 RWF) for heavy monthly use"]

            }

        elif "quick" in text_lower or "short" in text_lower or "emergency" in text_lower:

            return {

                "type": "recommendation",

                "question": "Quick Calling Recommendation",

                "answer": "For quick and emergency calls:",

                "recommendation": "I recommend Pay As You Go 30 Secs at 50 RWF - perfect for short calls!",

                "alternatives": ["Pay As You Go 60 Secs (100 RWF) for medium calls", "Daily 50 Mins (300 RWF) if you make multiple short calls"]

            }

        else:

            return {

                "type": "recommendation",

                "question": "Bundle Recommendation",

                "answer": "For general use:",

                "recommendation": "I recommend the Monthly 10GB bundle for 5000 RWF - it's our most popular and offers great value!",

                "alternatives": ["Daily 1GB (200 RWF) for trying out", "Weekly 3GB (1200 RWF) for weekly use"]

            }

    

    def get_general_answer(self) -> Dict[str, Any]:

        """Get general answer when intent is unclear"""

        return {

            "type": "general",

            "question": "How can I help you?",

            "answer": "I can help you with information about:",

            "options": [

                "Daily internet bundles",

                "Weekly internet bundles", 

                "Monthly internet bundles",

                "Calling bundles with minutes and seconds",

                "Airtime plans",

                "Available services",

                "Pricing information"

            ],

            "summary": "Just ask about any of our services and I'll provide detailed information!",

            "examples": [

                "What daily internet bundles can I buy?",

                "How much is the weekly 3GB bundle?",

                "Recommend a bundle for streaming",

                "What calling bundles do you have?",

                "How many minutes in the daily calling bundle?",

                "What are your pay as you go seconds?",

                "What airtime plans do you have?"

            ]

        }



# Initialize Q&A engine

qa_engine = SmartQnAEngine()



# Pydantic models for Smart Q&A

class SmartQuestionRequest(BaseModel):

    phone: str = Field(..., description="User phone number")

    audio: Optional[str] = Field(None, description="Base64 encoded audio data")

    text: Optional[str] = Field(None, description="Text question if no audio")

    model_size: str = Field(default="base", description="Whisper model size")



# SMART Q&A ENDPOINTS

@app.post("/smart-ask")

@limiter.limit("10/minute")

def smart_ask_question(request: SmartQuestionRequest, req: Request):

    """

    Smart Q&A endpoint - understands questions and provides real answers

    """

    try:

        # Get question text (either from audio transcription or direct text)

        if request.audio:

            # Process audio file (base64 decoded)

            import base64

            audio_data = base64.b64decode(request.audio)

            

            # Save temporary audio file

            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:

                temp_file.write(audio_data)

                temp_path = temp_file.name

            

            try:

                # Transcribe using existing Whisper model

                model = get_whisper_model(request.model_size)

                result = model.transcribe(temp_path)

                question_text = result["text"].strip()

                

                if not question_text:

                    return {

                        "error": "I couldn't hear your question. Please try speaking clearly! 🎤",

                        "status": "error"

                    }

                

            finally:

                # Clean up temporary file

                os.unlink(temp_path)

                

        elif request.text:

            question_text = request.text.strip()

        else:

            return {

                "error": "Please provide either audio or text question!",

                "status": "error"

            }

        

        # Analyze question and get answer

        qa_result = qa_engine.analyze_question(question_text)

        

        return {

            "phone": request.phone,

            "question": question_text,

            "intent": qa_result["intent"],

            "confidence": qa_result["confidence"],

            "answer": qa_result["answer"],

            "status": "success",

            "message": "✅ Question understood and answered!"

        }

        

    except Exception as e:

        logger.error(f"Smart Q&A error: {e}")

        return {

            "error": f"Oops! Something went wrong: {str(e)}",

            "status": "error"

        }



@app.post("/smart-ask-upload")

@limiter.limit("10/minute")

def smart_ask_upload(request: Request, phone: str = Form(...), model_size: str = Form(default="base"), audio: UploadFile = File(...)):

    """

    Smart Q&A endpoint with file upload - understands questions and provides real answers

    """

    try:

        # Validate audio file

        if not audio.content_type.startswith('audio/'):

            return {"error": "Please upload an audio file! 🎵", "status": "error"}

        

        # Save temporary file

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:

            content = audio.file.read()

            temp_file.write(content)

            temp_path = temp_file.name

        

        try:

            # Transcribe using existing Whisper model

            model = get_whisper_model(model_size)

            result = model.transcribe(temp_path)

            question_text = result["text"].strip()

            

            if not question_text:

                return {

                    "error": "I couldn't hear your question. Please try speaking clearly! 🎤",

                    "status": "error"

                }

            

            # Analyze question and get answer

            qa_result = qa_engine.analyze_question(question_text)

            

            return {

                "phone": phone,

                "question": question_text,

                "intent": qa_result["intent"],

                "confidence": qa_result["confidence"],

                "answer": qa_result["answer"],

                "status": "success",

                "message": "✅ Question understood and answered!"

            }

            

        finally:

            # Clean up

            os.unlink(temp_path)

            

    except Exception as e:

        logger.error(f"Smart Q&A upload error: {e}")

        return {

            "error": f"Oops! Something went wrong: {str(e)}",

            "status": "error"

        }



@app.get("/smart-health")

def smart_health():

    """Smart Q&A health check"""

    return {

        "status": "🤖 Smart Q&A ASR is ready!",

        "message": "I can understand questions and provide real answers about services!",

        "capabilities": [

            "Understands questions about bundles",

            "Provides real pricing information", 

            "Gives personalized recommendations",

            "Answers service inquiries",

            "Handles airtime questions",

            "Knows calling bundles with minutes and seconds"

        ],

        "supported_questions": [

            "Daily internet bundles (GB)",

            "Weekly data packages (GB)",

            "Monthly plans (GB)",

            "Calling bundles (min)",

            "Airtime recharge options (RWF)",

            "Available services",

            "Pricing information (RWF)",

            "Personalized recommendations"

        ],

        "units": {

            "data": "GB",

            "calling": "min",

            "money": "RWF"

        }

    }



# SMART Q&A HTML INTERFACE - Simplified Direct Q&A

SMART_HTML = """

<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>🤖 Smart Q&A - Ask Direct Questions</title>

    <style>

        body {

            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;

            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

            color: white;

            margin: 0;

            padding: 20px;

            min-height: 100vh;

        }

        .container {

            max-width: 800px;

            margin: 0 auto;

            background: rgba(255, 255, 255, 0.1);

            border-radius: 20px;

            padding: 30px;

            backdrop-filter: blur(10px);

            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);

        }

        h1 {

            text-align: center;

            font-size: 2.2em;

            margin-bottom: 10px;

            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);

        }

        .subtitle {

            text-align: center;

            font-size: 1.1em;

            margin-bottom: 30px;

            opacity: 0.9;

        }

        .quick-questions {

            background: rgba(255, 255, 255, 0.15);

            border-radius: 15px;

            padding: 20px;

            margin: 20px 0;

        }

        .question-btn {

            background: rgba(255, 255, 255, 0.2);

            border: 2px solid rgba(255, 255, 255, 0.3);

            color: white;

            padding: 15px 20px;

            border-radius: 10px;

            margin: 8px;

            cursor: pointer;

            transition: all 0.3s ease;

            font-size: 1em;

            text-align: left;

            width: 100%;

            box-sizing: border-box;

        }

        .question-btn:hover {

            background: rgba(255, 255, 255, 0.3);

            border-color: rgba(255, 255, 255, 0.6);

            transform: translateY(-2px);

        }

        .custom-question {

            background: rgba(255, 255, 255, 0.15);

            border-radius: 15px;

            padding: 20px;

            margin: 20px 0;

        }

        .upload-area {

            border: 3px dashed rgba(255, 255, 255, 0.3);

            border-radius: 15px;

            padding: 30px;

            text-align: center;

            margin: 20px 0;

            transition: all 0.3s ease;

            cursor: pointer;

        }

        .upload-area:hover {

            border-color: rgba(255, 255, 255, 0.6);

            background: rgba(255, 255, 255, 0.05);

        }

        .file-input {

            display: none;

        }

        .ask-btn {

            background: linear-gradient(45deg, #ff6b6b, #ee5a24);

            color: white;

            border: none;

            padding: 15px 30px;

            border-radius: 25px;

            font-size: 1.1em;

            cursor: pointer;

            transition: all 0.3s ease;

            margin: 10px;

        }

        .ask-btn:hover {

            transform: translateY(-2px);

            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);

        }

        .action-btn {

            background: linear-gradient(45deg, #4caf50, #45a049);

            color: white;

            border: none;

            padding: 10px 20px;

            border-radius: 20px;

            font-size: 0.9em;

            cursor: pointer;

            transition: all 0.3s ease;

            margin: 5px;

            font-weight: bold;

            min-width: 100px;

        }

        .action-btn:hover {

            transform: translateY(-2px);

            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);

            background: linear-gradient(45deg, #45a049, #4caf50);

        }

        @keyframes pulse {

            0% { opacity: 1; }

            50% { opacity: 0.7; }

            100% { opacity: 1; }

        }

        .result {

            background: rgba(255, 255, 255, 0.15);

            border-radius: 15px;

            padding: 20px;

            margin: 20px 0;

            display: none;

        }

        .question {

            font-size: 1.3em;

            font-weight: bold;

            color: #ff6b6b;

            margin-bottom: 15px;

        }

        .answer {

            font-size: 1.1em;

            line-height: 1.5;

            margin: 15px 0;

        }

        .quick-answer {

            background: rgba(76, 175, 80, 0.2);

            border-left: 4px solid #4caf50;

            padding: 15px;

            margin: 10px 0;

            border-radius: 5px;

        }

        .loading {

            display: none;

            text-align: center;

            margin: 20px 0;

        }

        .spinner {

            border: 4px solid rgba(255, 255, 255, 0.3);

            border-radius: 50%;

            border-top: 4px solid #ff6b6b;

            width: 40px;

            height: 40px;

            animation: spin 1s linear infinite;

            margin: 0 auto;

        }

        @keyframes spin {

            0% { transform: rotate(0deg); }

            100% { transform: rotate(360deg); }

        }

        .emoji {

            font-size: 1.5em;

            margin-right: 10px;

        }

    </style>

</head>

<body>

    <div class="container">

        <h1>🤖 Smart Q&A</h1>

        <p class="subtitle">Ask about bundles, calling, and airtime - Get instant answers!</p>

        

        <div class="quick-questions">

            <h3>🎯 Quick Questions (Click to Ask)</h3>

            <button class="question-btn" onclick="askQuestion('What daily internet bundles can I buy?')">

                <span class="emoji">📱</span> What daily internet bundles can I buy?

            </button>

            <button class="question-btn" onclick="askQuestion('How much is the weekly 3GB bundle?')">

                <span class="emoji">📊</span> How much is the weekly 3GB bundle?

            </button>

            <button class="question-btn" onclick="askQuestion('Recommend a bundle for streaming')">

                <span class="emoji">💡</span> Recommend a bundle for streaming

            </button>

            <button class="question-btn" onclick="askQuestion('What calling bundles do you have?')">

                <span class="emoji">📞</span> What calling bundles do you have?

            </button>

            <button class="question-btn" onclick="askQuestion('How many min in the daily calling bundle?')">

                <span class="emoji">⏰</span> How many min in the daily calling bundle?

            </button>

            <button class="question-btn" onclick="askQuestion('What are your pay as you go seconds?')">

                <span class="emoji">⚡</span> What are your pay as you go seconds?

            </button>

            <button class="question-btn" onclick="askQuestion('What airtime plans do you have?')">

                <span class="emoji">💰</span> What airtime plans do you have?

            </button>

        </div>

        

        <div class="custom-question">

            <h3>🎤 Ask Your Own Question</h3>

            <div class="upload-area" id="uploadArea">

                <div class="emoji">🎤</div>

                <h3>Record or upload your question</h3>

                <p>I'll understand and give you a direct answer!</p>

                <input type="file" id="fileInput" class="file-input" accept="audio/*">

                <button class="ask-btn" onclick="document.getElementById('fileInput').click()">

                    🎤 Upload & Ask

                </button>

            </div>

        </div>

        

        <div class="loading" id="loading">

            <div class="spinner"></div>

            <p>🤔 Getting your answer...</p>

        </div>

        

        <div class="result" id="result">

            <div class="question" id="questionText"></div>

            <div class="answer" id="answerText"></div>

            <div id="detailsArea"></div>

        </div>

    </div>

    

    <script>

        const uploadArea = document.getElementById('uploadArea');

        const fileInput = document.getElementById('fileInput');

        const loading = document.getElementById('loading');

        const result = document.getElementById('result');

        const questionText = document.getElementById('questionText');

        const answerText = document.getElementById('answerText');

        const detailsArea = document.getElementById('detailsArea');

        

        uploadArea.addEventListener('click', () => {

            fileInput.click();

        });

        

        uploadArea.addEventListener('dragover', (e) => {

            e.preventDefault();

            uploadArea.style.background = 'rgba(255, 255, 255, 0.05)';

        });

        

        uploadArea.addEventListener('dragleave', () => {

            uploadArea.style.background = 'rgba(255, 255, 255, 0.0)';

        });

        

        uploadArea.addEventListener('drop', (e) => {

            e.preventDefault();

            uploadArea.style.background = 'rgba(255, 255, 255, 0.0)';

            

            const files = e.dataTransfer.files;

            if (files.length > 0) {

                handleFile(files[0]);

            }

        });

        

        fileInput.addEventListener('change', (e) => {

            if (e.target.files.length > 0) {

                handleFile(e.target.files[0]);

            }

        });

        

        // Quick question buttons

        function askQuestion(question) {

            questionText.textContent = `📝 Question: "${question}"`;

            answerText.innerHTML = '<div class="quick-answer">🤖 Processing your question...</div>';

            result.style.display = 'block';

            

            // Simulate asking the question via API

            fetch('/smart-ask', {

                method: 'POST',

                headers: {

                    'Content-Type': 'application/json',

                },

                body: JSON.stringify({

                    phone: '0783524048',

                    text: question,

                    model_size: 'base'

                })

            })

            .then(response => response.json())

            .then(data => {

                if (data.status === 'error') {

                    answerText.innerHTML = `<div class="quick-answer">❌ ${data.error}</div>`;

                } else {

                    displaySimpleAnswer(data);

                }

            })

            .catch(error => {

                answerText.innerHTML = '<div class="quick-answer">❌ Something went wrong. Please try again!</div>';

            });

        }

        

        async function handleFile(file) {

            if (!file.type.startsWith('audio/')) {

                alert('Please upload an audio file! 🎵');

                return;

            }

            

            loading.style.display = 'block';

            result.style.display = 'none';

            

            const formData = new FormData();

            formData.append('phone', '0783524048');

            formData.append('audio', file);

            

            try {

                const response = await fetch('/smart-ask-upload', {

                    method: 'POST',

                    body: formData

                });

                

                const data = await response.json();

                

                loading.style.display = 'none';

                

                if (data.status === 'error') {

                    questionText.textContent = '❌ Error';

                    answerText.textContent = data.error;

                    detailsArea.innerHTML = '';

                } else {

                    displaySimpleAnswer(data);

                }

                

                result.style.display = 'block';

                

            } catch (error) {

                loading.style.display = 'none';

                questionText.textContent = '❌ Error';

                answerText.textContent = 'Something went wrong. Please try again!';

                detailsArea.innerHTML = '';

                result.style.display = 'block';

            }

        }

        

        function displaySimpleAnswer(data) {

            const answer = data.answer;

            

            questionText.textContent = `📝 You asked: "${data.question}"`;

            answerText.innerHTML = `<div class="quick-answer">${answer.answer}</div>`;

            

            let detailsHtml = '';

            

            // Show bundles in a simple, direct format with action buttons

            if (answer.type === 'daily_bundles' || answer.type === 'weekly_bundles' || answer.type === 'monthly_bundles') {

                const bundles = answer.bundles || [];

                detailsHtml = '<div style="margin: 15px 0;"><h4>📊 Available Bundles:</h4>';

                bundles.forEach(bundle => {

                    detailsHtml += `

                        <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; margin: 8px 0; border-radius: 8px;">

                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">

                                <div style="flex: 1;">

                                    <strong>${bundle.name}</strong><br>

                                    📊 ${bundle.data} | ⏰ ${bundle.validity}<br>

                                    <span style="color: #4caf50; font-size: 1.2em; font-weight: bold;">${bundle.price} RWF</span>

                                </div>

                                <div style="margin-left: 15px;">

                                    <button class="action-btn" onclick="buyBundle('${bundle.name}', ${bundle.price}, '${bundle.data}')">

                                        🛒 Buy Now

                                    </button>

                                </div>

                            </div>

                        </div>

                    `;

                });

                detailsHtml += '</div>';

            }

            

            if (answer.type === 'calling_bundles') {

                const bundles = answer.bundles || [];

                detailsHtml = '<div style="margin: 15px 0;"><h4>📞 Calling Bundles:</h4>';

                bundles.forEach(bundle => {

                    const timeUnit = bundle.minutes || bundle.seconds;

                    detailsHtml += `

                        <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; margin: 8px 0; border-radius: 8px;">

                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">

                                <div style="flex: 1;">

                                    <strong>${bundle.name}</strong><br>

                                    ⏰ ${timeUnit} | 📅 ${bundle.validity}<br>

                                    <span style="color: #4caf50; font-size: 1.2em; font-weight: bold;">${bundle.price} RWF</span>

                                </div>

                                <div style="margin-left: 15px;">

                                    <button class="action-btn" onclick="buyBundle('${bundle.name}', ${bundle.price}, '${timeUnit}')">

                                        📞 Activate

                                    </button>

                                </div>

                            </div>

                        </div>

                    `;

                });

                detailsHtml += '</div>';

            }

            

            if (answer.type === 'airtime') {

                const plans = answer.plans || [];

                detailsHtml = '<div style="margin: 15px 0;"><h4>💰 Airtime Plans:</h4>';

                plans.forEach(plan => {

                    detailsHtml += `

                        <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; margin: 8px 0; border-radius: 8px;">

                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">

                                <div style="flex: 1;">

                                    <strong>${plan.name}</strong><br>

                                    💵 ${plan.airtime} airtime ${plan.bonus > 0 ? `+ 🎁 ${plan.bonus} bonus` : ''}<br>

                                    <span style="color: #4caf50; font-size: 1.2em; font-weight: bold;">${plan.price} RWF</span>

                                </div>

                                <div style="margin-left: 15px;">

                                    <button class="action-btn" onclick="rechargeAirtime('${plan.name}', ${plan.price}, '${plan.airtime}')">

                                        💳 Recharge

                                    </button>

                                </div>

                            </div>

                        </div>

                    `;

                });

                detailsHtml += '</div>';

            }

            

            if (answer.recommendation) {

                detailsHtml += `

                    <div class="quick-answer">

                        <h4>💡 Recommendation:</h4>

                        <p>${answer.recommendation}</p>

                    </div>

                `;

            }

            

            if (answer.summary) {

                detailsHtml += `

                    <div style="margin: 15px 0; padding: 15px; background: rgba(255, 255, 255, 0.05); border-radius: 10px;">

                        <strong>📊 Summary:</strong> ${answer.summary}

                    </div>

                `;

            }

            

            detailsArea.innerHTML = detailsHtml;

        }

        

        // Action functions for real operations

        function buyBundle(bundleName, price, dataAmount) {

            if (confirm(`🛒 Buy ${bundleName} for ${price} RWF?\\n\\nYou will get ${dataAmount} of data.`)) {

                // Simulate purchase

                showProcessing(`🛒 Purchasing ${bundleName}...`);

                

                setTimeout(() => {

                    showSuccess(`✅ Success!\\n\\n${bundleName} has been activated.\\n💰 ${price} RWF deducted from your account.\\n📊 ${dataAmount} data added.`);

                }, 2000);

            }

        }

        

        function rechargeAirtime(planName, price, airtimeAmount) {

            if (confirm(`💳 Recharge ${planName} for ${price} RWF?\\n\\nYou will get ${airtimeAmount} airtime.`)) {

                // Simulate recharge

                showProcessing(`💳 Processing ${planName}...`);

                

                setTimeout(() => {

                    showSuccess(`✅ Success!\\n\\n${planName} recharge completed.\\n💰 ${price} RWF deducted from your account.\\n💵 ${airtimeAmount} airtime added to your balance.`);

                }, 2000);

            }

        }

        

        function showProcessing(message) {

            detailsArea.innerHTML += `

                <div id="processing" style="background: rgba(255, 193, 7, 0.2); border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 5px; animation: pulse 1.5s infinite;">

                    <h4>⏳ Processing...</h4>

                    <p>${message}</p>

                </div>

            `;

        }

        

        function showSuccess(message) {

            const processingDiv = document.getElementById('processing');

            if (processingDiv) {

                processingDiv.remove();

            }

            

            detailsArea.innerHTML += `

                <div style="background: rgba(76, 175, 80, 0.2); border-left: 4px solid #4caf50; padding: 15px; margin: 10px 0; border-radius: 5px;">

                    <h4>✅ Success!</h4>

                    <p>${message}</p>

                    <button class="action-btn" onclick="location.reload()" style="margin-top: 10px;">

                        🔄 Ask Another Question

                    </button>

                </div>

            `;

        }

    </script>

</body>

</html>

"""


@app.get("/smart-qa", response_class=HTMLResponse)

def smart_qa_home():

    """Smart Q&A home page"""

    return SMART_HTML