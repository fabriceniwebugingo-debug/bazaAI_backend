#!/usr/bin/env python3
"""
Intelligent Friendly ASR - Understands User Intent & Provides Many Options
Super smart system that understands what users really want and offers comprehensive options
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
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
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import shutil
import tempfile
import time
import hashlib
import random
import base64

# ASR imports
import whisper
import speech_recognition as sr

# TTS imports
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    pyttsx3 = None

# ML imports
try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    torchaudio = None

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="🧠 Intelligent Friendly ASR",
    description="Smart system that understands user intent and provides many helpful options",
    version="4.0.0"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database config
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "intelligent_asr"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "port": os.getenv("DB_PORT", "5432")
}

class IntelligentIntentAnalyzer:
    """Analyzes user intent and provides comprehensive options"""
    
    def __init__(self):
        self.intent_patterns = {
            # Balance/Account related
            "balance": {
                "keywords": ["balance", "account", "money", "amount", "how much", "remaining", "left"],
                "patterns": [r"\bbalance\b", r"\baccount\b", r"\bmoney\b", r"\bhow much\b"],
                "options": ["check_balance", "account_statement", "transaction_history", "mini_statement"]
            },
            
            # Airtime/Data related
            "airtime": {
                "keywords": ["airtime", "data", "bundle", "recharge", "top up", "credit", "minutes"],
                "patterns": [r"\bairtime\b", r"\bdata\b", r"\bbundle\b", r"\brecharge\b"],
                "options": ["buy_airtime", "buy_data", "special_offers", "auto_recharge", "data_calculator"]
            },
            
            # Transfer related
            "transfer": {
                "keywords": ["transfer", "send", "money", "payment", "pay", "give"],
                "patterns": [r"\btransfer\b", r"\bsend\b", r"\bpay\b", r"\bgive\b"],
                "options": ["send_money", "bill_payment", "international_transfer", "scheduled_transfer", "quick_send"]
            },
            
            # Help/Support related
            "help": {
                "keywords": ["help", "support", "problem", "issue", "trouble", "can't", "unable"],
                "patterns": [r"\bhelp\b", r"\bsupport\b", r"\bproblem\b", r"\btrouble\b"],
                "options": ["live_chat", "faq", "tutorial", "contact_agent", "self_service", "emergency_help"]
            },
            
            # Services/Features related
            "services": {
                "keywords": ["service", "feature", "activate", "deactivate", "subscribe", "unsubscribe"],
                "patterns": [r"\bservice\b", r"\bfeature\b", r"\bactivate\b", r"\bsubscribe\b"],
                "options": ["browse_services", "manage_subscriptions", "new_offers", "service_settings", "recommendations"]
            },
            
            # General greeting
            "greeting": {
                "keywords": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
                "patterns": [r"\bhello\b", r"\bhi\b", r"\bhey\b"],
                "options": ["main_menu", "quick_actions", "account_overview", "recent_activity", "personalized_suggestions"]
            }
        }
        
        self.response_templates = {
            "balance": {
                "primary": "I can help you check your account balance! 💰",
                "options": [
                    "Check current balance",
                    "View account statement", 
                    "See transaction history",
                    "Get mini statement",
                    "Check data usage"
                ]
            },
            "airtime": {
                "primary": "I can help you with airtime and data! 📱",
                "options": [
                    "Buy airtime",
                    "Purchase data bundle",
                    "View special offers",
                    "Set up auto-recharge",
                    "Calculate data needs"
                ]
            },
            "transfer": {
                "primary": "I can help you send money! 💸",
                "options": [
                    "Send money to person",
                    "Pay bills",
                    "International transfer",
                    "Scheduled transfer",
                    "Quick send options"
                ]
            },
            "help": {
                "primary": "I'm here to help you! How can I assist? 🤝",
                "options": [
                    "Live chat with agent",
                    "Browse FAQ",
                    "Watch tutorials",
                    "Self-service options",
                    "Emergency support"
                ]
            },
            "services": {
                "primary": "I can help you manage your services! ⚙️",
                "options": [
                    "Browse all services",
                    "Manage subscriptions",
                    "View new offers",
                    "Service settings",
                    "Personalized recommendations"
                ]
            },
            "greeting": {
                "primary": "Hello! How can I help you today? 😊",
                "options": [
                    "Main menu",
                    "Quick actions",
                    "Account overview",
                    "Recent activity",
                    "Personalized suggestions"
                ]
            }
        }
    
    def analyze_intent(self, text: str) -> Dict[str, Any]:
        """Analyze user intent from transcribed text"""
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
                "intent": "unknown",
                "confidence": 0.0,
                "text": text,
                "suggestions": self.get_general_suggestions()
            }
        
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = min(1.0, intent_scores[best_intent] / 10.0)
        
        return {
            "intent": best_intent,
            "confidence": confidence,
            "text": text,
            "suggestions": self.get_intent_suggestions(best_intent),
            "response_template": self.response_templates.get(best_intent, {})
        }
    
    def get_intent_suggestions(self, intent: str) -> List[Dict]:
        """Get comprehensive suggestions for specific intent"""
        intent_data = self.intent_patterns.get(intent, {})
        options = intent_data.get("options", [])
        
        suggestions = []
        for i, option in enumerate(options):
            suggestions.append({
                "id": f"{intent}_{i}",
                "title": option.replace("_", " ").title(),
                "description": self.get_option_description(option),
                "action": option,
                "icon": self.get_option_icon(option)
            })
        
        return suggestions
    
    def get_general_suggestions(self) -> List[Dict]:
        """Get suggestions when intent is unclear"""
        return [
            {
                "id": "general_0",
                "title": "Check Balance",
                "description": "View your current account balance and usage",
                "action": "check_balance",
                "icon": "💰"
            },
            {
                "id": "general_1", 
                "title": "Buy Airtime/Data",
                "description": "Purchase airtime or data bundles",
                "action": "buy_airtime",
                "icon": "📱"
            },
            {
                "id": "general_2",
                "title": "Send Money",
                "description": "Transfer money to other accounts",
                "action": "send_money",
                "icon": "💸"
            },
            {
                "id": "general_3",
                "title": "Get Help",
                "description": "Contact support or browse FAQ",
                "action": "get_help",
                "icon": "🤝"
            },
            {
                "id": "general_4",
                "title": "Browse Services",
                "description": "Explore available services and features",
                "action": "browse_services",
                "icon": "⚙️"
            }
        ]
    
    def get_option_description(self, option: str) -> str:
        """Get detailed description for an option"""
        descriptions = {
            "check_balance": "View your current account balance, available credit, and recent usage",
            "account_statement": "Download detailed account statement for the last 30 days",
            "transaction_history": "See all your recent transactions and payments",
            "mini_statement": "Get a quick summary of your account activity",
            "buy_airtime": "Recharge your account with airtime instantly",
            "buy_data": "Purchase data bundles for internet access",
            "special_offers": "View current promotions and special deals",
            "auto_recharge": "Set up automatic airtime recharge when balance is low",
            "data_calculator": "Calculate how much data you need based on usage",
            "send_money": "Transfer money to other users or pay bills",
            "bill_payment": "Pay your utility bills and other payments",
            "international_transfer": "Send money internationally with competitive rates",
            "scheduled_transfer": "Schedule recurring payments or transfers",
            "quick_send": "Quick transfer options for frequent recipients",
            "live_chat": "Chat with a customer service representative",
            "faq": "Browse frequently asked questions and answers",
            "tutorial": "Watch video tutorials for common tasks",
            "contact_agent": "Connect with a human agent for assistance",
            "self_service": "Access self-service options and tools",
            "emergency_help": "Get immediate help for urgent issues",
            "browse_services": "Explore all available services and features",
            "manage_subscriptions": "View and manage your active subscriptions",
            "new_offers": "Check out new services and promotional offers",
            "service_settings": "Customize your service preferences",
            "recommendations": "Get personalized service recommendations",
            "main_menu": "Return to main menu with all options",
            "quick_actions": "Access frequently used actions quickly",
            "account_overview": "See summary of your account status",
            "recent_activity": "View your recent account activities",
            "personalized_suggestions": "Get AI-powered suggestions based on your usage"
        }
        return descriptions.get(option, "Learn more about this option")
    
    def get_option_icon(self, option: str) -> str:
        """Get appropriate icon for an option"""
        icons = {
            "check_balance": "💰",
            "account_statement": "📄",
            "transaction_history": "📊",
            "mini_statement": "📋",
            "buy_airtime": "📱",
            "buy_data": "📶",
            "special_offers": "🎁",
            "auto_recharge": "🔄",
            "data_calculator": "🧮",
            "send_money": "💸",
            "bill_payment": "🧾",
            "international_transfer": "🌍",
            "scheduled_transfer": "📅",
            "quick_send": "⚡",
            "live_chat": "💬",
            "faq": "❓",
            "tutorial": "🎥",
            "contact_agent": "👥",
            "self_service": "🛠️",
            "emergency_help": "🚨",
            "browse_services": "⚙️",
            "manage_subscriptions": "📦",
            "new_offers": "🆕",
            "service_settings": "🔧",
            "recommendations": "✨",
            "main_menu": "🏠",
            "quick_actions": "⚡",
            "account_overview": "📊",
            "recent_activity": "🕒",
            "personalized_suggestions": "🤖"
        }
        return icons.get(option, "📋")

class IntelligentASRManager:
    """Intelligent ASR manager with intent analysis"""
    
    def __init__(self):
        self.whisper_models = {}
        self.speech_recognizer = sr.Recognizer() if sr else None
        self.tts_engine = None
        self.intent_analyzer = IntelligentIntentAnalyzer()
        self.setup_tts()
        self.conversation_history = []
    
    def setup_tts(self):
        """Setup intelligent TTS engine"""
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                voices = self.tts_engine.getProperty('voices')
                
                # Choose intelligent voice (clear and friendly)
                for voice in voices:
                    if any(keyword in voice.name.lower() for keyword in ['david', 'karen', 'samantha', 'zira']):
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                
                # Intelligent speech settings
                self.tts_engine.setProperty('rate', 145)  # Optimal comprehension rate
                self.tts_engine.setProperty('volume', 0.9)  # Clear but comfortable
                
                logger.info("🧠 Intelligent TTS engine initialized!")
            except Exception as e:
                logger.error(f"Failed to initialize TTS: {e}")
                self.tts_engine = None
    
    def get_whisper_model(self, model_size="base"):
        """Get Whisper model with caching"""
        if model_size not in self.whisper_models:
            try:
                logger.info(f"🤖 Loading intelligent Whisper model: {model_size}")
                self.whisper_models[model_size] = whisper.load_model(model_size)
                logger.info(f"✅ Whisper model {model_size} loaded successfully!")
            except Exception as e:
                logger.error(f"Failed to load Whisper model {model_size}: {e}")
                raise
        return self.whisper_models[model_size]
    
    def transcribe_and_analyze(self, audio_path: str, model_size="base") -> Dict[str, Any]:
        """Transcribe audio and analyze user intent"""
        try:
            model = self.get_whisper_model(model_size)
            
            # Use intelligent prompts for better business context
            business_prompts = [
                "Professional business communication",
                "Customer service interaction", 
                "Financial services conversation",
                "Technical support discussion",
                "Account management discussion"
            ]
            
            result = model.transcribe(
                audio_path,
                language="en",
                initial_prompt=random.choice(business_prompts),
                temperature=0.0,  # Consistent for intent analysis
                best_of=1,
                beam_size=1
            )
            
            transcribed_text = result["text"].strip()
            
            # Analyze intent
            intent_analysis = self.intent_analyzer.analyze_intent(transcribed_text)
            
            # Store in conversation history
            self.conversation_history.append({
                "timestamp": datetime.now(),
                "text": transcribed_text,
                "intent": intent_analysis["intent"],
                "confidence": intent_analysis["confidence"]
            })
            
            return {
                "text": transcribed_text,
                "confidence": max(0.0, result.get("avg_logprob", 0.0) + 1.0),
                "model_used": f"whisper-{model_size}",
                "language": "en",
                "processing_time": time.time(),
                "intent_analysis": intent_analysis,
                "conversation_id": len(self.conversation_history)
            }
            
        except Exception as e:
            logger.error(f"Intelligent transcription failed: {e}")
            return {"error": str(e)}
    
    def get_intelligent_response(self, intent_analysis: Dict, user_context: Dict = None) -> Dict[str, Any]:
        """Generate intelligent response based on intent and context"""
        intent = intent_analysis.get("intent", "unknown")
        confidence = intent_analysis.get("confidence", 0.0)
        suggestions = intent_analysis.get("suggestions", [])
        response_template = intent_analysis.get("response_template", {})
        
        # Generate intelligent response
        response = {
            "primary_message": response_template.get("primary", "I'm here to help you! How can I assist? 😊"),
            "intent": intent,
            "confidence": confidence,
            "suggestions": suggestions,
            "contextual_options": self.get_contextual_options(intent, user_context),
            "follow_up_questions": self.get_follow_up_questions(intent),
            "helpful_tips": self.get_helpful_tips(intent),
            "quick_actions": self.get_quick_actions(intent)
        }
        
        return response
    
    def get_contextual_options(self, intent: str, user_context: Dict) -> List[Dict]:
        """Get options based on user context and history"""
        base_options = self.intent_analyzer.get_intent_suggestions(intent)
        
        # Add contextual intelligence
        if user_context:
            # Time-based suggestions
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 17:  # Business hours
                base_options.append({
                    "id": "context_business",
                    "title": "Business Hours Support",
                    "description": "Connect with business support team",
                    "action": "business_support",
                    "icon": "💼"
                })
            
            # Usage-based suggestions
            if len(self.conversation_history) > 3:
                recent_intents = [h["intent"] for h in self.conversation_history[-3:]]
                if "balance" in recent_intents:
                    base_options.append({
                        "id": "context_balance_alert",
                        "title": "Set Balance Alerts",
                        "description": "Get notified when balance is low",
                        "action": "set_alerts",
                        "icon": "🔔"
                    })
        
        return base_options
    
    def get_follow_up_questions(self, intent: str) -> List[str]:
        """Get intelligent follow-up questions"""
        questions = {
            "balance": [
                "Would you like to set up low balance alerts?",
                "Do you need a detailed transaction history?",
                "Would you like to see your data usage breakdown?"
            ],
            "airtime": [
                "What amount of airtime do you need?",
                "Would you like to see current data offers?",
                "Do you want to set up auto-recharge?",
                "Would you like me to calculate your data needs?"
            ],
            "transfer": [
                "Who would you like to send money to?",
                "Is this a one-time or recurring transfer?",
                "Would you like to see exchange rates?",
                "Do you need to pay any bills?"
            ],
            "help": [
                "What specific issue are you experiencing?",
                "Would you prefer live chat or self-service?",
                "Is this urgent or can it wait?",
                "Have you tried our troubleshooting steps?"
            ],
            "services": [
                "Are you looking for personal or business services?",
                "Would you like to see current promotions?",
                "Do you need to manage existing subscriptions?",
                "Would you like personalized recommendations?"
            ]
        }
        
        return questions.get(intent, [
            "How else can I help you today?",
            "Is there anything specific you'd like to know?",
            "Would you like more information about any of these options?"
        ])
    
    def get_helpful_tips(self, intent: str) -> List[str]:
        """Get helpful tips based on intent"""
        tips = {
            "balance": [
                "💡 Tip: You can check your balance anytime by dialing *123#",
                "💡 Tip: Set up SMS alerts for low balance",
                "💡 Tip: Use our app to track your spending"
            ],
            "airtime": [
                "💡 Tip: Data bundles offer better value than pay-as-you-go",
                "💡 Tip: Auto-recharge ensures you never run out of data",
                "💡 Tip: Check our special offers before purchasing"
            ],
            "transfer": [
                "💡 Tip: Save frequent recipients for quick transfers",
                "💡 Tip: Use scheduled transfers for regular payments",
                "💡 Tip: International transfers have better rates on weekdays"
            ],
            "help": [
                "💡 Tip: Our FAQ answers 80% of common questions",
                "💡 Tip: Live chat is fastest way to get help",
                "💡 Tip: Restart your device to fix common issues"
            ],
            "services": [
                "💡 Tip: Bundle services for better value",
                "💡 Tip: Check personalized recommendations regularly",
                "💡 Tip: Some services offer free trial periods"
            ]
        }
        
        return tips.get(intent, [
            "💡 Tip: Explore all our services for maximum benefits",
            "💡 Tip: Contact us anytime for personalized assistance",
            "💡 Tip: Use our mobile app for convenient management"
        ])
    
    def get_quick_actions(self, intent: str) -> List[Dict]:
        """Get quick action buttons for common tasks"""
        actions = {
            "balance": [
                {"id": "quick_balance", "title": "Check Balance", "icon": "💰"},
                {"id": "quick_data", "title": "Data Usage", "icon": "📶"}
            ],
            "airtime": [
                {"id": "quick_recharge", "title": "Quick Recharge", "icon": "⚡"},
                {"id": "quick_bundles", "title": "Popular Bundles", "icon": "📦"}
            ],
            "transfer": [
                {"id": "quick_recent", "title": "Recent Recipients", "icon": "👥"},
                {"id": "quick_schedule", "title": "Schedule Transfer", "icon": "📅"}
            ],
            "help": [
                {"id": "quick_chat", "title": "Live Chat", "icon": "💬"},
                {"id": "quick_faq", "title": "FAQ", "icon": "❓"}
            ]
        }
        
        return actions.get(intent, [
            {"id": "quick_menu", "title": "Main Menu", "icon": "🏠"},
            {"id": "quick_search", "title": "Search Help", "icon": "🔍"}
        ])

# Initialize intelligent ASR manager
asr_manager = IntelligentASRManager()

# Database connection
def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

# Pydantic models
class IntelligentTranscriptionRequest(BaseModel):
    phone: str = Field(..., description="User phone number")
    model_size: str = Field(default="base", description="Model size")
    enable_tts: bool = Field(default=True, description="Enable intelligent TTS")
    user_context: Optional[Dict] = Field(default=None, description="User context for personalization")

class IntelligentResponse(BaseModel):
    intent: str
    confidence: float
    primary_message: str
    suggestions: List[Dict]
    contextual_options: List[Dict]
    follow_up_questions: List[str]
    helpful_tips: List[str]
    quick_actions: List[Dict]

# Intelligent HTML Interface
INTELLIGENT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Intelligent Friendly ASR</title>
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
            max-width: 900px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            margin-bottom: 30px;
            opacity: 0.9;
        }
        .section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
        }
        .button {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            margin: 5px;
        }
        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        .input {
            width: 100%;
            padding: 15px;
            border-radius: 10px;
            border: none;
            font-size: 1em;
            margin: 10px 0;
            background: rgba(255, 255, 255, 0.9);
        }
        .suggestions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .suggestion-card {
            background: rgba(255, 255, 255, 0.15);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .suggestion-card:hover {
            background: rgba(255, 255, 255, 0.25);
            transform: translateY(-2px);
        }
        .suggestion-icon {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .suggestion-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .suggestion-description {
            opacity: 0.9;
            line-height: 1.4;
        }
        .response-area {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            border-left: 5px solid #ff6b6b;
        }
        .primary-message {
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .follow-up {
            margin-top: 15px;
        }
        .follow-up h4 {
            margin-bottom: 10px;
            color: #ff6b6b;
        }
        .tips {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
        }
        .tips h4 {
            margin-bottom: 10px;
        }
        .tip {
            margin: 5px 0;
            padding: 5px 0;
        }
        .quick-actions {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin: 15px 0;
        }
        .quick-action {
            background: rgba(255, 255, 255, 0.15);
            padding: 10px 15px;
            border-radius: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .quick-action:hover {
            background: rgba(255, 255, 255, 0.25);
        }
        .emoji {
            font-size: 1.5em;
            margin-right: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Intelligent Friendly ASR</h1>
        <p class="subtitle">Smart system that understands what you really want and provides many helpful options! 🤖</p>
        
        <div class="section">
            <h2>🎙️ Speech Recognition & Intent Analysis</h2>
            <p>Upload your audio and I'll understand what you need!</p>
            <form id="transcribeForm" enctype="multipart/form-data">
                <input type="text" name="phone" placeholder="Your phone number" class="input" required>
                <select name="model_size" class="input">
                    <option value="tiny">Tiny (Fastest) 🚀</option>
                    <option value="base" selected>Base (Intelligent) 🧠</option>
                    <option value="small">Small (Accurate) ✨</option>
                    <option value="medium">Medium (Professional) 💼</option>
                </select>
                <label>
                    <input type="checkbox" name="enable_tts" checked> Enable intelligent TTS 🔊
                </label>
                <input type="file" name="audio" accept="audio/*" class="input" required>
                <button type="submit" class="button">🧠 Analyze & Understand</button>
            </form>
        </div>
        
        <div class="section" id="responseArea" style="display: none;">
            <div class="response-area">
                <div class="primary-message" id="primaryMessage"></div>
                <div id="intentInfo"></div>
            </div>
            
            <div class="suggestions" id="suggestionsContainer"></div>
            
            <div class="follow-up" id="followUpContainer"></div>
            
            <div class="tips" id="tipsContainer"></div>
            
            <div class="quick-actions" id="quickActionsContainer"></div>
        </div>
    </div>
    
    <script>
        document.getElementById('transcribeForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            
            document.getElementById('responseArea').style.display = 'block';
            document.getElementById('responseArea').innerHTML = '<p>🧠 Analyzing your speech and understanding your needs... ⏳</p>';
            
            try {
                const response = await fetch('/intelligent-transcribe', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.error) {
                    document.getElementById('responseArea').innerHTML = '<p>❌ ' + result.error + '</p>';
                    return;
                }
                
                // Display intelligent response
                displayIntelligentResponse(result);
                
            } catch (error) {
                document.getElementById('responseArea').innerHTML = '<p>❌ Oops! Something went wrong. Please try again!</p>';
            }
        });
        
        function displayIntelligentResponse(result) {
            const response = result.intelligent_response;
            
            // Primary message
            document.getElementById('primaryMessage').innerHTML = response.primary_message;
            
            // Intent info
            const intentInfo = `
                <h3>🎯 Understood Intent: ${response.intent} (Confidence: ${(response.confidence * 100).toFixed(1)}%)</h3>
                <p>I believe you want to: ${response.intent.replace('_', ' ').toUpperCase()}</p>
            `;
            document.getElementById('intentInfo').innerHTML = intentInfo;
            
            // Suggestions
            const suggestionsHtml = response.suggestions.map(suggestion => `
                <div class="suggestion-card" onclick="selectOption('${suggestion.action}')">
                    <div class="suggestion-icon">${suggestion.icon}</div>
                    <div class="suggestion-title">${suggestion.title}</div>
                    <div class="suggestion-description">${suggestion.description}</div>
                </div>
            `).join('');
            
            document.getElementById('suggestionsContainer').innerHTML = `
                <h3>💡 Here are your options:</h3>
                <div class="suggestions">${suggestionsHtml}</div>
            `;
            
            // Follow-up questions
            if (response.follow_up_questions && response.follow_up_questions.length > 0) {
                const followUpHtml = response.follow_up_questions.map(q => `<p>❓ ${q}</p>`).join('');
                document.getElementById('followUpContainer').innerHTML = `
                    <h4>🤔 Follow-up Questions:</h4>
                    <div class="follow-up">${followUpHtml}</div>
                `;
            }
            
            // Helpful tips
            if (response.helpful_tips && response.helpful_tips.length > 0) {
                const tipsHtml = response.helpful_tips.map(tip => `<div class="tip">${tip}</div>`).join('');
                document.getElementById('tipsContainer').innerHTML = `
                    <h4>💡 Helpful Tips:</h4>
                    <div class="tips">${tipsHtml}</div>
                `;
            }
            
            // Quick actions
            if (response.quick_actions && response.quick_actions.length > 0) {
                const actionsHtml = response.quick_actions.map(action => `
                    <div class="quick-action" onclick="selectOption('${action.id}')">
                        <span class="emoji">${action.icon}</span>
                        ${action.title}
                    </div>
                `).join('');
                
                document.getElementById('quickActionsContainer').innerHTML = `
                    <h4>⚡ Quick Actions:</h4>
                    <div class="quick-actions">${actionsHtml}</div>
                `;
            }
        }
        
        function selectOption(action) {
            alert('You selected: ' + action + '\\n\\nIn a real system, this would execute the action!');
        }
    </script>
</body>
</html>
"""

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def intelligent_home():
    """Intelligent home page with enhanced interface"""
    return INTELLIGENT_HTML

@app.post("/intelligent-transcribe")
@limiter.limit("10/minute")
async def intelligent_transcribe(
    request: Request,
    phone: str = Form(...),
    model_size: str = Form(default="base"),
    enable_tts: bool = Form(default=True),
    user_context: Optional[str] = Form(default=None),
    audio: UploadFile = File(...)
):
    """
    Intelligent transcription with intent analysis and comprehensive options
    """
    try:
        # Validate audio file
        if not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Please upload an audio file 🎵")
        
        # Save temporary audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            shutil.copyfileobj(audio.file, temp_file)
            temp_path = temp_file.name
        
        try:
            # Transcribe and analyze intent
            start_time = time.time()
            result = asr_manager.transcribe_and_analyze(temp_path, model_size)
            processing_time = time.time() - start_time
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            
            # Generate intelligent response
            user_context_data = {}
            if user_context:
                try:
                    user_context_data = json.loads(user_context)
                except:
                    user_context_data = {}
            
            intelligent_response = asr_manager.get_intelligent_response(
                result["intent_analysis"], 
                user_context_data
            )
            
            # Add TTS if requested
            if enable_tts and asr_manager.tts_engine:
                tts_text = intelligent_response["primary_message"]
                tts_result = asr_manager.text_to_speech(tts_text)
                if tts_result.get("success"):
                    audio_base64 = base64.b64encode(tts_result["audio_data"]).decode('utf-8')
                    intelligent_response["tts_audio"] = audio_base64
            
            # Prepare final response
            response_data = {
                "phone": phone,
                "text": result["text"],
                "confidence": result.get("confidence", 0.0),
                "model_used": result["model_used"],
                "language": result["language"],
                "processing_time": processing_time,
                "intent_analysis": result["intent_analysis"],
                "intelligent_response": intelligent_response,
                "conversation_id": result["conversation_id"]
            }
            
            return response_data
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligent transcription error: {e}")
        raise HTTPException(status_code=500, detail="I had trouble understanding your request. Please try again! 🤔")

@app.get("/intelligent-health")
async def intelligent_health():
    """Intelligent health check"""
    return {
        "status": "🧠 Super intelligent and ready!",
        "message": "I'm here to understand your needs and provide comprehensive help! 🤖",
        "capabilities": [
            "Intent recognition and analysis",
            "Comprehensive response options", 
            "Contextual suggestions",
            "Follow-up questions",
            "Helpful tips",
            "Quick actions"
        ],
        "whisper_models": list(asr_manager.whisper_models.keys()),
        "tts_available": TTS_AVAILABLE,
        "speech_recognition_available": sr is not None,
        "conversation_history_size": len(asr_manager.conversation_history)
    }

@app.get("/conversation-history")
async def get_conversation_history():
    """Get conversation history for context"""
    return {
        "history": asr_manager.conversation_history[-10:],  # Last 10 interactions
        "total_interactions": len(asr_manager.conversation_history),
        "intents_detected": list(set(h["intent"] for h in asr_manager.conversation_history)),
        "most_common_intent": get_most_common_intent()
    }

def get_most_common_intent():
    """Get most common intent from history"""
    if not asr_manager.conversation_history:
        return "unknown"
    
    intents = [h["intent"] for h in asr_manager.conversation_history if h["intent"] != "unknown"]
    if not intents:
        return "unknown"
    
    from collections import Counter
    intent_counts = Counter(intents)
    most_common = intent_counts.most_common(1)[0][0]
    return most_common

if __name__ == "__main__":
    import uvicorn
    
    print("🧠 Intelligent Friendly ASR System")
    print("=" * 50)
    print(f"🤖 Intent Analysis: Ready")
    print(f"💡 Smart Suggestions: Enabled")
    print(f"🎯 Contextual Options: Active")
    print(f"❓ Follow-up Questions: Intelligent")
    print(f"💡 Helpful Tips: Personalized")
    print(f"⚡ Quick Actions: Available")
    print(f"🔊 TTS: {'Available' if TTS_AVAILABLE else 'Not Available'}")
    print()
    print("🌈 Starting intelligent server...")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
