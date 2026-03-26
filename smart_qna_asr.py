#!/usr/bin/env python3
"""
Smart Q&A ASR - Understands Questions and Provides Real Answers
Speech recognition that actually answers questions about services, bundles, and more!
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import whisper
import tempfile
import os
import time
import re
from typing import Dict, List, Any

# Initialize FastAPI
app = FastAPI(
    title="🤖 Smart Q&A ASR",
    description="Speech recognition that answers your questions!",
    version="1.0.0"
)

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model
print("🤖 Loading speech recognition model...")
model = whisper.load_model("base")
print("✅ Ready to understand and answer!")

# Service Database - Real information about your services
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
            "name": "Daily 50 Mins",
            "price": 300,
            "currency": "RWF",
            "minutes": "50 mins",
            "validity": "1 day",
            "description": "Perfect for daily calls to all networks",
            "suitable_for": "Daily calling needs"
        },
        {
            "name": "Daily 100 Mins",
            "price": 500,
            "currency": "RWF",
            "minutes": "100 mins",
            "validity": "1 day",
            "description": "Great for heavy daily calling",
            "suitable_for": "Business calls, family calls"
        },
        {
            "name": "Pay As You Go 30 Secs",
            "price": 50,
            "currency": "RWF",
            "seconds": "30 secs",
            "validity": "Per call",
            "description": "Pay per second for short calls",
            "suitable_for": "Quick calls, emergency calls"
        },
        {
            "name": "Pay As You Go 60 Secs",
            "price": 100,
            "currency": "RWF",
            "seconds": "60 secs",
            "validity": "Per call",
            "description": "Pay per second for medium calls",
            "suitable_for": "Standard calls, business calls"
        },
        {
            "name": "Weekly 200 Mins",
            "price": 1500,
            "currency": "RWF",
            "minutes": "200 mins",
            "validity": "7 days",
            "description": "Good for weekly moderate usage",
            "suitable_for": "Weekly calling needs"
        },
        {
            "name": "Weekly 500 Mins",
            "price": 3000,
            "currency": "RWF",
            "minutes": "500 mins",
            "validity": "7 days",
            "description": "Great for weekly heavy usage",
            "suitable_for": "Business users, heavy callers"
        },
        {
            "name": "Monthly 1000 Mins",
            "price": 8000,
            "currency": "RWF",
            "minutes": "1000 mins",
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
            # Internet bundle questions
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

# Smart HTML Interface
SMART_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Smart Q&A ASR - Understands & Answers</title>
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
        .button {
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
        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
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
        .bundles-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .bundle-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .bundle-name {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .bundle-price {
            font-size: 1.5em;
            color: #4caf50;
            margin: 10px 0;
        }
        .bundle-data {
            font-size: 1.1em;
            margin: 5px 0;
        }
        .recommendation {
            background: rgba(76, 175, 80, 0.2);
            border-left: 4px solid #4caf50;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .examples {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }
        .examples h4 {
            margin-bottom: 10px;
            color: #ff6b6b;
        }
        .example {
            margin: 5px 0;
            padding: 5px 0;
            opacity: 0.9;
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
            font-size: 2em;
            margin: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Smart Q&A ASR</h1>
        <p class="subtitle">Ask questions about services, bundles, and more - I'll understand and answer!</p>
        
        <div class="section">
            <h3>🎤 Ask Your Question</h3>
            <div class="upload-area" id="uploadArea">
                <div class="emoji">🎤</div>
                <h3>Record or upload your question</h3>
                <p>I'll understand what you're asking and provide real answers!</p>
                <input type="file" id="fileInput" class="file-input" accept="audio/*">
                <button class="button" onclick="document.getElementById('fileInput').click()">
                    🎤 Upload Question
                </button>
            </div>
            
            <div class="examples">
                <h4>💡 Try asking:</h4>
                <div class="example">📱 "What daily internet bundles can I buy?"</div>
                <div class="example">📊 "How much is the weekly 3GB bundle?"</div>
                <div class="example">💡 "Recommend a bundle for streaming"</div>
                <div class="example">📞 "What calling bundles do you have?"</div>
                <div class="example">⏰ "How many minutes in the daily calling bundle?"</div>
                <div class="example">⚡ "What are your pay as you go seconds?"</div>
                <div class="example">💰 "What airtime plans do you have?"</div>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>🤔 Understanding your question...</p>
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
        
        async function handleFile(file) {
            if (!file.type.startsWith('audio/')) {
                alert('Please upload an audio file! 🎵');
                return;
            }
            
            loading.style.display = 'block';
            result.style.display = 'none';
            
            const formData = new FormData();
            formData.append('audio', file);
            
            try {
                const response = await fetch('/ask-question', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                loading.style.display = 'none';
                
                if (data.error) {
                    questionText.textContent = '❌ Error';
                    answerText.textContent = data.error;
                    detailsArea.innerHTML = '';
                } else {
                    displayAnswer(data);
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
        
        function displayAnswer(data) {
            const answer = data.answer;
            
            questionText.textContent = `📝 You asked: "${data.original_text}"`;
            answerText.textContent = answer.answer;
            
            let detailsHtml = '';
            
            if (answer.type === 'daily_bundles' || answer.type === 'weekly_bundles' || answer.type === 'monthly_bundles') {
                const bundles = answer.bundles || [];
                detailsHtml = '<div class="bundles-grid">';
                bundles.forEach(bundle => {
                    detailsHtml += `
                        <div class="bundle-card">
                            <div class="bundle-name">${bundle.name}</div>
                            <div class="bundle-price">${bundle.price} RWF</div>
                            <div class="bundle-data">📊 ${bundle.data}</div>
                            <div class="bundle-data">⏰ ${bundle.validity}</div>
                            <div style="font-size: 0.9em; margin-top: 10px;">${bundle.description}</div>
                        </div>
                    `;
                });
                detailsHtml += '</div>';
            }
            
            if (answer.type === 'calling_bundles') {
                const bundles = answer.bundles || [];
                detailsHtml = '<div class="bundles-grid">';
                bundles.forEach(bundle => {
                    const timeUnit = bundle.minutes || bundle.seconds;
                    detailsHtml += `
                        <div class="bundle-card">
                            <div class="bundle-name">${bundle.name}</div>
                            <div class="bundle-price">${bundle.price} RWF</div>
                            <div class="bundle-data">⏰ ${timeUnit}</div>
                            <div class="bundle-data">📅 ${bundle.validity}</div>
                            <div style="font-size: 0.9em; margin-top: 10px;">${bundle.description}</div>
                        </div>
                    `;
                });
                detailsHtml += '</div>';
            }
            
            if (answer.type === 'airtime') {
                const plans = answer.plans || [];
                detailsHtml = '<div class="bundles-grid">';
                plans.forEach(plan => {
                    detailsHtml += `
                        <div class="bundle-card">
                            <div class="bundle-name">${plan.name}</div>
                            <div class="bundle-price">${plan.price} RWF</div>
                            <div class="bundle-data">💵 ${plan.airtime} RWF airtime</div>
                            ${plan.bonus > 0 ? `<div class="bundle-data">🎁 ${plan.bonus} RWF bonus</div>` : ''}
                            <div style="font-size: 0.9em; margin-top: 10px;">${plan.description}</div>
                        </div>
                    `;
                });
                detailsHtml += '</div>';
            }
            
            if (answer.type === 'services') {
                const services = answer.services || [];
                detailsHtml = '<div class="bundles-grid">';
                services.forEach(service => {
                    detailsHtml += `
                        <div class="bundle-card">
                            <div class="bundle-name">${service.name}</div>
                            <div style="margin: 10px 0;">${service.description}</div>
                            <div style="font-size: 0.9em;">${service.features.join(' • ')}</div>
                            <div style="margin-top: 10px; color: #4caf50;">${service.cost}</div>
                        </div>
                    `;
                });
                detailsHtml += '</div>';
            }
            
            if (answer.recommendation) {
                detailsHtml += `
                    <div class="recommendation">
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
            
            if (answer.options) {
                detailsHtml += '<h4>🔍 I can help with:</h4><ul>';
                answer.options.forEach(option => {
                    detailsHtml += `<li>${option}</li>`;
                });
                detailsHtml += '</ul>';
            }
            
            if (answer.examples) {
                detailsHtml += '<div class="examples"><h4>💡 Try asking:</h4>';
                answer.examples.forEach(example => {
                    detailsHtml += `<div class="example">${example}</div>`;
                });
                detailsHtml += '</div>';
            }
            
            detailsArea.innerHTML = detailsHtml;
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def smart_home():
    """Smart Q&A home page"""
    return SMART_HTML

@app.post("/ask-question")
async def ask_question(audio: UploadFile = File(...)):
    """
    Understand user question and provide real answers
    """
    try:
        # Validate audio file
        if not audio.content_type.startswith('audio/'):
            return {"error": "Please upload an audio file! 🎵"}
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Transcribe audio
            start_time = time.time()
            result = model.transcribe(temp_path)
            processing_time = time.time() - start_time
            
            transcribed_text = result["text"].strip()
            
            if not transcribed_text:
                return {"error": "I couldn't hear your question. Please try speaking clearly! 🎤"}
            
            # Analyze question and get answer
            qa_result = qa_engine.analyze_question(transcribed_text)
            
            return {
                "original_text": transcribed_text,
                "intent": qa_result["intent"],
                "confidence": qa_result["confidence"],
                "answer": qa_result["answer"],
                "processing_time": processing_time,
                "message": "✅ Question understood and answered!"
            }
            
        finally:
            # Clean up
            os.unlink(temp_path)
            
    except Exception as e:
        return {"error": f"Oops! Something went wrong: {str(e)}"}

@app.get("/health")
async def smart_health():
    """Smart health check"""
    return {
        "status": "🤖 Smart Q&A ASR is ready!",
        "message": "I can understand questions and provide real answers about services!",
        "capabilities": [
            "Understands questions about bundles",
            "Provides real pricing information", 
            "Gives personalized recommendations",
            "Answers service inquiries",
            "Handles airtime questions",
            "Knows calling bundles with minutes"
        ],
        "supported_questions": [
            "Daily internet bundles (MB/GB)",
            "Weekly data packages (MB/GB)",
            "Monthly plans (MB/GB)",
            "Calling bundles (minutes/seconds)",
            "Airtime recharge options (RWF)",
            "Available services",
            "Pricing information (RWF)",
            "Personalized recommendations"
        ],
        "units": {
            "data": "MB/GB",
            "calling": "minutes/seconds",
            "money": "RWF"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🤖 Smart Q&A ASR System")
    print("=" * 50)
    print("✅ Understands real questions!")
    print("✅ Provides actual answers!")
    print("✅ Knows about bundles and services!")
    print("✅ Gives recommendations!")
    print("✅ Handles pricing inquiries!")
    print()
    print("🚀 Starting smart Q&A server...")
    print("🌐 Open http://localhost:8000 to ask questions!")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
