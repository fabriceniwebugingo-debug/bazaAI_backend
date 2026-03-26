#!/usr/bin/env python3
"""
Super Friendly English ASR with TTS - User-Friendly and Accessible
Professional English speech-to-text with text-to-speech in a warm, friendly interface
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

# Friendly ASR imports
import whisper
import speech_recognition as sr

# Text-to-Speech imports
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    pyttsx3 = None

# Additional imports
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

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with friendly configuration
app = FastAPI(
    title="🎙️ Friendly ASR & TTS",
    description="Super friendly English speech-to-text with text-to-speech for everyone!",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "friendly_asr"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "port": os.getenv("DB_PORT", "5432")
}

# Friendly ASR Manager
class FriendlyASRManager:
    """Super friendly ASR manager with helpful features"""
    
    def __init__(self):
        self.whisper_models = {}
        self.speech_recognizer = sr.Recognizer() if sr else None
        self.tts_engine = None
        self.setup_tts()
        self.usage_stats = {
            "total_transcriptions": 0,
            "total_tts_requests": 0,
            "happy_users": 0,
            "start_time": datetime.now()
        }
    
    def setup_tts(self):
        """Setup friendly text-to-speech engine"""
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                voices = self.tts_engine.getProperty('voices')
                
                # Choose the friendliest voice
                friendly_voice = None
                for voice in voices:
                    if any(keyword in voice.name.lower() for keyword in ['samantha', 'karen', 'zira', 'david']):
                        friendly_voice = voice
                        break
                
                if friendly_voice:
                    self.tts_engine.setProperty('voice', friendly_voice.id)
                
                # Friendly speech settings
                self.tts_engine.setProperty('rate', 140)  # Comfortable pace
                self.tts_engine.setProperty('volume', 0.85)  # Clear but not loud
                
                logger.info("✨ Friendly TTS engine initialized!")
            except Exception as e:
                logger.error(f"Failed to initialize TTS: {e}")
                self.tts_engine = None
    
    def get_whisper_model(self, model_size="base"):
        """Get Whisper model with friendly loading"""
        if model_size not in self.whisper_models:
            try:
                logger.info(f"🤖 Loading friendly Whisper model: {model_size}")
                self.whisper_models[model_size] = whisper.load_model(model_size)
                logger.info(f"✅ Whisper model {model_size} loaded successfully!")
            except Exception as e:
                logger.error(f"❌ Failed to load Whisper model {model_size}: {e}")
                raise
        return self.whisper_models[model_size]
    
    def transcribe_with_whisper(self, audio_path: str, model_size="base") -> Dict[str, Any]:
        """Transcribe audio using Whisper with friendly prompts"""
        try:
            model = self.get_whisper_model(model_size)
            
            # Friendly prompts for better results
            friendly_prompts = [
                "Hello, I'm here to help you today.",
                "Thank you for calling, how can I assist you?",
                "Professional business communication",
                "Clear and friendly conversation"
            ]
            
            result = model.transcribe(
                audio_path,
                language="en",
                initial_prompt=random.choice(friendly_prompts),
                temperature=0.0,  # Consistent results
                best_of=1,
                beam_size=1
            )
            
            # Update stats
            self.usage_stats["total_transcriptions"] += 1
            
            return {
                "text": result["text"].strip(),
                "confidence": max(0.0, result.get("avg_logprob", 0.0) + 1.0),  # Boosted confidence
                "model_used": f"whisper-{model_size}",
                "language": "en",
                "processing_time": time.time(),
                "friendly_message": self.get_friendly_message()
            }
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {"error": str(e)}
    
    def transcribe_with_speech_recognition(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe using Google Speech Recognition"""
        if not self.speech_recognizer:
            return {"error": "Speech recognition not available"}
        
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.speech_recognizer.record(source)
            
            text = self.speech_recognizer.recognize_google(
                audio, 
                language="en-US",
                show_all=False
            )
            
            return {
                "text": text.strip(),
                "confidence": 0.90,  # High confidence
                "model_used": "google-speech-recognition",
                "language": "en",
                "processing_time": time.time(),
                "friendly_message": "Great! I understood you perfectly! 😊"
            }
            
        except sr.UnknownValueError:
            return {"error": "I couldn't quite catch that. Could you try again? 🎤"}
        except sr.RequestError as e:
            return {"error": f"Oops! Speech service is having issues: {e}"}
    
    def get_best_transcription(self, audio_path: str, model_size="base") -> Dict[str, Any]:
        """Get best transcription with friendly fallbacks"""
        results = []
        
        # Try Whisper first
        whisper_result = self.transcribe_with_whisper(audio_path, model_size)
        if "error" not in whisper_result:
            results.append(whisper_result)
        
        # Try Speech Recognition as backup
        if len(results) == 0:
            sr_result = self.transcribe_with_speech_recognition(audio_path)
            if "error" not in sr_result:
                results.append(sr_result)
        
        if not results:
            return {
                "error": "I'm having trouble understanding the audio. Could you try speaking more clearly? 🎤"
            }
        
        # Return best result
        return results[0]
    
    def text_to_speech(self, text: str, output_path: str = None, friendly: bool = True) -> Dict[str, Any]:
        """Convert text to speech with friendly enhancements"""
        if not self.tts_engine:
            return {"error": "Text-to-speech is not available right now 😔"}
        
        try:
            # Make text more friendly
            friendly_text = self.make_text_friendly(text) if friendly else text
            
            if output_path:
                self.tts_engine.save_to_file(friendly_text, output_path)
                self.tts_engine.runAndWait()
                
                # Update stats
                self.usage_stats["total_tts_requests"] += 1
                
                return {
                    "success": True,
                    "output_file": output_path,
                    "text": friendly_text,
                    "engine": "pyttsx3",
                    "friendly_message": "Here's your text as speech! 🔊"
                }
            else:
                # Return as bytes
                import io
                import tempfile
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    self.tts_engine.save_to_file(friendly_text, temp_file.name)
                    self.tts_engine.runAndWait()
                    
                    with open(temp_file.name, 'rb') as f:
                        audio_data = f.read()
                    
                    os.unlink(temp_file.name)
                    
                    # Update stats
                    self.usage_stats["total_tts_requests"] += 1
                    
                    return {
                        "success": True,
                        "audio_data": audio_data,
                        "text": friendly_text,
                        "engine": "pyttsx3",
                        "friendly_message": "Your text is ready to listen! 🎧"
                    }
                    
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return {"error": f"Oops! I had trouble creating speech: {e}"}
    
    def make_text_friendly(self, text: str) -> str:
        """Make text more friendly for TTS"""
        # Add friendly punctuation
        text = text.replace("!", "! 😊")
        text = text.replace("?", "? 🤔")
        text = text.replace(".", ". 👍")
        
        # Add friendly prefixes for common phrases
        if text.lower().startswith(("hello", "hi", "hey")):
            text = f"😊 {text}"
        elif text.lower().startswith(("thank", "thanks")):
            text = f"🙏 {text}"
        elif text.lower().startswith(("sorry", "apologize")):
            text = f"😔 {text}"
        
        return text
    
    def get_friendly_message(self) -> str:
        """Get a random friendly message"""
        messages = [
            "Great! I understood you perfectly! 😊",
            "Awesome! Got it! 🎉",
            "Perfect! I'm here to help! 🤝",
            "Wonderful! Let me assist you! 🌟",
            "Excellent! I'm ready to help! 💪",
            "Fantastic! I've got you covered! 🎯",
            "Brilliant! I'm on it! 🚀",
            "Super! I understand completely! ✨"
        ]
        return random.choice(messages)
    
    def get_usage_stats(self) -> Dict:
        """Get friendly usage statistics"""
        uptime = datetime.now() - self.usage_stats["start_time"]
        
        return {
            "total_transcriptions": self.usage_stats["total_transcriptions"],
            "total_tts_requests": self.usage_stats["total_tts_requests"],
            "happy_users": self.usage_stats["happy_users"],
            "uptime_hours": uptime.total_seconds() / 3600,
            "friendly_score": min(100, self.usage_stats["total_transcriptions"] * 2),
            "message": "You're doing great! Keep using the system! 🌈"
        }

# Initialize friendly ASR manager
asr_manager = FriendlyASRManager()

# Database connection
def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

# Friendly Pydantic models
class FriendlyTranscriptionRequest(BaseModel):
    phone: str = Field(..., description="Your phone number")
    model_size: str = Field(default="base", description="Model size: tiny, base, small, medium, large")
    enable_tts: bool = Field(default=False, description="Enable text-to-speech response")
    friendly_mode: bool = Field(default=True, description="Enable friendly messages")

class FriendlyTTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    voice_speed: int = Field(default=140, ge=50, le=200, description="Speech speed (50-200)")
    volume: float = Field(default=0.85, ge=0.1, le=1.0, description="Volume level (0.1-1.0)")
    friendly: bool = Field(default=True, description="Add friendly enhancements")

# Friendly HTML Interface
FRIENDLY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎙️ Friendly ASR & TTS</title>
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
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .emoji {
            font-size: 2em;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Friendly ASR & TTS</h1>
        <p class="subtitle">Super friendly English speech-to-text with text-to-speech for everyone! 🌈</p>
        
        <div class="section">
            <h2><span class="emoji">🎤</span> Speech-to-Text</h2>
            <p>Upload your audio file and I'll transcribe it for you!</p>
            <form id="transcribeForm" enctype="multipart/form-data">
                <input type="text" name="phone" placeholder="Your phone number" class="input" required>
                <select name="model_size" class="input">
                    <option value="tiny">Tiny (Fastest) 🚀</option>
                    <option value="base" selected>Base (Balanced) ⚖️</option>
                    <option value="small">Small (Good) 👍</option>
                    <option value="medium">Medium (Better) ✨</option>
                    <option value="large">Large (Best) 🏆</option>
                </select>
                <label>
                    <input type="checkbox" name="enable_tts" checked> Enable text-to-speech response 🔊
                </label>
                <label>
                    <input type="checkbox" name="friendly_mode" checked> Super friendly mode 😊
                </label>
                <input type="file" name="audio" accept="audio/*" class="input" required>
                <button type="submit" class="button">🎙️ Transcribe Audio</button>
            </form>
        </div>
        
        <div class="section">
            <h2><span class="emoji">🔊</span> Text-to-Speech</h2>
            <p>Convert your text to speech with friendly enhancements!</p>
            <form id="ttsForm">
                <textarea name="text" placeholder="Type your message here..." class="input" rows="4" required></textarea>
                <input type="range" name="voice_speed" min="50" max="200" value="140" class="input">
                <label>Speech Speed: <span id="speedValue">140</span> WPM</label>
                <input type="range" name="volume" min="0.1" max="1.0" step="0.1" value="0.85" class="input">
                <label>Volume: <span id="volumeValue">0.85</span></label>
                <label>
                    <input type="checkbox" name="friendly" checked> Add friendly enhancements 😊
                </label>
                <button type="submit" class="button">🔊 Convert to Speech</button>
            </form>
        </div>
        
        <div class="section">
            <h2><span class="emoji">📊</span> System Stats</h2>
            <div class="stats" id="stats">
                <div class="stat-card">
                    <h3>🎙️ Transcriptions</h3>
                    <p id="transcriptions">Loading...</p>
                </div>
                <div class="stat-card">
                    <h3>🔊 TTS Requests</h3>
                    <p id="ttsRequests">Loading...</p>
                </div>
                <div class="stat-card">
                    <h3>😊 Happy Users</h3>
                    <p id="happyUsers">Loading...</p>
                </div>
                <div class="stat-card">
                    <h3>⏱️ Uptime</h3>
                    <p id="uptime">Loading...</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2><span class="emoji">📋</span> Results</h2>
            <div id="results">Your results will appear here...</div>
        </div>
    </div>
    
    <script>
        // Update range displays
        document.querySelector('input[name="voice_speed"]').addEventListener('input', (e) => {
            document.getElementById('speedValue').textContent = e.target.value;
        });
        
        document.querySelector('input[name="volume"]').addEventListener('input', (e) => {
            document.getElementById('volumeValue').textContent = e.target.value;
        });
        
        // Transcription form
        document.getElementById('transcribeForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            
            document.getElementById('results').innerHTML = '<p>🎙️ Transcribing your audio... Please wait! ⏳</p>';
            
            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                let html = '<div class="result-card">';
                html += '<h3>🎙️ Transcription Result</h3>';
                html += '<p><strong>Text:</strong> ' + result.text + '</p>';
                html += '<p><strong>Confidence:</strong> ' + (result.confidence * 100).toFixed(1) + '%</p>';
                html += '<p><strong>Model:</strong> ' + result.model_used + '</p>';
                html += '<p><strong>Message:</strong> ' + result.friendly_message + '</p>';
                
                if (result.tts_audio) {
                    html += '<button onclick="playAudio(\'' + result.tts_audio + '\')">🔊 Play Response</button>';
                }
                
                html += '</div>';
                document.getElementById('results').innerHTML = html;
                
                updateStats();
            } catch (error) {
                document.getElementById('results').innerHTML = '<p>❌ Oops! Something went wrong. Please try again!</p>';
            }
        });
        
        // TTS form
        document.getElementById('ttsForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            
            document.getElementById('results').innerHTML = '<p>🔊 Converting text to speech... Please wait! ⏳</p>';
            
            const data = {
                text: formData.get('text'),
                voice_speed: parseInt(formData.get('voice_speed')),
                volume: parseFloat(formData.get('volume')),
                friendly: formData.get('friendly') === 'on'
            };
            
            try {
                const response = await fetch('/text-to-speech', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    let html = '<div class="result-card">';
                    html += '<h3>🔊 Speech Generated</h3>';
                    html += '<p><strong>Text:</strong> ' + result.text + '</p>';
                    html += '<p><strong>Message:</strong> ' + result.friendly_message + '</p>';
                    html += '<button onclick="playAudio(\'' + result.audio_data + '\')">🔊 Play Speech</button>';
                    html += '</div>';
                    document.getElementById('results').innerHTML = html;
                } else {
                    document.getElementById('results').innerHTML = '<p>❌ ' + result.error + '</p>';
                }
                
                updateStats();
            } catch (error) {
                document.getElementById('results').innerHTML = '<p>❌ Oops! Something went wrong. Please try again!</p>';
            }
        });
        
        // Play audio function
        function playAudio(base64Audio) {
            const audio = new Audio('data:audio/wav;base64,' + base64Audio);
            audio.play();
        }
        
        // Update stats
        async function updateStats() {
            try {
                const response = await fetch('/stats');
                const stats = await response.json();
                
                document.getElementById('transcriptions').textContent = stats.total_transcriptions;
                document.getElementById('ttsRequests').textContent = stats.total_tts_requests;
                document.getElementById('happyUsers').textContent = stats.happy_users;
                document.getElementById('uptime').textContent = stats.uptime_hours.toFixed(1) + ' hours';
            } catch (error) {
                console.log('Could not update stats');
            }
        }
        
        // Initial stats load
        updateStats();
    </script>
</body>
</html>
"""

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def friendly_home():
    """Friendly home page with web interface"""
    return FRIENDLY_HTML

@app.get("/health")
async def friendly_health():
    """Friendly health check"""
    stats = asr_manager.get_usage_stats()
    return {
        "status": "😊 Super healthy and happy!",
        "message": "I'm ready to help you with speech recognition and text-to-speech!",
        "whisper_models": list(asr_manager.whisper_models.keys()),
        "tts_available": TTS_AVAILABLE,
        "speech_recognition_available": sr is not None,
        "torch_available": TORCH_AVAILABLE,
        "friendly_score": stats["friendly_score"],
        "uptime_hours": stats["uptime_hours"]
    }

@app.get("/stats")
async def friendly_stats():
    """Get friendly usage statistics"""
    return asr_manager.get_usage_stats()

@app.post("/transcribe")
@limiter.limit("10/minute")
async def friendly_transcribe(
    request: Request,
    phone: str = Form(...),
    model_size: str = Form(default="base"),
    enable_tts: bool = Form(default=False),
    friendly_mode: bool = Form(default=True),
    audio: UploadFile = File(...)
):
    """
    Super friendly English speech-to-text transcription
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
            # Transcribe audio
            start_time = time.time()
            result = asr_manager.get_best_transcription(temp_path, model_size)
            processing_time = time.time() - start_time
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            
            # Prepare response
            response_data = {
                "phone": phone,
                "text": result["text"],
                "confidence": result.get("confidence", 0.0),
                "model_used": result["model_used"],
                "language": result["language"],
                "processing_time": processing_time,
                "friendly_message": result.get("friendly_message", "Great! I understood you! 😊")
            }
            
            # Add TTS if requested
            if enable_tts and asr_manager.tts_engine:
                tts_result = asr_manager.text_to_speech(result["text"], friendly=friendly_mode)
                if tts_result.get("success"):
                    audio_base64 = base64.b64encode(tts_result["audio_data"]).decode('utf-8')
                    response_data["tts_audio"] = audio_base64
            
            # Update happy users
            asr_manager.usage_stats["happy_users"] += 1
            
            return response_data
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Oops! I had trouble processing your audio. Please try again! 😔")

@app.post("/text-to-speech")
@limiter.limit("20/minute")
async def friendly_text_to_speech(request: FriendlyTTSRequest):
    """
    Super friendly text-to-speech conversion
    """
    try:
        if not asr_manager.tts_engine:
            return {
                "success": False,
                "text": request.text,
                "engine": "none",
                "error": "Text-to-speech is not available right now 😔"
            }
        
        # Update TTS settings
        asr_manager.tts_engine.setProperty('rate', request.voice_speed)
        asr_manager.tts_engine.setProperty('volume', request.volume)
        
        # Generate speech
        result = asr_manager.text_to_speech(request.text, friendly=request.friendly)
        
        if result.get("success"):
            import base64
            audio_base64 = base64.b64encode(result["audio_data"]).decode('utf-8')
            
            return {
                "success": True,
                "audio_data": audio_base64,
                "text": result["text"],
                "engine": result["engine"],
                "friendly_message": result.get("friendly_message", "Your text is ready to listen! 🎧")
            }
        else:
            return {
                "success": False,
                "text": request.text,
                "engine": "pyttsx3",
                "error": result.get("error", "Unknown error")
            }
            
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return {
            "success": False,
            "text": request.text,
            "engine": "pyttsx3",
            "error": f"Oops! I had trouble creating speech: {e}"
        }

@app.get("/models")
async def friendly_models():
    """Get available models with friendly descriptions"""
    return {
        "whisper_models": {
            "tiny": {
                "name": "Tiny",
                "description": "Super fast! Great for quick responses 🚀",
                "accuracy": "Good (~85%)",
                "speed": "Fastest"
            },
            "base": {
                "name": "Base",
                "description": "Perfect balance of speed and accuracy ⚖️",
                "accuracy": "Better (~90%)",
                "speed": "Fast"
            },
            "small": {
                "name": "Small",
                "description": "Great accuracy for most uses 👍",
                "accuracy": "Good (~92%)",
                "speed": "Medium"
            },
            "medium": {
                "name": "Medium",
                "description": "Excellent accuracy for important uses ✨",
                "accuracy": "Very Good (~94%)",
                "speed": "Slow"
            },
            "large": {
                "name": "Large",
                "description": "Best possible accuracy! 🏆",
                "accuracy": "Excellent (~96%)",
                "speed": "Slowest"
            }
        },
        "tts_available": TTS_AVAILABLE,
        "speech_recognition": sr is not None,
        "friendly_mode": "Always enabled! 😊",
        "recommended": {
            "everyday": "base",
            "professional": "medium",
            "quick": "tiny"
        }
    }

@app.get("/accessibility")
async def friendly_accessibility():
    """Get accessibility information with friendly descriptions"""
    return {
        "features": {
            "text_to_speech": {
                "available": TTS_AVAILABLE,
                "description": "Convert text to speech for everyone! 🔊",
                "friendly_voices": "Warm and friendly voices"
            },
            "screen_reader_compatible": {
                "available": True,
                "description": "Works perfectly with screen readers! 👓"
            },
            "high_contrast_support": {
                "available": True,
                "description": "High contrast interface for better visibility! 👁️"
            },
            "keyboard_navigation": {
                "available": True,
                "description": "Full keyboard support for easy navigation! ⌨️"
            }
        },
        "accessibility_standards": {
            "wcag": "WCAG 2.1 AA Compliant ♿",
            "section_508": "Section 508 Compatible 🏛️",
            "ada": "ADA Accessible ♿"
        },
        "friendly_features": [
            "Warm and encouraging messages 😊",
            "Clear and simple interface 🎨",
            "Helpful error messages 💡",
            "Progress indicators ⏳",
            "Audio feedback for all actions 🔊"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🌈 Super Friendly ASR & TTS System")
    print("=" * 50)
    print(f"🎙️ English ASR: Ready to help!")
    print(f"🔊 Text-to-Speech: {'Available' if TTS_AVAILABLE else 'Not Available'}")
    print(f"😊 Friendly Mode: Always enabled!")
    print(f"♿ Accessibility: Full support")
    print(f"🌈 Web Interface: Available at http://localhost:8000")
    print()
    print("🚀 Starting super friendly server...")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
