#!/usr/bin/env python3
"""
Personal Voice Training for Baza AI
Train a Kinyarwanda model using your own voice and Baza AI specific vocabulary
"""

import os
import json
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import threading
import queue
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonalVoiceTrainer:
    """Train Kinyarwanda model with your personal voice and Baza AI vocabulary"""
    
    def __init__(self, output_dir: str = "personal_baza_dataset"):
        self.output_dir = Path(output_dir)
        self.audio_dir = self.output_dir / "audio_files"
        self.setup_directories()
        
        # Baza AI specific vocabulary
        self.baza_vocabulary = [
            # Core Baza AI terms
            "baza", "baza ai", "artificial intelligence", "machine learning",
            
            # Telecom terms (Rwanda context)
            "amafaranga", "konti", "balance", "data bundles", "airtime",
            "transfer", "mobile money", "momo", "tigo", "airtel", "mtn",
            
            # Customer service phrases
            "murakoze", "bite mbega", "mwaramutse", "muraho", "yego", "oya",
            "ndashaka", "nshaka", "nifuza", "nkunda", "sinzi", "ndabizi",
            
            # Baza AI commands
            "show me", "check my", "tell me", "help me", "find my", "calculate",
            "translate", "summarize", "analyze", "predict", "recommend",
            
            # Rwanda specific
            "frw", "rwandan franc", "kigali", "rwanda", "kinyarwanda",
            "ikinyarwanda", "telefoni", "simu", "internet", "wifi",
            
            # Code-mixed phrases (common in Rwanda)
            "please show me", "murakoze thank you", "check my balance",
            "transfer amafaranga", "buy data bundles", "help me please",
            "nshaka kuri get", "please help me", "murakoze for your help"
        ]
        
        # Training phrases for Baza AI
        self.training_phrases = [
            # Greetings
            "murakoze bite mbega",
            "mwaramutse",
            "muraho",
            "yego",
            "oya",
            
            # Baza AI specific
            "baza ai please help me",
            "murakoze baza ai",
            "baza ai show me my balance",
            "baza ai transfer amafaranga",
            "baza ai buy data bundles",
            
            # Telecom commands
            "please check my account balance",
            "nshaka kuri transfer amafaranga",
            "show me my data usage",
            "buy airtime for my phone",
            "check my mobile money balance",
            
            # Customer service
            "murakoze for your help",
            "thank you very much",
            "I need customer service",
            "please help me with my account",
            "can you check my balance",
            
            # Code-mixed (natural Rwanda speech)
            "murakoze I want to check my balance",
            "please show me amafaranga yanjye",
            "baza ai help me transfer money",
            "nshaka kuri get data bundles",
            "thank you murakoze for your service",
            
            # Numbers and amounts
            "frw five thousand",
            "amafaranga ibihumbi bitanu",
            "one hundred francs",
            "amafarana magana abiri",
            
            # Technical terms
            "check my internet connection",
            "why is my wifi not working",
            "baza ai analyze my usage",
            "predict my data consumption"
        ]
    
    def setup_directories(self):
        """Create necessary directories"""
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def record_audio(self, phrase: str, duration: float = 5.0) -> str:
        """Record audio for a specific phrase"""
        print(f"\n🎤 Recording: '{phrase}'")
        print(f"⏱️  Duration: {duration} seconds")
        print(f"🔴 Press ENTER to start recording...")
        input()
        
        print("🎙️  Recording... Speak now!")
        
        # Record audio
        sample_rate = 16000
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"baza_phrase_{timestamp}.wav"
        audio_path = self.audio_dir / filename
        
        # Save audio
        sf.write(str(audio_path), audio_data, sample_rate)
        
        print(f"✅ Saved: {filename}")
        return str(audio_path)
    
    def interactive_recording_session(self):
        """Interactive recording session for all phrases"""
        print("🇷🇼 Personal Voice Training for Baza AI")
        print("=" * 50)
        print(f"📝 You will record {len(self.training_phrases)} phrases")
        print(f"🎯 These phrases are specific to your Baza AI service")
        print(f"🗣️  Use your natural Kinyarwanda-English code-mixing")
        print(f"🎤 Speak clearly and naturally")
        print()
        
        recorded_data = []
        
        for i, phrase in enumerate(self.training_phrases, 1):
            print(f"\n📋 Phrase {i}/{len(self.training_phrases)}")
            print(f"💬 Say: '{phrase}'")
            
            try:
                # Adjust duration based on phrase length
                duration = min(max(len(phrase.split()) * 0.8, 3.0), 8.0)
                
                audio_path = self.record_audio(phrase, duration)
                
                # Create metadata entry
                entry = {
                    "audio": f"audio_files/{Path(audio_path).name}",
                    "text": phrase,
                    "duration": duration,
                    "speaker": "personal",
                    "recording_date": datetime.now().isoformat(),
                    "language": "mixed" if any(word in phrase.lower() for word in ["murakoze", "bite", "mbega", "yego", "nshaka"]) else "en",
                    "category": self.categorize_phrase(phrase)
                }
                
                recorded_data.append(entry)
                
                # Ask if user wants to re-record
                if i < len(self.training_phrases):
                    print(f"\n❓ Re-record this phrase? (y/n, default=n): ", end="")
                    choice = input().strip().lower()
                    if choice == 'y':
                        # Remove last entry and re-record
                        recorded_data.pop()
                        os.remove(audio_path)
                        i -= 1
                        continue
                
            except KeyboardInterrupt:
                print("\n❌ Recording cancelled by user")
                break
            except Exception as e:
                logger.error(f"Error recording phrase {i}: {e}")
                continue
        
        return recorded_data
    
    def categorize_phrase(self, phrase: str) -> str:
        """Categorize phrase by type"""
        phrase_lower = phrase.lower()
        
        if any(word in phrase_lower for word in ["murakoze", "thank", "bite mbega"]):
            return "greeting"
        elif any(word in phrase_lower for word in ["balance", "amafaranga", "money"]):
            return "financial"
        elif any(word in phrase_lower for word in ["data", "internet", "wifi"]):
            return "data"
        elif any(word in phrase_lower for word in ["baza ai", "baza"]):
            return "baza_ai"
        elif any(word in phrase_lower for word in ["help", "show", "check"]):
            return "request"
        else:
            return "general"
    
    def save_training_data(self, recorded_data: List[Dict]):
        """Save recorded data for training"""
        # Save metadata
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(recorded_data, f, indent=2, ensure_ascii=False)
        
        # Create training configuration
        config = {
            "model_size": "base",
            "batch_size": 4,
            "learning_rate": 1e-5,
            "epochs": 20,
            "max_audio_length": 10,
            "sample_rate": 16000,
            "output_dir": "models/personal_baza_model",
            "data_dir": str(self.output_dir),
            "device": "cuda" if os.system("nvidia-smi") == 0 else "cpu"
        }
        
        config_file = self.output_dir / "training_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Create statistics
        stats = {
            "total_samples": len(recorded_data),
            "categories": {},
            "languages": {},
            "avg_duration": 0,
            "total_duration": 0,
            "recording_date": datetime.now().isoformat()
        }
        
        for entry in recorded_data:
            # Count categories
            category = entry.get("category", "general")
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            
            # Count languages
            language = entry.get("language", "mixed")
            stats["languages"][language] = stats["languages"].get(language, 0) + 1
            
            # Duration
            stats["total_duration"] += entry.get("duration", 0)
        
        if recorded_data:
            stats["avg_duration"] = stats["total_duration"] / len(recorded_data)
        
        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Training data saved!")
        print(f"📁 Metadata: {metadata_file}")
        print(f"⚙️  Config: {config_file}")
        print(f"📊 Stats: {stats_file}")
        
        return stats
    
    def create_custom_phrases(self):
        """Allow user to add custom phrases"""
        print(f"\n📝 Add Custom Phrases (Optional)")
        print("=" * 35)
        print("Add phrases specific to your Baza AI service")
        print("Press ENTER with empty text to finish")
        
        custom_phrases = []
        while True:
            phrase = input(f"\n💬 Custom phrase {len(custom_phrases) + 1}: ").strip()
            if not phrase:
                break
            custom_phrases.append(phrase)
        
        if custom_phrases:
            print(f"\n🎤 Recording {len(custom_phrases)} custom phrases...")
            custom_data = []
            
            for i, phrase in enumerate(custom_phrases, 1):
                print(f"\n📋 Custom Phrase {i}/{len(custom_phrases)}")
                try:
                    duration = min(max(len(phrase.split()) * 0.8, 3.0), 8.0)
                    audio_path = self.record_audio(phrase, duration)
                    
                    entry = {
                        "audio": f"audio_files/{Path(audio_path).name}",
                        "text": phrase,
                        "duration": duration,
                        "speaker": "personal",
                        "recording_date": datetime.now().isoformat(),
                        "language": "mixed",
                        "category": "custom"
                    }
                    
                    custom_data.append(entry)
                    
                except Exception as e:
                    logger.error(f"Error recording custom phrase {i}: {e}")
                    continue
            
            return custom_data
        
        return []
    
    def start_training(self):
        """Start the training process"""
        print(f"\n🚀 Starting Personal Model Training")
        print("=" * 40)
        
        try:
            # Import training module
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from train_kinyarwanda_model import KinyarwandaTrainer, TrainingConfig
            
            # Load config
            config_file = self.output_dir / "training_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                config = TrainingConfig(**config_dict)
            else:
                config = TrainingConfig(
                    model_size="base",
                    batch_size=4,
                    learning_rate=1e-5,
                    epochs=20,
                    device="cpu"
                )
            
            print(f"🎯 Training Configuration:")
            print(f"   Model: {config.model_size}")
            print(f"   Epochs: {config.epochs}")
            print(f"   Device: {config.device}")
            
            # Start training
            trainer = KinyarwandaTrainer(config)
            trainer.train_model()
            
            print(f"\n✅ Personal Baza AI model training completed!")
            print(f"📁 Model saved in: {config.output_dir}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            print(f"\n❌ Training failed: {e}")
            print("Check the error and try again")

def main():
    """Main function"""
    trainer = PersonalVoiceTrainer()
    
    print("🇷🇼 Personal Voice Training for Baza AI")
    print("=" * 50)
    print("This will train a Kinyarwanda model using YOUR voice")
    print("and Baza AI specific vocabulary for perfect results!")
    
    # Interactive recording session
    recorded_data = trainer.interactive_recording_session()
    
    if recorded_data:
        # Save training data
        stats = trainer.save_training_data(recorded_data)
        
        print(f"\n📊 Recording Summary:")
        print(f"   Total phrases: {stats['total_samples']}")
        print(f"   Categories: {list(stats['categories'].keys())}")
        print(f"   Languages: {list(stats['languages'].keys())}")
        print(f"   Duration: {stats['total_duration']:.1f} seconds")
        
        # Ask about custom phrases
        print(f"\n❓ Add custom phrases? (y/n): ", end="")
        choice = input().strip().lower()
        
        if choice == 'y':
            custom_data = trainer.create_custom_phrases()
            if custom_data:
                recorded_data.extend(custom_data)
                stats = trainer.save_training_data(recorded_data)
                print(f"✅ Added {len(custom_data)} custom phrases")
        
        # Ask about training
        print(f"\n🚀 Start training your personal model? (y/n): ", end="")
        choice = input().strip().lower()
        
        if choice == 'y':
            trainer.start_training()
        else:
            print(f"\n💾 Training data saved!")
            print(f"Run this script again and choose 'y' to start training")
    
    else:
        print(f"\n❌ No audio recorded")

if __name__ == "__main__":
    main()
