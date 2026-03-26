#!/usr/bin/env python3
"""
Setup Personal Baza AI Training
Create directories and initial files for personal voice training
"""

import os
import json
from pathlib import Path
from datetime import datetime

def setup_directories():
    """Create necessary directories"""
    directories = [
        "personal_baza_dataset",
        "personal_baza_dataset/audio_files", 
        "personal_baza_dataset/uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def create_training_phrases():
    """Create training phrases file"""
    phrases = [
        {
            "id": 1,
            "text": "baza ai please help me",
            "filename": "baza_phrase_001.wav",
            "category": "baza_command",
            "language": "mixed"
        },
        {
            "id": 2,
            "text": "murakoze baza ai",
            "filename": "baza_phrase_002.wav", 
            "category": "greeting",
            "language": "mixed"
        },
        {
            "id": 3,
            "text": "baza ai show me my balance",
            "filename": "baza_phrase_003.wav",
            "category": "financial",
            "language": "mixed"
        },
        {
            "id": 4,
            "text": "please check my account balance",
            "filename": "baza_phrase_004.wav",
            "category": "financial",
            "language": "en"
        },
        {
            "id": 5,
            "text": "nshaka kuri transfer amafaranga",
            "filename": "baza_phrase_005.wav",
            "category": "financial",
            "language": "kin"
        },
        {
            "id": 6,
            "text": "show me my data usage",
            "filename": "baza_phrase_006.wav",
            "category": "data",
            "language": "en"
        },
        {
            "id": 7,
            "text": "buy airtime for my phone",
            "filename": "baza_phrase_007.wav",
            "category": "telecom",
            "language": "en"
        },
        {
            "id": 8,
            "text": "murakoze for your help",
            "filename": "baza_phrase_008.wav",
            "category": "gratitude",
            "language": "mixed"
        },
        {
            "id": 9,
            "text": "thank you very much",
            "filename": "baza_phrase_009.wav",
            "category": "gratitude",
            "language": "en"
        },
        {
            "id": 10,
            "text": "I need customer service",
            "filename": "baza_phrase_010.wav",
            "category": "support",
            "language": "en"
        },
        {
            "id": 11,
            "text": "murakoze I want to check my balance",
            "filename": "baza_phrase_011.wav",
            "category": "financial",
            "language": "mixed"
        },
        {
            "id": 12,
            "text": "please show me amafaranga yanjye",
            "filename": "baza_phrase_012.wav",
            "category": "financial",
            "language": "mixed"
        },
        {
            "id": 13,
            "text": "baza ai help me transfer money",
            "filename": "baza_phrase_013.wav",
            "category": "baza_command",
            "language": "mixed"
        },
        {
            "id": 14,
            "text": "frw five thousand",
            "filename": "baza_phrase_014.wav",
            "category": "financial",
            "language": "mixed"
        },
        {
            "id": 15,
            "text": "amafaranga ibihumbi bitanu",
            "filename": "baza_phrase_015.wav",
            "category": "financial",
            "language": "kin"
        },
        {
            "id": 16,
            "text": "check my internet connection",
            "filename": "baza_phrase_016.wav",
            "category": "technical",
            "language": "en"
        },
        {
            "id": 17,
            "text": "why is my wifi not working",
            "filename": "baza_phrase_017.wav",
            "category": "technical",
            "language": "en"
        },
        {
            "id": 18,
            "text": "baza ai analyze my usage",
            "filename": "baza_phrase_018.wav",
            "category": "baza_command",
            "language": "mixed"
        }
    ]
    
    # Save phrases file
    phrases_file = Path("personal_baza_dataset/training_phrases.json")
    with open(phrases_file, 'w', encoding='utf-8') as f:
        json.dump(phrases, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created: {phrases_file}")
    return phrases

def create_initial_metadata():
    """Create initial metadata file"""
    metadata = []
    
    metadata_file = Path("personal_baza_dataset/metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created: {metadata_file}")

def create_training_config():
    """Create training configuration"""
    config = {
        "model_size": "base",
        "batch_size": 4,
        "learning_rate": 1e-5,
        "epochs": 30,
        "max_audio_length": 10,
        "sample_rate": 16000,
        "output_dir": "models/personal_baza_model",
        "data_dir": "personal_baza_dataset",
        "device": "cuda" if os.system("nvidia-smi") == 0 else "cpu",
        "personal_training": True,
        "target_speaker": "personal"
    }
    
    config_file = Path("personal_baza_dataset/training_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created: {config_file}")

def create_recording_checklist():
    """Create recording checklist"""
    checklist = """
# 🎤 Personal Baza AI Recording Checklist

## 📋 Files to Record (18 total)

### ✅ Core Baza AI Commands
[ ] baza_phrase_001.wav - "baza ai please help me"
[ ] baza_phrase_002.wav - "murakoze baza ai"
[ ] baza_phrase_003.wav - "baza ai show me my balance"
[ ] baza_phrase_013.wav - "baza ai help me transfer money"
[ ] baza_phrase_018.wav - "baza ai analyze my usage"

### ✅ Financial Commands
[ ] baza_phrase_004.wav - "please check my account balance"
[ ] baza_phrase_005.wav - "nshaka kuri transfer amafaranga"
[ ] baza_phrase_011.wav - "murakoze I want to check my balance"
[ ] baza_phrase_012.wav - "please show me amafaranga yanjye"
[ ] baza_phrase_014.wav - "frw five thousand"
[ ] baza_phrase_015.wav - "amafaranga ibihumbi bitanu"

### ✅ Telecom Commands
[ ] baza_phrase_006.wav - "show me my data usage"
[ ] baza_phrase_007.wav - "buy airtime for my phone"

### ✅ Customer Service
[ ] baza_phrase_008.wav - "murakoze for your help"
[ ] baza_phrase_009.wav - "thank you very much"
[ ] baza_phrase_010.wav - "I need customer service"

### ✅ Technical Support
[ ] baza_phrase_016.wav - "check my internet connection"
[ ] baza_phrase_017.wav - "why is my wifi not working"

## 🎯 Recording Tips
- Speak naturally (don't over-enunciate)
- Use your normal Rwanda accent
- Include natural code-mixing
- Record in quiet environment
- 3-6 seconds per phrase
- WAV format preferred

## 📁 After Recording
1. Place files in: personal_baza_dataset/uploads/
2. Run: python voice_upload_manager.py
3. Choose option 1 to process uploads
4. Check training readiness
5. Start training!

## 🚀 Expected Results
- 95%+ accuracy for your voice
- Perfect code-mixing understanding
- Baza AI terms recognition
- Natural customer interaction
"""
    
    checklist_file = Path("personal_baza_dataset/RECORDING_CHECKLIST.md")
    with open(checklist_file, 'w', encoding='utf-8') as f:
        f.write(checklist)
    
    print(f"✅ Created: {checklist_file}")

def main():
    """Main setup function"""
    print("🇷🇼 Setting Up Personal Baza AI Training")
    print("=" * 45)
    
    # Create directories
    setup_directories()
    
    # Create training phrases
    phrases = create_training_phrases()
    
    # Create initial metadata
    create_initial_metadata()
    
    # Create training config
    create_training_config()
    
    # Create recording checklist
    create_recording_checklist()
    
    print(f"\n🎯 Setup Complete!")
    print(f"📁 Directory: personal_baza_dataset/")
    print(f"📝 Checklist: RECORDING_CHECKLIST.md")
    print(f"📋 Phrases: training_phrases.json")
    
    print(f"\n📤 Next Steps:")
    print(f"1. Record 18 phrases (see checklist)")
    print(f"2. Place files in: personal_baza_dataset/uploads/")
    print(f"3. Run: python voice_upload_manager.py")
    print(f"4. Start training: python train_kinyarwanda_model.py")
    
    print(f"\n🎉 Your personal Baza AI model will understand YOUR voice!")

if __name__ == "__main__":
    main()
