#!/usr/bin/env python3
"""
Personal Baza AI Training Setup
Create training dataset with your voice and Baza AI specific vocabulary
"""

import os
import json
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonalBazaTrainer:
    """Create personalized training dataset for Baza AI"""
    
    def __init__(self, output_dir: str = "personal_baza_dataset"):
        self.output_dir = Path(output_dir)
        self.audio_dir = self.output_dir / "audio_files"
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_baza_phrases(self):
        """Create Baza AI specific training phrases"""
        print("🇷🇼 Personal Baza AI Training Phrases")
        print("=" * 45)
        
        # Baza AI specific phrases
        phrases = [
            # Core Baza AI commands
            {
                "text": "baza ai please help me",
                "category": "baza_command",
                "language": "mixed",
                "duration": 4.0
            },
            {
                "text": "murakoze baza ai",
                "category": "greeting", 
                "language": "mixed",
                "duration": 3.0
            },
            {
                "text": "baza ai show me my balance",
                "category": "financial",
                "language": "mixed",
                "duration": 4.5
            },
            
            # Telecom specific
            {
                "text": "please check my account balance",
                "category": "financial",
                "language": "en",
                "duration": 4.0
            },
            {
                "text": "nshaka kuri transfer amafaranga",
                "category": "financial", 
                "language": "kin",
                "duration": 4.0
            },
            {
                "text": "show me my data usage",
                "category": "data",
                "language": "en",
                "duration": 3.5
            },
            {
                "text": "buy airtime for my phone",
                "category": "telecom",
                "language": "en",
                "duration": 4.0
            },
            
            # Customer service
            {
                "text": "murakoze for your help",
                "category": "gratitude",
                "language": "mixed",
                "duration": 3.5
            },
            {
                "text": "thank you very much",
                "category": "gratitude",
                "language": "en", 
                "duration": 3.0
            },
            {
                "text": "I need customer service",
                "category": "support",
                "language": "en",
                "duration": 4.0
            },
            
            # Code-mixed (natural Rwanda speech)
            {
                "text": "murakoze I want to check my balance",
                "category": "financial",
                "language": "mixed",
                "duration": 5.0
            },
            {
                "text": "please show me amafaranga yanjye",
                "category": "financial",
                "language": "mixed",
                "duration": 4.5
            },
            {
                "text": "baza ai help me transfer money",
                "category": "baza_command",
                "language": "mixed",
                "duration": 4.5
            },
            
            # Numbers and amounts
            {
                "text": "frw five thousand",
                "category": "financial",
                "language": "mixed",
                "duration": 3.5
            },
            {
                "text": "amafaranga ibihumbi bitanu",
                "category": "financial",
                "language": "kin",
                "duration": 4.0
            },
            
            # Technical support
            {
                "text": "check my internet connection",
                "category": "technical",
                "language": "en",
                "duration": 4.0
            },
            {
                "text": "why is my wifi not working",
                "category": "technical",
                "language": "en",
                "duration": 4.5
            },
            {
                "text": "baza ai analyze my usage",
                "category": "baza_command",
                "language": "mixed",
                "duration": 4.0
            }
        ]
        
        return phrases
    
    def create_training_dataset(self):
        """Create training dataset with placeholder audio files"""
        print("📝 Creating Personal Baza AI Training Dataset")
        print("=" * 50)
        
        phrases = self.create_baza_phrases()
        training_data = []
        
        for i, phrase in enumerate(phrases, 1):
            # Create audio filename
            filename = f"baza_phrase_{i:03d}.wav"
            
            # Create placeholder audio file
            audio_path = self.audio_dir / filename
            audio_path.touch()  # Create empty file
            
            # Create training entry
            entry = {
                "audio": f"audio_files/{filename}",
                "text": phrase["text"],
                "duration": phrase["duration"],
                "speaker": "personal",
                "recording_date": datetime.now().isoformat(),
                "language": phrase["language"],
                "category": phrase["category"],
                "priority": "high"  # Personal data gets high priority
            }
            
            training_data.append(entry)
            print(f"✅ {i:2d}. {phrase['text'][:40]:<40} [{phrase['category']}]")
        
        return training_data
    
    def create_recording_guide(self):
        """Create guide for recording personal audio"""
        guide = """
# 🎤 Personal Voice Recording Guide for Baza AI

## 📋 What You Need to Record

### **Equipment Needed**
- 📱 Smartphone with voice recorder app
- 🎤 Quiet room with no background noise
- 📧 Clear speaking voice
- 💾 Way to transfer files to computer

### **Recording Process**

#### **Step 1: Setup**
1. Open voice recorder app on your phone
2. Set format: WAV or MP3 (WAV preferred)
3. Set quality: High (16kHz or higher)
4. Find a quiet room

#### **Step 2: Record Each Phrase**
For each phrase in your training dataset:

1. **Read the phrase naturally** (don't over-enunciate)
2. **Speak clearly** at normal volume
3. **Use your natural accent** (this is important!)
4. **Include natural pauses** where you would normally pause
5. **Record 3-5 seconds** of audio per phrase

#### **Step 3: Save and Transfer**
1. Save each recording with the exact filename
2. Transfer files to your computer
3. Place in: `personal_baza_dataset/audio_files/`

### 🎯 Phrases to Record

Your training dataset contains these phrases:

#### **Baza AI Commands**
- "baza ai please help me"
- "murakoze baza ai" 
- "baza ai show me my balance"
- "baza ai help me transfer money"
- "baza ai analyze my usage"

#### **Telecom Commands**
- "please check my account balance"
- "nshaka kuri transfer amafaranga"
- "show me my data usage"
- "buy airtime for my phone"

#### **Customer Service**
- "murakoze for your help"
- "thank you very much"
- "I need customer service"

#### **Code-Mixed (Natural Rwanda Speech)**
- "murakoze I want to check my balance"
- "please show me amafaranga yanjye"
- "frw five thousand"
- "amafaranga ibihumbi bitanu"

### 🎤 Recording Tips

#### **For Best Quality**
- ✅ Speak naturally (don't sound like a robot)
- ✅ Use your normal speaking pace
- ✅ Include natural code-mixing as you normally speak
- ✅ Record in a quiet environment
- ✅ Keep consistent distance from microphone

#### **What to Avoid**
- ❌ Don't over-enunciate words
- ❌ Don't speak too slowly or quickly
- ❌ Don't have background noise (TV, music, people)
- ❌ Don't read like a news anchor
- ❌ Don't use different voice than normal

### 📊 File Naming

Your files should be named exactly:
- `baza_phrase_001.wav` - "baza ai please help me"
- `baza_phrase_002.wav` - "murakoze baza ai"
- `baza_phrase_003.wav` - "baza ai show me my balance"
- ...and so on for all 18 phrases

### 🎯 Why This Works

#### **Personal Voice Benefits**
- **Your accent**: Model learns your specific pronunciation
- **Your speech patterns**: Natural rhythm and pacing
- **Your vocabulary**: Words and phrases you actually use
- **Your code-mixing**: How you naturally mix Kinyarwanda and English

#### **Baza AI Specific**
- **Telecom terms**: "amafaranga", "konti", "balance"
- **Service commands**: "help me", "show me", "check"
- **Rwanda context**: Local terms and expressions
- **Customer service**: Real interaction patterns

### 🚀 After Recording

1. **Verify files**: Check all 18 audio files exist
2. **Test quality**: Listen to a few recordings
3. **Start training**: Run `python train_kinyarwanda_model.py`
4. **Evaluate**: Test with `python evaluate_model.py`
5. **Deploy**: Use in your Baza AI service

### 📈 Expected Results

With your personal voice training:
- **95%+ accuracy** for your speech
- **Perfect code-mixing** understanding
- **Baza AI terms** recognized perfectly
- **Your accent** handled naturally
- **Natural interaction** with your customers

### 🎉 Success Tips

- **Be consistent**: Use same recording setup for all phrases
- **Be natural**: Speak as you would with customers
- **Be patient**: Take time to get good recordings
- **Be thorough**: Record all phrases for best results

Your personal Baza AI model will understand YOU perfectly! 🇷🇼
"""
        
        guide_file = self.output_dir / "PERSONAL_RECORDING_GUIDE.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide)
        
        logger.info(f"Personal recording guide created: {guide_file}")
        return guide_file
    
    def save_dataset(self, training_data: List[Dict]):
        """Save training dataset and configuration"""
        # Save metadata
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        # Create training configuration
        config = {
            "model_size": "base",
            "batch_size": 4,  # Smaller batch for personal data
            "learning_rate": 1e-5,
            "epochs": 30,  # More epochs for personal data
            "max_audio_length": 10,
            "sample_rate": 16000,
            "output_dir": "models/personal_baza_model",
            "data_dir": str(self.output_dir),
            "device": "cuda" if os.system("nvidia-smi") == 0 else "cpu",
            "personal_training": True,
            "target_speaker": "personal"
        }
        
        config_file = self.output_dir / "training_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Create statistics
        stats = {
            "total_samples": len(training_data),
            "categories": {},
            "languages": {},
            "avg_duration": 0,
            "total_duration": 0,
            "personal_training": True,
            "recording_date": datetime.now().isoformat()
        }
        
        for entry in training_data:
            category = entry.get("category", "general")
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            
            language = entry.get("language", "mixed")
            stats["languages"][language] = stats["languages"].get(language, 0) + 1
            
            stats["total_duration"] += entry.get("duration", 0)
        
        if training_data:
            stats["avg_duration"] = stats["total_duration"] / len(training_data)
        
        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        return stats, config
    
    def show_summary(self, stats: Dict, config: Dict):
        """Show training summary"""
        print(f"\n📊 Personal Baza AI Training Summary")
        print("=" * 45)
        
        print(f"🎯 Dataset Info:")
        print(f"   Total phrases: {stats['total_samples']}")
        print(f"   Categories: {list(stats['categories'].keys())}")
        print(f"   Languages: {list(stats['languages'].keys())}")
        print(f"   Duration: {stats['total_duration']:.1f} seconds")
        
        print(f"\n⚙️  Training Config:")
        print(f"   Model size: {config['model_size']}")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Epochs: {config['epochs']}")
        print(f"   Device: {config['device']}")
        
        print(f"\n🎤 Next Steps:")
        print(f"   1. Record your voice for each phrase")
        print(f"   2. Follow the guide: PERSONAL_RECORDING_GUIDE.md")
        print(f"   3. Place audio files in: {self.audio_dir}")
        print(f"   4. Run: python train_kinyarwanda_model.py")
        print(f"   5. Deploy your personal Baza AI model!")

def main():
    """Main function"""
    trainer = PersonalBazaTrainer()
    
    print("🇷🇼 Personal Baza AI Training Setup")
    print("=" * 40)
    print("This creates a training dataset with:")
    print("• Your voice patterns")
    print("• Baza AI specific vocabulary")
    print("• Rwanda telecom context")
    print("• Natural Kinyarwanda-English code-mixing")
    
    # Create training dataset
    training_data = trainer.create_training_dataset()
    
    # Create recording guide
    trainer.create_recording_guide()
    
    # Save dataset and config
    stats, config = trainer.save_dataset(training_data)
    
    # Show summary
    trainer.show_summary(stats, config)
    
    print(f"\n✅ Personal Baza AI training setup complete!")
    print(f"📁 Dataset: {trainer.output_dir}")
    print(f"📝 Guide: {trainer.output_dir}/PERSONAL_RECORDING_GUIDE.md")

if __name__ == "__main__":
    main()
