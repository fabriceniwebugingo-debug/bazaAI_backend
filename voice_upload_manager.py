#!/usr/bin/env python3
"""
Voice Upload Manager for Personal Baza AI Training
Upload and organize recorded voices matching each phrase
"""

import os
import json
import shutil
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceUploadManager:
    """Manage uploaded voice recordings for Baza AI training"""
    
    def __init__(self, dataset_dir: str = "personal_baza_dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.audio_dir = self.dataset_dir / "audio_files"
        self.upload_dir = self.dataset_dir / "uploads"
        self.setup_directories()
        
        # Baza AI training phrases
        self.training_phrases = [
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
    
    def setup_directories(self):
        """Create necessary directories"""
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def show_upload_guide(self):
        """Show upload guide and current status"""
        print("🇷🇼 Voice Upload Manager for Baza AI")
        print("=" * 45)
        
        print(f"\n📁 Upload Directory: {self.upload_dir}")
        print(f"📁 Target Directory: {self.audio_dir}")
        
        print(f"\n📋 Upload Instructions:")
        print(f"1. Record your voice for each phrase")
        print(f"2. Save recordings with these exact names:")
        
        for phrase in self.training_phrases:
            status = "✅" if (self.audio_dir / phrase["filename"]).exists() else "⭕"
            print(f"   {status} {phrase['filename']}")
            print(f"      '{phrase['text']}'")
        
        print(f"\n📤 How to Upload:")
        print(f"1. Place your recorded files in: {self.upload_dir}")
        print(f"2. Run: python voice_upload_manager.py")
        print(f"3. Choose option 1 to process uploads")
        print(f"4. Files will be automatically organized")
    
    def scan_uploads(self) -> List[Path]:
        """Scan upload directory for audio files"""
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac']
        uploaded_files = []
        
        for file_path in self.upload_dir.rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                uploaded_files.append(file_path)
        
        return uploaded_files
    
    def match_upload_to_phrase(self, uploaded_file: Path) -> Dict:
        """Match uploaded file to training phrase"""
        filename = uploaded_file.name.lower()
        
        # Try exact filename match
        for phrase in self.training_phrases:
            if filename == phrase["filename"].lower():
                return phrase
        
        # Try partial match (if user named differently)
        for phrase in self.training_phrases:
            phrase_words = phrase["text"].lower().split()
            filename_words = filename.replace('_', ' ').replace('-', ' ').replace('.wav', '').split()
            
            # Check if significant words match
            matches = sum(1 for word in phrase_words if word in filename_words)
            if matches >= 2:  # At least 2 words match
                return phrase
        
        return None
    
    def process_uploads(self):
        """Process uploaded files and organize them"""
        print("\n📤 Processing Uploaded Files")
        print("=" * 35)
        
        uploaded_files = self.scan_uploads()
        
        if not uploaded_files:
            print("❌ No audio files found in upload directory")
            print(f"📁 Upload directory: {self.upload_dir}")
            return
        
        print(f"📁 Found {len(uploaded_files)} uploaded files:")
        
        processed = []
        unmatched = []
        
        for uploaded_file in uploaded_files:
            print(f"\n📄 {uploaded_file.name}")
            
            # Match to phrase
            matched_phrase = self.match_upload_to_phrase(uploaded_file)
            
            if matched_phrase:
                # Convert to WAV if needed
                target_filename = matched_phrase["filename"]
                target_path = self.audio_dir / target_filename
                
                try:
                    # Copy file to target location
                    shutil.copy2(uploaded_file, target_path)
                    
                    print(f"✅ Matched: '{matched_phrase['text']}'")
                    print(f"📁 Saved as: {target_filename}")
                    
                    processed.append({
                        "original": str(uploaded_file),
                        "target": str(target_path),
                        "phrase": matched_phrase
                    })
                    
                except Exception as e:
                    print(f"❌ Error copying file: {e}")
                    unmatched.append(uploaded_file)
            else:
                print(f"⭕ No matching phrase found")
                unmatched.append(uploaded_file)
        
        # Show summary
        print(f"\n📊 Upload Summary:")
        print(f"   Processed: {len(processed)} files")
        print(f"   Unmatched: {len(unmatched)} files")
        
        if unmatched:
            print(f"\n⚠️  Unmatched files:")
            for file in unmatched:
                print(f"   - {file.name}")
            print(f"\n💡 Tips:")
            print(f"   - Rename files to match: baza_phrase_001.wav, baza_phrase_002.wav, etc.")
            print(f"   - Or include phrase words in filename")
        
        return processed, unmatched
    
    def check_training_readiness(self) -> Dict:
        """Check if dataset is ready for training"""
        print("\n🎯 Training Readiness Check")
        print("=" * 35)
        
        status = {
            "total_phrases": len(self.training_phrases),
            "ready_phrases": 0,
            "missing_phrases": [],
            "ready_percentage": 0,
            "can_train": False
        }
        
        for phrase in self.training_phrases:
            audio_path = self.audio_dir / phrase["filename"]
            if audio_path.exists():
                status["ready_phrases"] += 1
                print(f"✅ {phrase['filename']}")
            else:
                status["missing_phrases"].append(phrase)
                print(f"⭕ {phrase['filename']} - '{phrase['text']}'")
        
        status["ready_percentage"] = (status["ready_phrases"] / status["total_phrases"]) * 100
        status["can_train"] = status["ready_percentage"] >= 50  # At least 50% ready
        
        print(f"\n📊 Status:")
        print(f"   Ready: {status['ready_phrases']}/{status['total_phrases']} ({status['ready_percentage']:.1f}%)")
        print(f"   Can train: {'✅ Yes' if status['can_train'] else '❌ Need more files'}")
        
        if status["can_train"]:
            print(f"\n🚀 Ready to train!")
            print(f"   Run: python train_kinyarwanda_model.py")
        else:
            print(f"\n📝 Need more recordings:")
            for phrase in status["missing_phrases"][:5]:  # Show first 5
                print(f"   - {phrase['filename']}: '{phrase['text']}'")
        
        return status
    
    def create_upload_template(self):
        """Create a template for easy recording"""
        template_file = self.upload_dir / "RECORDING_TEMPLATE.txt"
        
        template_content = """
# 🎤 Baza AI Voice Recording Template
# Record each phrase and save with the exact filename shown

1. baza_phrase_001.wav
   Phrase: "baza ai please help me"
   Tips: Natural tone, clear pronunciation

2. baza_phrase_002.wav  
   Phrase: "murakoze baza ai"
   Tips: Warm, friendly tone

3. baza_phrase_003.wav
   Phrase: "baza ai show me my balance"
   Tips: Clear, professional tone

4. baza_phrase_004.wav
   Phrase: "please check my account balance"
   Tips: Customer service tone

5. baza_phrase_005.wav
   Phrase: "nshaka kuri transfer amafaranga"
   Tips: Natural Kinyarwanda pronunciation

6. baza_phrase_006.wav
   Phrase: "show me my data usage"
   Tips: Clear, helpful tone

7. baza_phrase_007.wav
   Phrase: "buy airtime for my phone"
   Tips: Transaction-focused tone

8. baza_phrase_008.wav
   Phrase: "murakoze for your help"
   Tips: Grateful, warm tone

9. baza_phrase_009.wav
   Phrase: "thank you very much"
   Tips: Polite, professional tone

10. baza_phrase_010.wav
    Phrase: "I need customer service"
    Tips: Clear, slightly urgent tone

11. baza_phrase_011.wav
    Phrase: "murakoze I want to check my balance"
    Tips: Natural code-mixing

12. baza_phrase_012.wav
    Phrase: "please show me amafaranga yanjye"
    Tips: Natural code-mixing

13. baza_phrase_013.wav
    Phrase: "baza ai help me transfer money"
    Tips: Command-oriented tone

14. baza_phrase_014.wav
    Phrase: "frw five thousand"
    Tips: Clear number pronunciation

15. baza_phrase_015.wav
    Phrase: "amafaranga ibihumbi bitanu"
    Tips: Clear Kinyarwanda numbers

16. baza_phrase_016.wav
    Phrase: "check my internet connection"
    Tips: Technical support tone

17. baza_phrase_017.wav
    Phrase: "why is my wifi not working"
    Tips: Problem-solving tone

18. baza_phrase_018.wav
    Phrase: "baza ai analyze my usage"
    Tips: Analytical, professional tone

# 🎯 Recording Tips:
# - Speak naturally (don't over-enunciate)
# - Use your normal speaking pace
# - Include natural code-mixing
# - Record in quiet environment
# - Keep consistent distance from microphone
# - Each recording: 3-6 seconds

# 📁 After Recording:
# 1. Place all files in this upload directory
# 2. Run: python voice_upload_manager.py
# 3. Choose option 1 to process uploads
# 4. Check training readiness
"""
        
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        print(f"📝 Recording template created: {template_file}")
        return template_file
    
    def update_metadata(self):
        """Update metadata with current file status"""
        metadata_file = self.dataset_dir / "metadata.json"
        
        # Load existing metadata or create new
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = []
        
        # Update metadata with current files
        updated_metadata = []
        
        for phrase in self.training_phrases:
            audio_path = self.audio_dir / phrase["filename"]
            
            if audio_path.exists():
                entry = {
                    "audio": f"audio_files/{phrase['filename']}",
                    "text": phrase["text"],
                    "duration": 4.0,  # Default duration
                    "speaker": "personal",
                    "recording_date": datetime.now().isoformat(),
                    "language": phrase["language"],
                    "category": phrase["category"],
                    "priority": "high",
                    "file_exists": True
                }
                updated_metadata.append(entry)
        
        # Save updated metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(updated_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Metadata updated: {len(updated_metadata)} files ready")
        return updated_metadata

def main():
    """Main function"""
    manager = VoiceUploadManager()
    
    print("🇷🇼 Voice Upload Manager for Baza AI")
    print("=" * 45)
    
    while True:
        print(f"\n📋 Options:")
        print(f"1. 📤 Process uploaded files")
        print(f"2. 📊 Check training readiness")
        print(f"3. 📝 Show upload guide")
        print(f"4. 📋 Create recording template")
        print(f"5. 🔄 Update metadata")
        print(f"6. 🚪 Exit")
        
        choice = input(f"\n❓ Choose option (1-6): ").strip()
        
        if choice == "1":
            manager.process_uploads()
        elif choice == "2":
            manager.check_training_readiness()
        elif choice == "3":
            manager.show_upload_guide()
        elif choice == "4":
            manager.create_upload_template()
        elif choice == "5":
            manager.update_metadata()
        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
