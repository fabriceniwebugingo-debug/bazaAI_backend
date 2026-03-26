#!/usr/bin/env python3
"""
Create training dataset from mbazaNLP metadata (audio files not available)
We'll create synthetic audio or use existing audio files
"""

import os
import json
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDatasetCreator:
    """Create training dataset from available sources"""
    
    def __init__(self, output_dir: str = "kinyarwanda_dataset"):
        self.output_dir = Path(output_dir)
        self.audio_dir = self.output_dir / "audio_files"
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_dataset_metadata(self) -> List[Dict]:
        """Download dataset metadata from Hugging Face"""
        try:
            # Download CSV files that contain transcriptions
            train_url = "https://huggingface.co/datasets/mbazaNLP/common-voice-kinyarwanda-english-dataset/resolve/main/train.csv"
            test_url = "https://huggingface.co/datasets/mbazaNLP/common-voice-kinyarwanda-english-dataset/resolve/main/test.csv"
            
            train_data = pd.read_csv(train_url)
            test_data = pd.read_csv(test_url)
            
            logger.info(f"Downloaded {len(train_data)} training samples")
            logger.info(f"Downloaded {len(test_data)} test samples")
            
            return pd.concat([train_data, test_data]).to_dict('records')
            
        except Exception as e:
            logger.error(f"Failed to download metadata: {e}")
            return []
    
    def create_synthetic_dataset(self, num_samples: int = 100) -> List[Dict]:
        """Create synthetic dataset for testing"""
        logger.info(f"Creating synthetic dataset with {num_samples} samples")
        
        # Kinyarwanda-English code-mixed examples
        kinyarwanda_phrases = [
            "murakoze bite mbega ndashaka kuri gura data bundles",
            "mwaramutse nshaka kureka balance yanjye",
            "yego ndabizi kuri transfer amafaranga",
            "nifuza kuri guhingura amafaranga kuri telefoni",
            "please show me my account balance",
            "thank you for your help today",
            "I want to buy airtime for my phone",
            "can you check my data usage",
            "how much is the monthly subscription",
            "I need to speak with customer service"
        ]
        
        english_phrases = [
            "hello thank you for calling",
            "I would like to check my balance",
            "please help me with my account",
            "can I get more data bundles",
            "thank you very much",
            "I have a problem with my service",
            "how do I transfer money",
            "what is my phone number",
            "I need to recharge my account"
        ]
        
        code_mixed_phrases = [
            "murakoze I want to check my balance",
            "please show me amafaranga yanjye",
            "nashaka kuri transfer money but it failed",
            "thank you murakoze for your help",
            "I need data bundles kuri telefoni yanjye",
            "please help me nakuze kuri konti",
            "yego I understand the problem",
            "murakoze bite mbega for your service",
            "I want to buy airtime na amafaranga"
        ]
        
        all_phrases = kinyarwanda_phrases + english_phrases + code_mixed_phrases
        
        dataset = []
        for i in range(num_samples):
            phrase = all_phrases[i % len(all_phrases)]
            
            # Create synthetic audio filename (we'll need to record actual audio later)
            entry = {
                "audio": f"audio_files/sample_{i:06d}.wav",
                "text": phrase,
                "duration": np.random.uniform(3.0, 15.0),  # Random duration
                "source": "synthetic",
                "row_index": i,
                "language": "mixed" if any(word in phrase for word in ["murakoze", "bite", "mbega", "yego", "nashaka"]) else "en"
            }
            dataset.append(entry)
        
        return dataset
    
    def create_placeholder_dataset(self) -> List[Dict]:
        """Create dataset with placeholder audio files for testing"""
        logger.info("Creating placeholder dataset for testing")
        
        # Use the sample data we already have
        placeholder_data = [
            {
                "audio": "audio_files/sample_000000.wav",
                "text": "murakoze bite mbega ndashaka kuri gura data bundles",
                "duration": 5.2,
                "source": "placeholder",
                "row_index": 0,
                "language": "mixed"
            },
            {
                "audio": "audio_files/sample_000001.wav",
                "text": "mwaramutse nshaka kureka balance yanjye",
                "duration": 3.8,
                "source": "placeholder",
                "row_index": 1,
                "language": "kin"
            },
            {
                "audio": "audio_files/sample_000002.wav",
                "text": "yego ndabizi kuri transfer amafaranga",
                "duration": 4.1,
                "source": "placeholder",
                "row_index": 2,
                "language": "mixed"
            },
            {
                "audio": "audio_files/sample_000003.wav",
                "text": "please show me my account balance",
                "duration": 4.5,
                "source": "placeholder",
                "row_index": 3,
                "language": "en"
            },
            {
                "audio": "audio_files/sample_000004.wav",
                "text": "nifuza kuri guhingura amafaranga kuri telefoni",
                "duration": 5.7,
                "source": "placeholder",
                "row_index": 4,
                "language": "mixed"
            }
        ]
        
        return placeholder_data
    
    def create_audio_recording_guide(self):
        """Create guide for recording audio files"""
        guide = """
# 🎤 Audio Recording Guide for Kinyarwanda Training

## 📋 What You Need to Record

### **Required Equipment**
- 📱 Smartphone or voice recorder
- 🎤 Quiet environment
- 🎧 Good microphone
- 💾 Audio recording app

### **Recording Guidelines**
- **Duration**: 3-15 seconds per clip
- **Quality**: Clear speech, minimal background noise
- **Format**: WAV, 16kHz, 16-bit (recommended)
- **Content**: Natural speech as used in daily life

### **Phrases to Record**
Use the transcriptions from your dataset. Examples:

#### **Kinyarwanda Phrases**
1. "murakoze bite mbega ndashaka kuri gura data bundles"
2. "mwaramutse nshaka kureka balance yanjye"
3. "yego ndabizi kuri transfer amafaranga"

#### **English Phrases**
1. "hello thank you for calling"
2. "I would like to check my balance"
3. "can I get more data bundles"

#### **Code-Mixed Phrases**
1. "murakoze I want to check my balance"
2. "please show me amafaranga yanjye"
3. "I need data bundles kuri telefoni yanjye"

### **Recording Process**
1. Open voice recording app
2. Set format: WAV, 16kHz, 16-bit
3. Record each phrase clearly
4. Save as: sample_000000.wav, sample_000001.wav, etc.
5. Place files in: kinyarwanda_dataset/audio_files/

### **Quality Check**
- ✅ Clear speech, no mumbling
- ✅ Consistent volume throughout
- ✅ No background noise or echo
- ✅ Natural speaking pace
- ✅ Proper pronunciation

### **Next Steps**
1. Record all audio files
2. Verify they match the transcriptions
3. Run: python train_kinyarwanda_model.py
4. Evaluate: python evaluate_model.py

## 🎯 Tips for Best Results

- **Multiple speakers**: Record different people (age, gender)
- **Natural speech**: Speak normally, don't over-enunciate
- **Environment**: Quiet room, no TV/music in background
- **Consistency**: Same recording setup for all samples
- **Review**: Listen back to check quality

## 📞 Telecom-Specific Content

Focus on phrases your customers actually use:
- Balance inquiries
- Data bundle purchases
- Money transfers
- Account information
- Customer service requests
- Technical support

This will ensure your trained model works perfectly for your Rwanda telecom service! 🇷🇼
"""
        
        guide_file = self.output_dir / "AUDIO_RECORDING_GUIDE.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide)
        
        logger.info(f"Audio recording guide created: {guide_file}")
        return guide_file
    
    def save_dataset(self, dataset: List[Dict], filename: str = "metadata.json"):
        """Save dataset to file"""
        metadata_file = self.output_dir / filename
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved: {metadata_file}")
        logger.info(f"Total samples: {len(dataset)}")
        
        return dataset
    
    def create_dataset_statistics(self, dataset: List[Dict]) -> Dict:
        """Create dataset statistics"""
        if not dataset:
            return {}
        
        stats = {
            "total_samples": len(dataset),
            "kin_samples": len([d for d in dataset if d.get("language") == "kin"]),
            "en_samples": len([d for d in dataset if d.get("language") == "en"]),
            "mixed_samples": len([d for d in dataset if d.get("language") == "mixed"]),
            "total_duration": sum(d.get("duration", 0) for d in dataset),
            "avg_duration": 0,
            "sources": list(set(d.get("source", "unknown") for d in dataset))
        }
        
        if dataset:
            stats["avg_duration"] = stats["total_duration"] / len(dataset)
        
        return stats

def main():
    """Main function to create training dataset"""
    print("🇷🇼 Kinyarwanda Training Dataset Creator")
    print("=" * 50)
    
    creator = TrainingDatasetCreator()
    
    print("📊 Dataset Creation Options:")
    print("1. Create placeholder dataset (for testing)")
    print("2. Create synthetic dataset (for pipeline testing)")
    print("3. Try to download actual metadata")
    print("4. Create audio recording guide")
    
    choice = input("\n❓ Choose option (1-4): ").strip()
    
    if choice == "1":
        print("\n🎯 Creating placeholder dataset...")
        dataset = creator.create_placeholder_dataset()
        creator.save_dataset(dataset)
        
        stats = creator.create_dataset_statistics(dataset)
        print(f"✅ Created {stats['total_samples']} placeholder samples")
        print(f"   Kinyarwanda: {stats['kin_samples']}")
        print(f"   English: {stats['en_samples']}")
        print(f"   Mixed: {stats['mixed_samples']}")
        
    elif choice == "2":
        num_samples = input("📥 Number of synthetic samples (default 100): ").strip()
        num_samples = int(num_samples) if num_samples else 100
        
        print(f"\n🎯 Creating synthetic dataset with {num_samples} samples...")
        dataset = creator.create_synthetic_dataset(num_samples)
        creator.save_dataset(dataset)
        
        stats = creator.create_dataset_statistics(dataset)
        print(f"✅ Created {stats['total_samples']} synthetic samples")
        
    elif choice == "3":
        print("\n📥 Downloading actual metadata...")
        dataset = creator.download_dataset_metadata()
        if dataset:
            creator.save_dataset(dataset)
            stats = creator.create_dataset_statistics(dataset)
            print(f"✅ Downloaded {stats['total_samples']} samples")
        else:
            print("❌ Failed to download metadata")
    
    elif choice == "4":
        print("\n📝 Creating audio recording guide...")
        creator.create_audio_recording_guide()
        print("✅ Audio recording guide created!")
        print("📁 File: kinyarwanda_dataset/AUDIO_RECORDING_GUIDE.md")
    
    else:
        print("❌ Invalid choice")
        return
    
    print("\n🎯 Next Steps:")
    print("1. Record audio files following the guide")
    print("2. Place them in: kinyarwanda_dataset/audio_files/")
    print("3. Run: python train_kinyarwanda_model.py")
    print("4. Evaluate: python evaluate_model.py")

if __name__ == "__main__":
    main()
