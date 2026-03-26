#!/usr/bin/env python3
"""
Quick Start Script for Kinyarwanda Model Training
Get started with training your own Kinyarwanda ASR model
"""

import os
import json
from pathlib import Path

def setup_training_environment():
    """Setup the training environment"""
    print("🇷🇼 Kinyarwanda Model Training - Quick Start")
    print("=" * 50)
    
    # Create directories
    directories = [
        "kinyarwanda_dataset",
        "kinyarwanda_dataset/audio_files",
        "test_dataset",
        "models/trained_kinyarwanda",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create sample dataset
    create_sample_dataset()
    
    # Create training configuration
    create_training_config()
    
    print("\n🎯 Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. Add your Kinyarwanda audio files to 'kinyarwanda_dataset/audio_files/'")
    print("2. Update 'kinyarwanda_dataset/metadata.json' with your transcriptions")
    print("3. Run: python train_kinyarwanda_model.py")
    print("4. Evaluate: python evaluate_model.py")

def create_sample_dataset():
    """Create sample dataset structure"""
    sample_data = [
        {
            "audio": "audio_files/sample1.wav",
            "text": "murakoze bite mbega ndashaka kuri gura data bundles",
            "speaker_id": "speaker_001",
            "metadata": {"topic": "telecom", "duration": 5.2}
        },
        {
            "audio": "audio_files/sample2.wav",
            "text": "mwaramutse nshaka kureka balance yanjye",
            "speaker_id": "speaker_002", 
            "metadata": {"topic": "balance", "duration": 3.8}
        },
        {
            "audio": "audio_files/sample3.wav",
            "text": "yego ndabizi kuri transfer amafaranga",
            "speaker_id": "speaker_001",
            "metadata": {"topic": "transfer", "duration": 4.1}
        },
        {
            "audio": "audio_files/sample4.wav",
            "text": "please show me my account balance",
            "speaker_id": "speaker_003",
            "metadata": {"topic": "balance", "duration": 4.5}
        },
        {
            "audio": "audio_files/sample5.wav",
            "text": "nifuza kuri guhingura amafaranga kuri telefoni",
            "speaker_id": "speaker_002",
            "metadata": {"topic": "transfer", "duration": 5.7}
        }
    ]
    
    metadata_file = Path("kinyarwanda_dataset/metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created sample metadata: {metadata_file}")
    print(f"   Added {len(sample_data)} sample entries")

def create_training_config():
    """Create training configuration file"""
    config = {
        "model_size": "base",
        "batch_size": 8,
        "learning_rate": 1e-5,
        "epochs": 30,
        "save_steps": 500,
        "eval_steps": 250,
        "max_audio_length": 30,
        "sample_rate": 16000,
        "output_dir": "models/trained_kinyarwanda",
        "data_dir": "kinyarwanda_dataset",
        "device": "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    }
    
    config_file = Path("training_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created training config: {config_file}")
    print(f"   Device: {config['device']}")
    print(f"   Model size: {config['model_size']}")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n🔍 Checking Dependencies:")
    
    required_packages = [
        "torch",
        "torchaudio", 
        "whisper",
        "librosa",
        "soundfile",
        "pandas",
        "numpy",
        "jiwer"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements_new.txt")
    else:
        print("\n✅ All dependencies installed!")

def show_data_collection_tips():
    """Show tips for collecting training data"""
    print("\n📚 Data Collection Tips:")
    print("=" * 30)
    
    tips = [
        "🎤 Record clear speech with minimal background noise",
        "📏 Keep audio clips between 1-30 seconds",
        "🗣️ Include diverse speakers (different ages, genders)",
        "🌍 Cover various topics (telecom, daily life, business)",
        "🔄 Include code-mixed Kinyarwanda-English speech",
        "📝 Ensure accurate transcriptions",
        "🎯 Aim for 100+ hours for good quality model",
        "📊 Use consistent audio format (16kHz, 16-bit WAV recommended)"
    ]
    
    for tip in tips:
        print(f"   {tip}")

def show_training_timeline():
    """Show expected training timeline"""
    print("\n⏱️  Expected Training Timeline:")
    print("=" * 35)
    
    scenarios = [
        {"data": "10 hours", "model": "tiny", "time": "2-4 hours", "quality": "Basic"},
        {"data": "50 hours", "model": "base", "time": "8-16 hours", "quality": "Good"},
        {"data": "100 hours", "model": "small", "time": "1-2 days", "quality": "Better"},
        {"data": "500 hours", "model": "medium", "time": "3-5 days", "quality": "Excellent"}
    ]
    
    for scenario in scenarios:
        print(f"   📊 {scenario['data']} data + {scenario['model']} model:")
        print(f"      ⏰ Time: {scenario['time']}")
        print(f"      🎯 Quality: {scenario['quality']}")
        print()

def main():
    """Main quick start function"""
    setup_training_environment()
    check_dependencies()
    show_data_collection_tips()
    show_training_timeline()
    
    print("\n🚀 Ready to Start Training!")
    print("=" * 30)
    print("1. Add your audio files to 'kinyarwanda_dataset/audio_files/'")
    print("2. Update 'kinyarwanda_dataset/metadata.json' with transcriptions")
    print("3. Run: python train_kinyarwanda_model.py")
    print("4. Monitor training progress in 'logs/' directory")
    print("5. Evaluate: python evaluate_model.py")
    print("6. Deploy: Update main.py to use your trained model")
    
    print("\n📖 For detailed guidance, see:")
    print("   - TRAINING_GUIDE.md")
    print("   - collect_training_data.py (for data management)")
    print("   - evaluate_model.py (for model evaluation)")

if __name__ == "__main__":
    main()
