#!/usr/bin/env python3
"""
Simple training test to validate the pipeline works
"""

import os
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_minimal_dataset():
    """Create minimal dataset for testing"""
    print("🇷🇼 Creating Minimal Training Dataset")
    print("=" * 45)
    
    # Create directories
    dataset_dir = Path("kinyarwanda_dataset")
    audio_dir = dataset_dir / "audio_files"
    dataset_dir.mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    
    # Create minimal dataset with text only (no audio for now)
    minimal_data = [
        {
            "audio": "audio_files/placeholder_001.wav",
            "text": "murakoze bite mbega ndashaka kuri gura data bundles",
            "duration": 5.2,
            "language": "mixed"
        },
        {
            "audio": "audio_files/placeholder_002.wav", 
            "text": "mwaramutse nshaka kureka balance yanjye",
            "duration": 3.8,
            "language": "kin"
        },
        {
            "audio": "audio_files/placeholder_003.wav",
            "text": "please show me my account balance",
            "duration": 4.5,
            "language": "en"
        }
    ]
    
    # Save metadata
    metadata_file = dataset_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(minimal_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created minimal dataset: {metadata_file}")
    print(f"📊 Samples: {len(minimal_data)}")
    
    # Create placeholder audio files (empty for now)
    for i, entry in enumerate(minimal_data):
        audio_path = audio_dir / f"placeholder_{i+1:03d}.wav"
        audio_path.touch()  # Create empty file
        
    print(f"📁 Created {len(minimal_data)} placeholder audio files")
    
    return minimal_data

def test_training_pipeline():
    """Test if training pipeline can load the data"""
    print("\n🧪 Testing Training Pipeline")
    print("=" * 35)
    
    try:
        # Test dataset loading
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from train_kinyarwanda_model import KinyarwandaDataset, TrainingConfig
        
        config = TrainingConfig(
            model_size="tiny",  # Use tiny for fast testing
            batch_size=2,
            learning_rate=1e-5,
            epochs=1,  # Just 1 epoch for testing
            device="cpu"
        )
        
        dataset = KinyarwandaDataset("kinyarwanda_dataset", config)
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Test data loading
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Sample data: {sample}")
            print("🎯 Training pipeline is ready!")
        else:
            print("❌ No data in dataset")
            
    except Exception as e:
        print(f"❌ Error testing pipeline: {e}")
        return False
    
    return True

def show_next_steps():
    """Show what to do next"""
    print("\n🎯 Next Steps for Real Training")
    print("=" * 40)
    
    steps = [
        "1. Record actual audio files",
        "2. Replace placeholder files with real audio",
        "3. Ensure audio matches transcriptions",
        "4. Run: python train_kinyarwanda_model.py",
        "5. Evaluate: python evaluate_model.py",
        "6. Deploy to your telecom service"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print(f"\n📝 Audio Recording Guide:")
    print(f"   See: kinyarwanda_dataset/AUDIO_RECORDING_GUIDE.md")
    
    print(f"\n🚀 When Ready:")
    print(f"   python train_kinyarwanda_model.py")
    print(f"   # This will train on your real Kinyarwanda data")

def main():
    """Main function"""
    # Create minimal dataset
    dataset = create_minimal_dataset()
    
    # Test pipeline
    success = test_training_pipeline()
    
    if success:
        show_next_steps()
    else:
        print("\n❌ Pipeline test failed")
        print("Check the error messages above")

if __name__ == "__main__":
    import sys
    main()
