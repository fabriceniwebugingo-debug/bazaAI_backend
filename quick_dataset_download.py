#!/usr/bin/env python3
"""
Quick download of mbazaNLP Kinyarwanda dataset sample
"""

import requests
import json
from pathlib import Path

def download_sample_dataset():
    """Download a small sample of the dataset"""
    print("🇷🇼 Downloading mbazaNLP Kinyarwanda Dataset Sample")
    print("=" * 50)
    
    # Get sample data
    url = "https://datasets-server.huggingface.co/rows?dataset=mbazaNLP%2Fcommon-voice-kinyarwanda-english-dataset&config=default&split=train&offset=0&length=10"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        print(f"📊 Dataset info:")
        print(f"   Total rows: {data.get('num_rows_total', 'Unknown')}")
        print(f"   Sample rows: {len(data.get('rows', []))}")
        
        # Create directories
        Path("kinyarwanda_dataset/audio_files").mkdir(parents=True, exist_ok=True)
        
        # Process sample rows
        metadata = []
        
        for i, row in enumerate(data.get("rows", [])):
            row_data = row["row"]
            
            # Create metadata entry (without downloading audio for now)
            entry = {
                "audio": f"audio_files/sample_{i:06d}.wav",
                "text": row_data.get("text", "").strip(),
                "duration": row_data.get("duration", 0),
                "source": "mbazaNLP",
                "row_index": row["row_idx"],
                "original_audio_path": row_data.get("audio_filepath", ""),
                "language": "mixed"
            }
            
            metadata.append(entry)
            
            print(f"\nSample {i+1}:")
            print(f"  Text: {entry['text'][:100]}...")
            print(f"  Duration: {entry['duration']:.2f}s")
            print(f"  Original: {entry['original_audio_path']}")
        
        # Save metadata
        with open("kinyarwanda_dataset/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Sample metadata created!")
        print(f"📁 File: kinyarwanda_dataset/metadata.json")
        print(f"📊 Samples: {len(metadata)}")
        
        print(f"\n🎯 Next steps:")
        print(f"   1. Run: python download_mbaza_dataset.py (for full download)")
        print(f"   2. Or use this sample to test training pipeline")
        print(f"   3. Run: python train_kinyarwanda_model.py")
        
        return metadata
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    download_sample_dataset()
