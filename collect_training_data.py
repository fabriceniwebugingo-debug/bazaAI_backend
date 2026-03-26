#!/usr/bin/env python3
"""
Kinyarwanda Training Data Collection Helper
Tools to collect, validate, and prepare training data
"""

import os
import json
import pandas as pd
from pathlib import Path
import librosa
import soundfile as sf
from typing import List, Dict, Any
import logging
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """Helper class for collecting and validating training data"""
    
    def __init__(self, data_dir: str = "kinyarwanda_dataset"):
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / "audio_files"
        self.metadata_file = self.data_dir / "metadata.json"
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def add_audio_sample(self, audio_path: str, transcription: str, 
                        speaker_id: str = None, metadata: Dict = None) -> bool:
        """Add a new audio sample to the dataset"""
        try:
            # Copy audio file to dataset directory
            audio_file = Path(audio_path)
            if not audio_file.exists():
                logger.error(f"Audio file not found: {audio_path}")
                return False
            
            # Generate unique filename
            file_hash = self.get_file_hash(audio_file)
            new_filename = f"sample_{file_hash[:8]}{audio_file.suffix}"
            new_audio_path = self.audio_dir / new_filename
            
            # Copy audio file
            import shutil
            shutil.copy2(audio_file, new_audio_path)
            
            # Validate audio
            if not self.validate_audio(new_audio_path):
                logger.error(f"Invalid audio file: {audio_path}")
                new_audio_path.unlink()  # Remove copied file
                return False
            
            # Create metadata entry
            entry = {
                "audio": f"audio_files/{new_filename}",
                "text": transcription.strip(),
                "duration": self.get_audio_duration(new_audio_path),
                "sample_rate": self.get_sample_rate(new_audio_path),
                "added_date": datetime.now().isoformat(),
                "speaker_id": speaker_id,
                "metadata": metadata or {}
            }
            
            # Add to metadata
            self.add_to_metadata(entry)
            
            logger.info(f"Added sample: {new_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding sample {audio_path}: {e}")
            return False
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file to avoid duplicates"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def validate_audio(self, audio_path: Path) -> bool:
        """Validate audio file"""
        try:
            # Try to load audio
            audio, sr = librosa.load(str(audio_path), sr=None)
            
            # Check duration (1-30 seconds)
            duration = len(audio) / sr
            if duration < 1.0 or duration > 30.0:
                logger.warning(f"Audio duration {duration:.2f}s not in range [1, 30]s")
                return False
            
            # Check sample rate
            if sr not in [16000, 22050, 44100, 48000]:
                logger.warning(f"Unusual sample rate: {sr}Hz")
            
            # Check for silence
            if np.max(np.abs(audio)) < 0.01:
                logger.warning("Audio appears to be silent")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating audio {audio_path}: {e}")
            return False
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds"""
        try:
            audio, sr = librosa.load(str(audio_path), sr=None)
            return len(audio) / sr
        except:
            return 0.0
    
    def get_sample_rate(self, audio_path: Path) -> int:
        """Get audio sample rate"""
        try:
            audio, sr = librosa.load(str(audio_path), sr=None)
            return sr
        except:
            return 0
    
    def add_to_metadata(self, entry: Dict):
        """Add entry to metadata file"""
        metadata = []
        
        # Load existing metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Check for duplicates
        existing_texts = [item["text"] for item in metadata]
        if entry["text"] in existing_texts:
            logger.warning(f"Duplicate transcription found: {entry['text'][:50]}...")
            return
        
        # Add new entry
        metadata.append(entry)
        
        # Save metadata
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadata updated. Total samples: {len(metadata)}")
    
    def import_from_csv(self, csv_path: str, audio_base_dir: str = None):
        """Import data from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            
            audio_base = Path(audio_base_dir) if audio_base_dir else Path.cwd()
            
            for _, row in df.iterrows():
                audio_path = audio_base / row["audio_path"]
                transcription = row["transcription"]
                speaker_id = row.get("speaker_id")
                
                self.add_audio_sample(str(audio_path), transcription, speaker_id)
            
            logger.info(f"Imported {len(df)} samples from {csv_path}")
            
        except Exception as e:
            logger.error(f"Error importing from CSV {csv_path}: {e}")
    
    def import_from_common_voice(self, manifest_path: str):
        """Import from Common Voice manifest"""
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    
                    audio_path = item["audio_filepath"]
                    transcription = item["text"]
                    speaker_id = item.get("client_id")
                    
                    self.add_audio_sample(audio_path, transcription, speaker_id)
            
            logger.info(f"Imported from Common Voice manifest: {manifest_path}")
            
        except Exception as e:
            logger.error(f"Error importing from Common Voice {manifest_path}: {e}")
    
    def validate_dataset(self) -> Dict:
        """Validate the entire dataset and return statistics"""
        if not self.metadata_file.exists():
            return {"error": "No metadata file found"}
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        stats = {
            "total_samples": len(metadata),
            "valid_audio": 0,
            "invalid_audio": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
            "unique_speakers": set(),
            "sample_rates": set(),
            "issues": []
        }
        
        for entry in metadata:
            audio_path = self.data_dir / entry["audio"]
            
            if audio_path.exists():
                if self.validate_audio(audio_path):
                    stats["valid_audio"] += 1
                    stats["total_duration"] += entry.get("duration", 0)
                    stats["sample_rates"].add(entry.get("sample_rate", 0))
                    
                    if entry.get("speaker_id"):
                        stats["unique_speakers"].add(entry["speaker_id"])
                else:
                    stats["invalid_audio"] += 1
                    stats["issues"].append(f"Invalid audio: {entry['audio']}")
            else:
                stats["invalid_audio"] += 1
                stats["issues"].append(f"Missing audio: {entry['audio']}")
        
        if stats["valid_audio"] > 0:
            stats["avg_duration"] = stats["total_duration"] / stats["valid_audio"]
        
        stats["unique_speakers"] = len(stats["unique_speakers"])
        stats["sample_rates"] = list(stats["sample_rates"])
        
        return stats
    
    def create_balanced_subset(self, output_dir: str, samples_per_speaker: int = 50):
        """Create a balanced subset of the dataset"""
        if not self.metadata_file.exists():
            logger.error("No metadata file found")
            return
        
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Group by speaker
        speaker_samples = {}
        for entry in metadata:
            speaker_id = entry.get("speaker_id", "unknown")
            if speaker_id not in speaker_samples:
                speaker_samples[speaker_id] = []
            speaker_samples[speaker_id].append(entry)
        
        # Create balanced subset
        balanced_metadata = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for speaker_id, samples in speaker_samples.items():
            # Randomly sample from each speaker
            if len(samples) > samples_per_speaker:
                import random
                random.shuffle(samples)
                samples = samples[:samples_per_speaker]
            
            balanced_metadata.extend(samples)
        
        # Save balanced metadata
        balanced_file = output_path / "metadata.json"
        with open(balanced_file, 'w', encoding='utf-8') as f:
            json.dump(balanced_metadata, f, indent=2, ensure_ascii=False)
        
        # Copy audio files
        audio_output_dir = output_path / "audio_files"
        audio_output_dir.mkdir(exist_ok=True)
        
        for entry in balanced_metadata:
            src_audio = self.data_dir / entry["audio"]
            dst_audio = audio_output_dir / Path(entry["audio"]).name
            
            if src_audio.exists():
                import shutil
                shutil.copy2(src_audio, dst_audio)
        
        logger.info(f"Created balanced subset: {output_dir}")
        logger.info(f"Total samples: {len(balanced_metadata)}")
        logger.info(f"Speakers: {len(speaker_samples)}")
    
    def export_statistics(self, output_file: str = "dataset_stats.json"):
        """Export dataset statistics"""
        stats = self.validate_dataset()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset statistics exported to {output_file}")
        return stats

def main():
    """Main function for data collection"""
    print("🇷🇼 Kinyarwanda Training Data Collection")
    print("=" * 50)
    
    collector = DataCollector()
    
    # Example usage
    print("\n📊 Dataset Statistics:")
    stats = collector.export_statistics()
    
    if "error" not in stats:
        print(f"Total samples: {stats['total_samples']}")
        print(f"Valid audio: {stats['valid_audio']}")
        print(f"Invalid audio: {stats['invalid_audio']}")
        print(f"Total duration: {stats['total_duration']:.2f} hours")
        print(f"Average duration: {stats['avg_duration']:.2f} seconds")
        print(f"Unique speakers: {stats['unique_speakers']}")
        
        if stats['issues']:
            print(f"\n⚠️  Issues found: {len(stats['issues'])}")
            for issue in stats['issues'][:5]:  # Show first 5
                print(f"   - {issue}")
    
    print("\n🎯 Next Steps:")
    print("1. Add audio samples using collector.add_audio_sample()")
    print("2. Import from CSV/CSV using collector.import_from_csv()")
    print("3. Validate dataset with collector.validate_dataset()")
    print("4. Create balanced subset with collector.create_balanced_subset()")
    print("5. Start training with python train_kinyarwanda_model.py")

if __name__ == "__main__":
    main()
