#!/usr/bin/env python3
"""
Kinyarwanda Model Evaluation Script
Evaluate trained models on test datasets
"""

import os
import json
import torch
import whisper
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime
import jiwer  # For WER calculation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate trained Kinyarwanda models"""
    
    def __init__(self, model_path: str, test_data_dir: str = "test_dataset"):
        self.model_path = model_path
        self.test_data_dir = Path(test_data_dir)
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            logger.info(f"Loading model: {self.model_path}")
            self.model = whisper.load_model(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test dataset"""
        test_data = []
        
        # Try different formats
        metadata_file = self.test_data_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        else:
            # Auto-discover files
            for audio_file in self.test_data_dir.rglob("*.wav"):
                txt_file = audio_file.with_suffix(".txt")
                if txt_file.exists():
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    test_data.append({
                        "audio": str(audio_file),
                        "text": text
                    })
        
        logger.info(f"Loaded {len(test_data)} test samples")
        return test_data
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe a single audio file"""
        try:
            result = self.model.transcribe(
                audio_path,
                language="rw",  # Kinyarwanda
                initial_prompt="murakoze bite mbega"
            )
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            return ""
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate"""
        return jiwer.wer(reference, hypothesis)
    
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate"""
        return jiwer.cer(reference, hypothesis)
    
    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate the model on test dataset"""
        test_data = self.load_test_data()
        
        if not test_data:
            return {"error": "No test data found"}
        
        results = {
            "total_samples": len(test_data),
            "wer_scores": [],
            "cer_scores": [],
            "transcriptions": [],
            "evaluation_time": datetime.now().isoformat(),
            "model_path": self.model_path
        }
        
        logger.info("Starting evaluation...")
        
        for i, sample in enumerate(test_data):
            audio_path = sample["audio"]
            reference = sample["text"]
            
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                continue
            
            # Transcribe
            hypothesis = self.transcribe_audio(audio_path)
            
            # Calculate metrics
            wer = self.calculate_wer(reference, hypothesis)
            cer = self.calculate_cer(reference, hypothesis)
            
            results["wer_scores"].append(wer)
            results["cer_scores"].append(cer)
            results["transcriptions"].append({
                "audio": audio_path,
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": wer,
                "cer": cer
            })
            
            # Progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(test_data)} samples")
        
        # Calculate averages
        if results["wer_scores"]:
            results["avg_wer"] = np.mean(results["wer_scores"])
            results["avg_cer"] = np.mean(results["cer_scores"])
            results["median_wer"] = np.median(results["wer_scores"])
            results["median_cer"] = np.median(results["cer_scores"])
            results["std_wer"] = np.std(results["wer_scores"])
            results["std_cer"] = np.std(results["cer_scores"])
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results"""
        print("\n🇷🇼 Kinyarwanda Model Evaluation Results")
        print("=" * 50)
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print(f"📊 Evaluation Summary:")
        print(f"   Total samples: {results['total_samples']}")
        print(f"   Model: {results['model_path']}")
        print(f"   Date: {results['evaluation_time']}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"   Average WER: {results['avg_wer']:.4f}")
        print(f"   Average CER: {results['avg_cer']:.4f}")
        print(f"   Median WER: {results['median_wer']:.4f}")
        print(f"   Median CER: {results['median_cer']:.4f}")
        print(f"   WER Std Dev: {results['std_wer']:.4f}")
        print(f"   CER Std Dev: {results['std_cer']:.4f}")
        
        # Quality assessment
        wer = results['avg_wer']
        if wer < 0.10:
            quality = "Excellent"
        elif wer < 0.15:
            quality = "Good"
        elif wer < 0.25:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"\n🎯 Quality Assessment: {quality}")
        
        # Show worst examples
        print(f"\n🔍 Worst Transcriptions (Top 5):")
        sorted_transcriptions = sorted(
            results["transcriptions"], 
            key=lambda x: x["wer"], 
            reverse=True
        )
        
        for i, trans in enumerate(sorted_transcriptions[:5]):
            print(f"\n{i+1}. WER: {trans['wer']:.4f}")
            print(f"   Reference: {trans['reference']}")
            print(f"   Hypothesis: {trans['hypothesis']}")
    
    def save_results(self, results: Dict[str, Any], output_file: str = "evaluation_results.json"):
        """Save evaluation results to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
    
    def compare_models(self, model_paths: List[str]) -> Dict[str, Any]:
        """Compare multiple models"""
        comparison = {
            "models": {},
            "comparison_date": datetime.now().isoformat()
        }
        
        for model_path in model_paths:
            print(f"\n🔍 Evaluating model: {model_path}")
            self.model_path = model_path
            self.load_model()
            results = self.evaluate_model()
            comparison["models"][model_path] = results
        
        # Find best model
        best_model = None
        best_wer = float('inf')
        
        for model_path, results in comparison["models"].items():
            if "avg_wer" in results and results["avg_wer"] < best_wer:
                best_wer = results["avg_wer"]
                best_model = model_path
        
        comparison["best_model"] = best_model
        comparison["best_wer"] = best_wer
        
        return comparison

def main():
    """Main evaluation function"""
    print("🇷🇼 Kinyarwanda Model Evaluation")
    print("=" * 40)
    
    # Default model path (update with your trained model)
    model_path = "models/trained_kinyarwanda/whisper_kinyarwanda_base.pt"
    test_data_dir = "test_dataset"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please update the model_path in the script")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path, test_data_dir)
    
    # Run evaluation
    results = evaluator.evaluate_model()
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    evaluator.save_results(results)
    
    print(f"\n✅ Evaluation completed!")
    print(f"📁 Results saved to: evaluation_results.json")

if __name__ == "__main__":
    main()
