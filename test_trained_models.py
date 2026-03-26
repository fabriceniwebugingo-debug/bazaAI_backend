#!/usr/bin/env python3
"""
Test the enhanced Kinyarwanda model loading system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import load_model_config, get_kinyarwanda_trained_model, get_kinyarwanda_asr_model

def test_model_loading():
    """Test the enhanced model loading system"""
    print("🇷🇼 Testing Enhanced Kinyarwanda Model Loading")
    print("=" * 60)
    
    # Test model config loading
    print("\n📋 Testing Model Configuration:")
    load_model_config()
    from main import model_config
    
    if model_config:
        print(f"✅ Model config loaded successfully")
        print(f"   Available models: {model_config.get('available_models', [])}")
        print(f"   Default model: {model_config.get('default_model', 'None')}")
        print(f"   Model paths: {len(model_config.get('model_paths', {}))}")
    else:
        print("❌ No model config found (expected if no models downloaded)")
    
    # Test trained model loading
    print("\n🤖 Testing Trained Model Loading:")
    try:
        model = get_kinyarwanda_trained_model()
        if model:
            print(f"✅ Trained Kinyarwanda model loaded: {type(model)}")
        else:
            print("❌ No trained Kinyarwanda models available")
    except Exception as e:
        print(f"❌ Error loading trained model: {e}")
    
    # Test fallback to standard Whisper
    print("\n🔄 Testing Fallback to Standard Whisper:")
    try:
        model = get_kinyarwanda_asr_model()
        print(f"✅ Model loaded (fallback): {type(model)}")
        print(f"   This is standard Whisper with Kinyarwanda prompting")
    except Exception as e:
        print(f"❌ Error loading fallback model: {e}")
    
    print("\n📊 Expected Behavior:")
    print("• If trained models exist: Uses trained Kinyarwanda model")
    print("• If no trained models: Falls back to standard Whisper")
    print("• Automatic model selection based on availability")
    print("• Caching prevents reloading models")
    
    print("\n🎯 Next Steps:")
    print("1. Find actual Kinyarwanda models online")
    print("2. Update URLs in download_kinyarwanda_models.py")
    print("3. Run the downloader to get trained models")
    print("4. Test improved transcription accuracy")

if __name__ == "__main__":
    test_model_loading()
