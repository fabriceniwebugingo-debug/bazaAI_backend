#!/usr/bin/env python3
"""
Test improved Kinyarwanda/English ASR with enhanced prompts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import transcribe_with_code_mixing, post_process_transcription

def test_improved_transcription():
    """Test the enhanced transcription capabilities"""
    print("🚀 Testing Enhanced Kinyarwanda/English ASR")
    print("=" * 60)
    
    # Test scenarios with expected improvements
    test_cases = [
        {
            "name": "Pure Kinyarwanda",
            "text": "murakoze bite mbega ndashaka kuri gura data",
            "expected_lang": "kin"
        },
        {
            "name": "Pure English", 
            "text": "hello thank you I want to buy data bundles",
            "expected_lang": "en"
        },
        {
            "name": "Code-Mixed",
            "text": "murakoze I want to show my balance",
            "expected_lang": "mixed"
        },
        {
            "name": "Telecom Context",
            "text": "please help me check my airtime account",
            "expected_lang": "en"
        },
        {
            "name": "Kinyarwanda Telecom",
            "text": "erekana amafaranga yanjye kuri konti yanjye",
            "expected_lang": "kin"
        }
    ]
    
    print("📝 Testing transcription improvements:")
    print("-" * 40)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   Input: {case['text']}")
        print(f"   Expected: {case['expected_lang']}")
        
        # Test post-processing
        processed = post_process_transcription(case['text'], {})
        print(f"   Processed: {processed}")
        
        # Test code-mixing detection
        mixing = transcribe_with_code_mixing.__globals__['detect_code_mixing'](case['text'])
        print(f"   Code-mixing: {mixing['has_mixing']}")
        print(f"   Kinyarwanda: {mixing['kin_ratio']:.1%}")
        print(f"   English: {mixing['en_ratio']:.1%}")
        
        if mixing['kin_words']:
            print(f"   Kinyarwanda words: {mixing['kin_words']}")
        if mixing['en_words']:
            print(f"   English words: {mixing['en_words']}")
    
    print("\n" + "=" * 60)
    print("✨ Key Improvements Made:")
    print("🎯 Enhanced prompts with telecom context")
    print("🎯 Language-specific temperature and beam settings")
    print("🎯 Expanded Kinyarwanda vocabulary (147+ words)")
    print("🎯 Telecom-specific corrections")
    print("🎯 Code-mixing detection and filtering")
    print("🎯 Multiple model approaches for best accuracy")
    
    print("\n📊 Expected Results:")
    print("• Better Kinyarwanda word recognition")
    print("• Improved code-mixing handling")
    print("• Telecom term accuracy")
    print("• Language filtering (only en/kin)")
    print("• Context-aware transcription")

if __name__ == "__main__":
    test_improved_transcription()
