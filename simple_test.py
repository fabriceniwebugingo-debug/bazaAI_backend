#!/usr/bin/env python3
"""
Simple test for language filtering
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import validate_detected_language, detect_code_mixing

def main():
    print("🔍 Testing Language Filtering for Kinyarwanda/English Only")
    print("=" * 60)
    
    # Test language validation
    print("\n🧪 Language Validation:")
    unsupported = ['fr', 'de', 'es', 'zh', 'ja']
    for lang in unsupported:
        result = validate_detected_language(lang)
        print(f"✅ '{lang}' -> {result} (should be None)")
    
    # Test supported languages
    supported = ['en', 'rw', 'kin', 'sw']
    for lang in supported:
        result = validate_detected_language(lang)
        print(f"✅ '{lang}' -> {result} (should be 'en' or 'kin')")
    
    # Test code-mixing
    print("\n🧪 Code-Mixing Detection:")
    test_texts = [
        "murakoze bite mbega",
        "hello thank you", 
        "murakoze thank you",
        ""
    ]
    
    for text in test_texts:
        result = detect_code_mixing(text)
        print(f"✅ '{text}'")
        print(f"   Mixing: {result['has_mixing']}")
        print(f"   Kinyarwanda: {result['kin_ratio']:.1%}, English: {result['en_ratio']:.1%}")
        print(f"   Kin words count: {len(result['kin_words'])}")
        print(f"   En words count: {len(result['en_words'])}")
        print()
    
    print("✅ All tests completed successfully!")

if __name__ == "__main__":
    main()
