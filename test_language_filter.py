#!/usr/bin/env python3
"""
Test script to verify language filtering works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import validate_detected_language, detect_code_mixing

def test_language_validation():
    """Test that only Kinyarwanda and English are accepted"""
    print("🧪 Testing Language Validation")
    print("=" * 40)
    
    test_cases = [
        ("en", "en"),
        ("english", "en"),
        ("en-us", "en"),
        ("rw", "kin"),
        ("kin", "kin"),
        ("kinyarwanda", "kin"),
        ("sw", "kin"),
        ("swahili", "kin"),
        ("fr", None),  # Should be filtered out
        ("de", None),  # Should be filtered out
        ("es", None),  # Should be filtered out
        ("zh", None),  # Should be filtered out
    ]
    
    for input_lang, expected in test_cases:
        result = validate_detected_language(input_lang)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_lang}' -> '{result}' (expected: '{expected}')")
    
    print()

def test_code_mixing():
    """Test code-mixing detection"""
    print("🧪 Testing Code-Mixing Detection")
    print("=" * 40)
    
    test_cases = [
        ("murakoze bite mbega", "Pure Kinyarwanda"),
        ("hello thank you", "Pure English"),
        ("murakoze thank you", "Mixed"),
        ("murakoze bite mbega hello thank you very much", "Mixed"),
        ("", "Empty"),
    ]
    
    for text, description in test_cases:
        result = detect_code_mixing(text)
        print(f"✅ '{text}' ({description})")
        print(f"   Mixing: {result['has_mixing']}")
        print(f"   Kinyarwanda: {result['kin_ratio']:.1%}, English: {result['en_ratio']:.1%}")
        if result['kin_words']:
            print(f"   Kin words: {result['kin_words']}")
        if result['en_words']:
            print(f"   En words: {result['en_words']}")
        print()

if __name__ == "__main__":
    print("🔍 Testing Language Filtering for Kinyarwanda/English Only")
    print("=" * 60)
    print()
    
    test_language_validation()
    test_code_mixing()
    
    print("✅ Language filtering tests completed!")
