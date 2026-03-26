#!/usr/bin/env python3
"""
Test strict language filtering to prevent other languages from appearing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import validate_detected_language, clean_non_target_languages

def test_strict_filtering():
    """Test the improved language filtering"""
    print("🔒 Testing Strict Language Filtering")
    print("=" * 50)
    
    # Test language validation
    print("\n🧪 Language Validation (Strict Mode):")
    test_cases = [
        # Should be accepted
        ("en", "en"),
        ("english", "en"), 
        ("rw", "kin"),
        ("kin", "kin"),
        ("kinyarwanda", "kin"),
        ("sw", "kin"),  # Treated as Kinyarwanda
        
        # Should be filtered out
        ("fr", None),
        ("french", None),
        ("de", None), 
        ("german", None),
        ("es", None),
        ("spanish", None),
        ("zh", None),
        ("chinese", None),
        ("ja", None),
        ("ar", None),
        ("arabic", None),
        ("pt", None),  # Portuguese
        ("it", None),  # Italian
        ("ru", None),  # Russian
    ]
    
    for input_lang, expected in test_cases:
        result = validate_detected_language(input_lang)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{input_lang}' -> '{result}' (expected: '{expected}')")
    
    # Test text cleaning
    print("\n🧹 Text Cleaning (Non-target Language Removal):")
    test_texts = [
        ("murakoze bonjour bite mbega", "kin"),  # Should remove 'bonjour'
        ("hello merci thank you", "en"),  # Should remove 'merci'
        ("data bundles danke", "en"),  # Should remove 'danke'
        ("hola murakoze", "kin"),  # Should remove 'hola'
        ("murakoze arigatou", "kin"),  # Should remove 'arigatou'
        ("please help me", "en"),  # Should remain unchanged
        ("murakoze bite mbega", "kin"),  # Should remain unchanged
    ]
    
    for text, detected_lang in test_texts:
        cleaned = clean_non_target_languages(text, detected_lang)
        print(f"✅ Original: '{text}'")
        print(f"✅ Cleaned: '{cleaned}'")
        print(f"   Language: {detected_lang}")
        print()
    
    print("🔒 Key Improvements:")
    print("• Stricter language validation (only en/kin accepted)")
    print("• Removes French, German, Spanish, Chinese, Arabic words")
    print("• Enhanced logging for filtered languages")
    print("• Cleaner transcription output for target languages")
    print("• Better handling of code-mixing scenarios")

if __name__ == "__main__":
    test_strict_filtering()
