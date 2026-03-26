#!/usr/bin/env python3
"""
Test Professional English ASR with TTS
Demonstrate the professional English speech-to-text with accessibility features
"""

import requests
import json
import time
import base64
from pathlib import Path

class ProfessionalASRTester:
    """Test the professional English ASR system"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.test_audio_dir = Path("test_audio")
        self.test_audio_dir.mkdir(exist_ok=True)
    
    def test_health_check(self):
        """Test system health"""
        print("🏥 Testing Health Check")
        print("=" * 30)
        
        try:
            response = requests.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ System is healthy")
                print(f"   Whisper models: {data.get('whisper_models', [])}")
                print(f"   TTS available: {data.get('tts_available', False)}")
                print(f"   Speech recognition: {data.get('speech_recognition_available', False)}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_available_models(self):
        """Test available models endpoint"""
        print("\n🤖 Testing Available Models")
        print("=" * 35)
        
        try:
            response = requests.get(f"{self.base_url}/models")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Models endpoint working")
                print(f"   Whisper models: {data.get('whisper_models', [])}")
                print(f"   Recommended: {data.get('recommended', {})}")
                return True
            else:
                print(f"❌ Models endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Models endpoint error: {e}")
            return False
    
    def test_accessibility_info(self):
        """Test accessibility information"""
        print("\n♿ Testing Accessibility Info")
        print("=" * 40)
        
        try:
            response = requests.get(f"{self.base_url}/accessibility-info")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Accessibility info available")
                features = data.get('features', {})
                print(f"   Text-to-speech: {features.get('text_to_speech', False)}")
                print(f"   Screen reader: {features.get('screen_reader_compatible', False)}")
                print(f"   WCAG compliant: {data.get('accessibility_standards', [])}")
                return True
            else:
                print(f"❌ Accessibility info failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Accessibility info error: {e}")
            return False
    
    def test_text_to_speech(self):
        """Test text-to-speech functionality"""
        print("\n🔊 Testing Text-to-Speech")
        print("=" * 35)
        
        test_texts = [
            "Hello, this is a test of the text-to-speech system.",
            "Professional English ASR with accessibility features.",
            "This message helps visually impaired users access content."
        ]
        
        for i, text in enumerate(test_texts, 1):
            try:
                payload = {
                    "text": text,
                    "voice_speed": 150,
                    "volume": 0.9
                }
                
                response = requests.post(
                    f"{self.base_url}/text-to-speech",
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        print(f"✅ TTS Test {i}: Success")
                        print(f"   Text: {data.get('text', '')[:50]}...")
                        print(f"   Engine: {data.get('engine', 'unknown')}")
                        
                        # Save audio file for testing
                        if data.get('audio_data'):
                            audio_bytes = base64.b64decode(data['audio_data'])
                            audio_file = self.test_audio_dir / f"tts_test_{i}.wav"
                            with open(audio_file, 'wb') as f:
                                f.write(audio_bytes)
                            print(f"   Audio saved: {audio_file}")
                    else:
                        print(f"❌ TTS Test {i}: {data.get('error', 'Unknown error')}")
                else:
                    print(f"❌ TTS Test {i}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"❌ TTS Test {i}: {e}")
    
    def create_test_audio_file(self):
        """Create a simple test audio file (placeholder)"""
        # This would normally be a real audio file
        # For testing, we'll create a placeholder
        test_file = self.test_audio_dir / "test_english.wav"
        test_file.touch()  # Create empty file
        return str(test_file)
    
    def test_transcription(self):
        """Test transcription endpoint"""
        print("\n🎙️ Testing Transcription")
        print("=" * 35)
        
        # Create test audio file (in real usage, this would be actual audio)
        audio_file = self.create_test_audio_file()
        
        try:
            with open(audio_file, 'rb') as f:
                files = {'audio': f}
                data = {
                    'phone': '+1234567890',
                    'model_size': 'base',
                    'enable_tts': 'true'
                }
                
                response = requests.post(
                    f"{self.base_url}/transcribe",
                    files=files,
                    data=data
                )
            
            # Note: This will likely fail with empty audio file, but tests the endpoint
            if response.status_code == 200:
                result = response.json()
                print("✅ Transcription endpoint working")
                print(f"   Phone: {result.get('phone', '')}")
                print(f"   Model: {result.get('model_used', '')}")
                print(f"   Language: {result.get('language', '')}")
                
                if result.get('tts_audio'):
                    print("   TTS response included")
            else:
                print(f"⚠️  Transcription test: HTTP {response.status_code}")
                print("   (Expected with empty test audio file)")
                
        except Exception as e:
            print(f"❌ Transcription test error: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Professional English ASR System Tests")
        print("=" * 50)
        print(f"Testing server: {self.base_url}")
        print()
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Available Models", self.test_available_models),
            ("Accessibility Info", self.test_accessibility_info),
            ("Text-to-Speech", self.test_text_to_speech),
            ("Transcription", self.test_transcription)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                print(f"❌ {test_name} failed: {e}")
                results[test_name] = False
        
        # Summary
        print("\n📊 Test Summary")
        print("=" * 20)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! System is ready.")
        else:
            print("⚠️  Some tests failed. Check the server status.")
        
        return results

def main():
    """Main test function"""
    print("🎙️ Professional English ASR with TTS Test Suite")
    print("=" * 55)
    print("This tests the professional English speech-to-text system")
    print("with text-to-speech capabilities for accessibility.\n")
    
    # Check if server is running
    tester = ProfessionalASRTester()
    
    print("🔍 Checking if server is running...")
    try:
        response = requests.get(f"{tester.base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
            print()
            tester.run_all_tests()
        else:
            print("❌ Server responded with error")
            print("Please start the server first:")
            print("python professional_english_asr.py")
    except requests.exceptions.RequestException:
        print("❌ Server is not running")
        print("\nTo start the server:")
        print("1. Install dependencies: pip install -r requirements_professional.txt")
        print("2. Run server: python professional_english_asr.py")
        print("3. Then run tests: python test_professional_asr.py")

if __name__ == "__main__":
    main()
