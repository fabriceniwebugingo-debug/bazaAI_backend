# Kinyarwanda Voice Recognition & Code-Mixing Guide

## Overview

This enhanced voice recognition system provides superior Kinyarwanda transcription capabilities with automatic code-mixing detection between Kinyarwanda and English.

## Features

### 1. Enhanced Kinyarwanda Support
- **Multi-model approach**: Uses multiple Whisper models for best accuracy
- **Custom model loading**: Supports fine-tuned Kinyarwanda models
- **Post-processing**: Automatic correction of common transcription errors
- **Vocabulary enhancement**: Extended Kinyarwanda word recognition

### 2. Code-Mixing Detection
- **Automatic detection**: Identifies when users mix Kinyarwanda and English
- **Language ratios**: Calculates percentage of each language in speech
- **Mixed word identification**: Detects words used in both languages
- **Smart transcription**: Chooses best model based on content analysis

### 3. Multiple Model Sizes
- **Tiny**: Fastest, lower accuracy (good for real-time)
- **Base**: Balanced speed and accuracy (default)
- **Small**: Better accuracy, moderate speed
- **Medium**: High accuracy, slower processing
- **Large**: Best accuracy, slowest processing

## API Endpoints

### Enhanced Transcription
```http
POST /transcribe
Content-Type: multipart/form-data

Parameters:
- phone: User phone number
- language: auto|en|kin|mixed (default: auto)
- model_size: tiny|base|small|medium|large (default: base)
- enable_code_mixing: true|false (default: true)
- confidence_threshold: 0.0-1.0 (default: 0.5)
- audio: Audio file (.m4a, .mp3, .wav, .ogg, .flac, .webm)
```

### Batch Transcription
```http
POST /transcribe/batch
Content-Type: multipart/form-data

Parameters:
- phone: User phone number
- language: auto|en|kin|mixed
- model_size: tiny|base|small|medium|large
- enable_code_mixing: true|false
- audio_files: Multiple audio files (max 10)
```

## Response Format

### Enhanced Response (with code-mixing)
```json
{
  "text": "Murakoze ndashaka kuri show data bundles",
  "detected_language": "kin",
  "confidence": 0.85,
  "model_used": "kinyarwanda_specific",
  "code_mixing": {
    "has_mixing": true,
    "kin_ratio": 0.75,
    "en_ratio": 0.25,
    "kin_words": ["murakoze", "ndashaka", "kuri", "data"],
    "en_words": ["show", "bundles"],
    "mixed_words": ["data"],
    "total_words": 7
  },
  "all_results": [...]
}
```

### Standard Response
```json
{
  "text": "Murakoze ndashaka kuri show data bundles",
  "detected_language": "rw",
  "model_used": "whisper_base",
  "confidence": 0.80
}
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements_new.txt
```

### 2. Optional: Fine-tuned Model Setup
Create a directory structure for custom models:
```
models/
├── kinyarwanda_whisper/
│   ├── model.bin
│   └── config.json
```

Set environment variable:
```bash
export KINYARWANDA_ASR_MODEL_PATH="models/kinyarwanda_whisper"
```

### 3. Enhanced Vocabulary
Create `kin_vocab.txt` with additional Kinyarwanda words:
```
murakoze
ndashaka
kuri
data
bundles
erekana
gura
...
```

## Usage Examples

### Python Client Example
```python
import requests

# Enhanced transcription with code-mixing
files = {'audio': open('voice_message.m4a', 'rb')}
data = {
    'phone': '+250788123456',
    'language': 'auto',
    'model_size': 'base',
    'enable_code_mixing': True,
    'confidence_threshold': 0.6
}

response = requests.post('http://localhost:8000/transcribe', 
                       files=files, data=data)
result = response.json()

if result.get('code_mixing', {}).get('has_mixing'):
    print(f"Code-mixing detected: {result['code_mixing']['kin_ratio']:.1%} Kinyarwanda")
print(f"Transcription: {result['text']}")
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "phone=+250788123456" \
  -F "language=auto" \
  -F "model_size=base" \
  -F "enable_code_mixing=true" \
  -F "confidence_threshold=0.5" \
  -F "audio=@voice_message.m4a"
```

## Code-Mixing Analysis

### Detection Logic
The system detects code-mixing when:
- Kinyarwanda words ≥ 10% of total words
- English words ≥ 10% of total words
- Both conditions are met simultaneously

### Language Ratios
- **Pure Kinyarwanda**: kin_ratio ≥ 0.9, en_ratio ≤ 0.1
- **Pure English**: en_ratio ≥ 0.9, kin_ratio ≤ 0.1
- **Mixed**: Both ratios between 0.1 and 0.9

### Common Patterns Supported
- **Kinyarwanda-dominant**: "Murakoze I want to buy data bundles"
- **English-dominant**: "I want to show amafaranga yanjye"
- **Balanced mixing**: "Murakoze I want kuri show bundles"

## Post-Processing Corrections

The system automatically corrects common transcription errors:

| Misrecognized | Correct |
|---------------|---------|
| muri ko | murakoze |
| mira koze | murakoze |
| ndabwi | ndabizi |
| bike | bicyo |
| ng' | ng' |

## Performance Optimization

### Model Selection Guidelines
- **Real-time applications**: Use `tiny` or `base` models
- **High accuracy requirements**: Use `medium` or `large` models
- **Code-mixing scenarios**: Enable `enable_code_mixing=true`

### Confidence Thresholds
- **High accuracy**: 0.7-1.0
- **Balanced**: 0.5-0.7 (default)
- **Inclusive**: 0.3-0.5
- **Permissive**: 0.0-0.3

## Troubleshooting

### Common Issues

1. **Low confidence scores**
   - Try larger model size
   - Check audio quality
   - Ensure clear speech

2. **Incorrect language detection**
   - Use specific language instead of 'auto'
   - Check for background noise
   - Verify vocabulary coverage

3. **Code-mixing not detected**
   - Ensure `enable_code_mixing=true`
   - Check if both languages meet 10% threshold
   - Review vocabulary files

### Logging
Monitor these log messages:
- `Enhanced transcription request:` - Request details
- `Kinyarwanda model failed:` - Custom model issues
- `Low confidence transcription:` - Quality warnings
- `Enhanced transcription completed:` - Success confirmation

## Future Enhancements

### Planned Features
1. **Real-time streaming transcription**
2. **Custom vocabulary training**
3. **Regional dialect support**
4. **Speaker identification**
5. **Emotion detection**

### Model Training
For organizations with specific needs:
1. Collect Kinyarwanda audio datasets
2. Fine-tune Whisper models
3. Deploy custom models via `KINYARWANDA_ASR_MODEL_PATH`

## Support

For issues and enhancement requests:
1. Check logs for detailed error messages
2. Verify audio file format and quality
3. Test with different model sizes
4. Review vocabulary coverage

## License

This enhanced ASR system maintains the same license as the base telecom chat service. Additional models and datasets may have separate licensing requirements.
