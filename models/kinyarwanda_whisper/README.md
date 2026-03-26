# Kinyarwanda Whisper Model

This directory is for a fine-tuned Whisper model specifically trained on Kinyarwanda speech data.

## How to Add Your Custom Model

1. **Fine-tune Whisper on Kinyarwanda data**:
   ```bash
   # Example training command
   whisper-finetune --model base --data kinyarwanda_dataset/ --output models/kinyarwanda_whisper/
   ```

2. **Place model files here**:
   - `model.bin` or `pytorch_model.bin`
   - `config.json`
   - `tokenizer.json`
   - Any other required model files

3. **Verify the setup**:
   ```python
   # Test the model loading
   from main import get_kinyarwanda_asr_model
   model = get_kinyarwanda_asr_model()
   print("Kinyarwanda model loaded successfully!")
   ```

## Current Status

Currently, this directory is empty. The system will fall back to the multilingual Whisper model with enhanced Kinyarwanda vocabulary and post-processing.

## Benefits of a Custom Model

- **Higher accuracy** for Kinyarwanda speech
- **Better recognition** of local accents and dialects
- **Improved handling** of code-mixed speech
- **Faster inference** for Kinyarwanda-only content

## Training Data Suggestions

For best results, include:
- **Conversational Kinyarwanda** (daily speech patterns)
- **Code-mixed examples** (Kinyarwanda + English)
- **Telecom-related vocabulary** (bundles, airtime, etc.)
- **Regional variations** if applicable
- **Different audio qualities** (phone recordings, etc.)

## Alternative: Use Existing Models

If you don't have a custom model, the system will automatically:
1. Use the multilingual Whisper model
2. Apply enhanced Kinyarwanda vocabulary
3. Perform post-processing corrections
4. Detect and handle code-mixing

This provides excellent results without requiring a custom model.
