
# 🎤 Audio Recording Guide for Kinyarwanda Training

## 📋 What You Need to Record

### **Required Equipment**
- 📱 Smartphone or voice recorder
- 🎤 Quiet environment
- 🎧 Good microphone
- 💾 Audio recording app

### **Recording Guidelines**
- **Duration**: 3-15 seconds per clip
- **Quality**: Clear speech, minimal background noise
- **Format**: WAV, 16kHz, 16-bit (recommended)
- **Content**: Natural speech as used in daily life

### **Phrases to Record**
Use the transcriptions from your dataset. Examples:

#### **Kinyarwanda Phrases**
1. "murakoze bite mbega ndashaka kuri gura data bundles"
2. "mwaramutse nshaka kureka balance yanjye"
3. "yego ndabizi kuri transfer amafaranga"

#### **English Phrases**
1. "hello thank you for calling"
2. "I would like to check my balance"
3. "can I get more data bundles"

#### **Code-Mixed Phrases**
1. "murakoze I want to check my balance"
2. "please show me amafaranga yanjye"
3. "I need data bundles kuri telefoni yanjye"

### **Recording Process**
1. Open voice recording app
2. Set format: WAV, 16kHz, 16-bit
3. Record each phrase clearly
4. Save as: sample_000000.wav, sample_000001.wav, etc.
5. Place files in: kinyarwanda_dataset/audio_files/

### **Quality Check**
- ✅ Clear speech, no mumbling
- ✅ Consistent volume throughout
- ✅ No background noise or echo
- ✅ Natural speaking pace
- ✅ Proper pronunciation

### **Next Steps**
1. Record all audio files
2. Verify they match the transcriptions
3. Run: python train_kinyarwanda_model.py
4. Evaluate: python evaluate_model.py

## 🎯 Tips for Best Results

- **Multiple speakers**: Record different people (age, gender)
- **Natural speech**: Speak normally, don't over-enunciate
- **Environment**: Quiet room, no TV/music in background
- **Consistency**: Same recording setup for all samples
- **Review**: Listen back to check quality

## 📞 Telecom-Specific Content

Focus on phrases your customers actually use:
- Balance inquiries
- Data bundle purchases
- Money transfers
- Account information
- Customer service requests
- Technical support

This will ensure your trained model works perfectly for your Rwanda telecom service! 🇷🇼
