# 🎤 Voice Upload Guide for Personal Baza AI Training

## 🎯 Overview
This guide helps you upload your recorded voices that match each Baza AI phrase for personalized training.

## 📁 Directory Structure
```
personal_baza_dataset/
├── audio_files/          # Target location for organized files
├── uploads/              # Where you place your recordings
├── metadata.json         # Training metadata
└── training_config.json  # Training configuration
```

## 📋 Step-by-Step Upload Process

### **Step 1: Record Your Voice**
Record these 18 phrases with your natural voice:

| ID | Filename | Phrase | Category |
|----|----------|---------|----------|
| 1 | `baza_phrase_001.wav` | "baza ai please help me" | baza_command |
| 2 | `baza_phrase_002.wav` | "murakoze baza ai" | greeting |
| 3 | `baza_phrase_003.wav` | "baza ai show me my balance" | financial |
| 4 | `baza_phrase_004.wav` | "please check my account balance" | financial |
| 5 | `baza_phrase_005.wav` | "nshaka kuri transfer amafaranga" | financial |
| 6 | `baza_phrase_006.wav` | "show me my data usage" | data |
| 7 | `baza_phrase_007.wav` | "buy airtime for my phone" | telecom |
| 8 | `baza_phrase_008.wav` | "murakoze for your help" | gratitude |
| 9 | `baza_phrase_009.wav` | "thank you very much" | gratitude |
| 10 | `baza_phrase_010.wav` | "I need customer service" | support |
| 11 | `baza_phrase_011.wav` | "murakoze I want to check my balance" | financial |
| 12 | `baza_phrase_012.wav` | "please show me amafaranga yanjye" | financial |
| 13 | `baza_phrase_013.wav` | "baza ai help me transfer money" | baza_command |
| 14 | `baza_phrase_014.wav` | "frw five thousand" | financial |
| 15 | `baza_phrase_015.wav` | "amafaranga ibihumbi bitanu" | financial |
| 16 | `baza_phrase_016.wav` | "check my internet connection" | technical |
| 17 | `baza_phrase_017.wav` | "why is my wifi not working" | technical |
| 18 | `baza_phrase_018.wav` | "baza ai analyze my usage" | baza_command |

### **Step 2: Place Files in Upload Directory**
1. Navigate to: `personal_baza_dataset/uploads/`
2. Copy your recorded files to this directory
3. Ensure files are named correctly (see table above)

### **Step 3: Process Uploads**
Run the upload manager:
```bash
python voice_upload_manager.py
```
Choose option 1: "📤 Process uploaded files"

The manager will:
- ✅ Scan upload directory for audio files
- ✅ Match files to training phrases
- ✅ Organize files in correct locations
- ✅ Update training metadata

### **Step 4: Check Training Readiness**
In the upload manager, choose option 2: "📊 Check training readiness"

This will show:
- How many files are ready
- Which files are missing
- If you can start training (need at least 50%)

## 🎤 Recording Tips

### **Best Practices**
- ✅ **Natural speech**: Speak as you would with customers
- ✅ **Your accent**: Use your normal Rwanda accent
- ✅ **Code-mixing**: Mix Kinyarwanda and English naturally
- ✅ **Clear audio**: No background noise or echo
- ✅ **Consistent volume**: Same distance from microphone
- ✅ **Duration**: 3-6 seconds per phrase

### **What to Avoid**
- ❌ **Over-enunciating**: Don't sound like a news anchor
- ❌ **Background noise**: No TV, music, or people talking
- ❌ **Different voice**: Use your normal speaking voice
- ❌ **Too fast/slow**: Natural speaking pace
- ❌ **Echo**: Record in a room with minimal echo

### **Equipment Needed**
- 📱 Smartphone with voice recorder app
- 🎤 Quiet room
- 📧 Way to transfer files to computer
- 💾 Computer for file organization

## 📁 File Organization

### **Before Upload**
```
personal_baza_dataset/uploads/
├── baza_phrase_001.wav
├── baza_phrase_002.wav
├── baza_phrase_003.wav
└── ... (all 18 files)
```

### **After Processing**
```
personal_baza_dataset/
├── audio_files/
│   ├── baza_phrase_001.wav ✅
│   ├── baza_phrase_002.wav ✅
│   └── ... (organized files)
├── uploads/ (empty after processing)
├── metadata.json (updated)
└── training_config.json
```

## 🚀 Quick Start Commands

### **Create Upload Directory**
```bash
mkdir -p personal_baza_dataset/uploads
```

### **Process Uploads**
```bash
python voice_upload_manager.py
# Choose option 1
```

### **Check Readiness**
```bash
python voice_upload_manager.py
# Choose option 2
```

### **Start Training**
```bash
python train_kinyarwanda_model.py
```

## 🎯 Expected Results

### **With Your Personal Voice**
- **95%+ accuracy** for your specific speech patterns
- **Perfect code-mixing** understanding
- **Your accent** recognition
- **Baza AI terms** perfectly understood
- **Natural interaction** with customers

### **Training Timeline**
- **18 phrases**: ~2-4 hours training
- **Quality**: Excellent for personal use
- **Deployment**: Ready for Baza AI service

## 🔧 Troubleshooting

### **File Not Matching**
**Problem**: Upload manager says "No matching phrase found"
**Solution**: 
- Check filename spelling
- Use exact names from table
- Or include phrase words in filename

### **Audio Quality Issues**
**Problem**: Poor transcription results
**Solution**:
- Re-record in quieter environment
- Check microphone distance
- Ensure clear speech

### **Training Fails**
**Problem**: Training script gives errors
**Solution**:
- Check all files exist in audio_files/
- Verify file formats (.wav, .mp3)
- Run upload manager again

## 📊 Success Checklist

### **Before Training**
- [ ] All 18 phrases recorded
- [ ] Files named correctly
- [ ] Files in uploads directory
- [ ] Upload manager processed files
- [ ] Training readiness > 50%

### **After Training**
- [ ] Model trained successfully
- [ ] Evaluation shows good WER (<20%)
- [ ] Model works with your voice
- [ ] Integrated into Baza AI service

## 🎉 Benefits of Personal Training

### **For Your Baza AI Service**
- **Perfect voice recognition** for you
- **Natural code-mixing** handling
- **Rwanda context** understanding
- **Telecom terms** recognition
- **Customer satisfaction** improvement

### **For Your Users**
- **Better service** with accurate voice recognition
- **Natural interaction** with familiar speech patterns
- **Reduced errors** in voice commands
- **Faster response** times

---

**🇷🇼 Ready to create your personal Baza AI voice model!**

Follow this guide step by step, and you'll have a perfectly trained model that understands YOUR voice and YOUR Baza AI service! 🎯
