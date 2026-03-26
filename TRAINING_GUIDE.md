# 🇷🇼 Kinyarwanda Model Training Guide

## 🎯 Overview

This guide helps you train a custom Kinyarwanda ASR model using your own dataset. The training pipeline supports multiple data formats and provides a complete end-to-end solution.

## 📁 Dataset Preparation

### **Required Data Structure**
```
kinyarwanda_dataset/
├── audio_files/
│   ├── sample1.wav
│   ├── sample2.mp3
│   └── sample3.flac
├── metadata.json          # OR data.csv OR manifest.json
└── README.md
```

### **Supported Data Formats**

#### **1. JSON Format (Recommended)**
```json
[
    {
        "audio": "audio_files/sample1.wav",
        "text": "murakoze bite mbega ndashaka kuri gura data bundles"
    },
    {
        "audio": "audio_files/sample2.wav", 
        "text": "mwaramutse nshaka kureka balance yanjye"
    }
]
```

#### **2. CSV Format**
```csv
audio_path,transcription
audio_files/sample1.wav,murakoze bite mbega ndashaka kuri gura data bundles
audio_files/sample2.wav,mwaramutse nshaka kureka balance yanjye
```

#### **3. Common Voice Manifest Format**
```json
{"audio_filepath": "audio_files/sample1.wav", "text": "murakoze bite mbega ndashaka kuri gura data bundles"}
{"audio_filepath": "audio_files/sample2.wav", "text": "mwaramutse nshaka kureka balance yanjye"}
```

#### **4. Auto-Discovery Format**
```
audio_files/
├── sample1.wav
├── sample1.txt     # Contains: murakoze bite mbega ndashaka kuri gura data bundles
├── sample2.wav
├── sample2.txt     # Contains: mwaramutse nshaka kureka balance yanjye
└── sample3.wav
└── sample3.json    # {"text": "yego ndabizi kuri transfer amafaranga"}
```

## 🎤 Audio Requirements

### **Audio Quality Guidelines**
- **Format**: WAV, MP3, FLAC, M4A
- **Sample Rate**: 16kHz (recommended)
- **Bit Depth**: 16-bit or higher
- **Duration**: 1-30 seconds per clip
- **Quality**: Clear speech, minimal background noise
- **Language**: Pure Kinyarwanda or Kinyarwanda-English code-mixing

### **Transcription Guidelines**
- **Accurate transcription**: Exact speech content
- **Kinyarwanda spelling**: Use proper Kinyarwanda orthography
- **Punctuation**: Include natural punctuation
- **Code-mixing**: Preserve mixed language as spoken
- **Case**: Use normal sentence case

### **Dataset Size Recommendations**
- **Minimum**: 100 hours for basic training
- **Good**: 500+ hours for production quality
- **Excellent**: 1000+ hours for state-of-the-art
- **For testing**: 10-50 hours to validate pipeline

## 🚀 Training Process

### **Step 1: Prepare Dataset**
```bash
# Create dataset directory
mkdir -p kinyarwanda_dataset/audio_files

# Add your audio files and create metadata
# Use one of the supported formats above
```

### **Step 2: Install Dependencies**
```bash
pip install torch torchaudio
pip install whisper
pip install librosa soundfile
pip install pandas numpy
pip install transformers datasets accelerate
```

### **Step 3: Configure Training**
Edit `train_kinyarwanda_model.py` configuration:

```python
config = TrainingConfig(
    model_size="base",        # tiny, base, small, medium, large
    batch_size=8,           # Adjust based on GPU memory
    learning_rate=1e-5,
    epochs=30,              # More epochs for larger datasets
    device="cuda"           # Use GPU if available
)
```

### **Step 4: Start Training**
```bash
python train_kinyarwanda_model.py
```

## ⚙️ Training Configuration

### **Model Sizes**
| Size | Parameters | VRAM Needed | Training Time | Quality |
|------|------------|-------------|---------------|---------|
| tiny | 39M | 2GB | Fast | Basic |
| base | 74M | 4GB | Medium | Good |
| small | 244M | 8GB | Slow | Better |
| medium | 769M | 16GB | Very Slow | Excellent |
| large | 1550M | 32GB | Extremely Slow | Best |

### **Training Parameters**
```python
TrainingConfig(
    model_size="base",        # Model complexity
    batch_size=8,           # Samples per batch (GPU memory dependent)
    learning_rate=1e-5,      # Learning rate
    epochs=30,               # Training iterations
    save_steps=500,          # Save checkpoint every N steps
    eval_steps=250,          # Evaluate every N steps
    max_audio_length=30,      # Max audio length in seconds
    device="cuda"            # Training device
)
```

### **Hardware Requirements**
- **Minimum**: 8GB RAM, 4GB VRAM (for base model)
- **Recommended**: 16GB RAM, 8GB VRAM (for small model)
- **Optimal**: 32GB RAM, 16GB VRAM (for medium model)

## 📊 Monitoring Training

### **Training Logs**
```
INFO:telecom-chat:Starting Kinyarwanda model training...
INFO:telecom-chat:Loaded 1000 training samples
INFO:telecom-chat:Training samples: 900
INFO:telecom-chat:Evaluation samples: 100
INFO:telecom-chat:Epoch 0, Step 100, Loss: 2.3456
INFO:telecom-chat:Epoch 0, Step 200, Loss: 2.1234
INFO:telecom-chat:Epoch 0, Eval Loss: 2.2345
```

### **Checkpoints**
- **Saved every**: `save_steps` iterations
- **Location**: `models/trained_kinyarwanda/checkpoint_step_*.pt`
- **Resume**: Can resume from any checkpoint

### **Final Model**
- **Location**: `models/trained_kinyarwanda/whisper_kinyarwanda_base.pt`
- **Metadata**: `models/trained_kinyarwanda/model_metadata.json`

## 🔧 Advanced Training Options

### **Transfer Learning**
```python
# Start from a pre-trained multilingual model
model = whisper.load_model("base")  # Multilingual base
# Fine-tune on Kinyarwanda data
```

### **Data Augmentation**
```python
# Add noise, speed changes, pitch shifts
# Implemented in dataset class
```

### **Multi-GPU Training**
```python
# Use DataParallel for multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## 📈 Expected Results

### **Training Progress**
- **Initial**: High loss (~3.0)
- **After 10 epochs**: Moderate loss (~1.5)
- **After 20 epochs**: Good loss (~0.8)
- **After 30 epochs**: Excellent loss (~0.3)

### **Quality Metrics**
- **Word Error Rate (WER)**: < 15% for good models
- **Character Error Rate (CER)**: < 10% for good models
- **BLEU Score**: > 0.6 for good models

## 🧪 Testing Your Model

### **Quick Test**
```python
import whisper

# Load your trained model
model = whisper.load_model("models/trained_kinyarwanda/whisper_kinyarwanda_base.pt")

# Test transcription
result = model.transcribe("test_audio.wav", language="rw")
print(f"Transcription: {result['text']}")
```

### **Evaluation Script**
```python
# Create evaluation script to test on held-out dataset
python evaluate_model.py --model_path models/trained_kinyarwanda/whisper_kinyarwanda_base.pt --test_data test_dataset/
```

## 🛠️ Troubleshooting

### **Common Issues**

#### **1. CUDA Out of Memory**
```python
# Reduce batch size
config.batch_size = 4  # Try smaller values

# Use smaller model
config.model_size = "tiny"  # or "base"
```

#### **2. Slow Training**
```python
# Increase batch size if GPU allows
config.batch_size = 16

# Use more workers
DataLoader(dataset, batch_size=16, num_workers=8)
```

#### **3. Poor Quality**
```python
# Increase training epochs
config.epochs = 50

# Lower learning rate
config.learning_rate = 5e-6

# Add more data
# Collect more audio samples
```

#### **4. Audio Loading Errors**
```python
# Check audio formats
# Ensure all audio files are supported formats

# Verify paths
# Make sure all audio paths in metadata are correct
```

## 📚 Data Collection Tips

### **Sources for Kinyarwanda Audio**
1. **Record your own**: Use phone/voice recorder
2. **Community contributions**: Ask Rwanda users
3. **Public datasets**: Mozilla Common Voice Kinyarwanda
4. **Radio/TV broadcasts**: Extract speech segments
5. **Call center recordings**: Anonymized customer calls

### **Quality Assurance**
- **Review transcriptions**: Ensure accuracy
- **Remove duplicates**: Avoid repeated samples
- **Balance content**: Mix of topics and speakers
- **Quality control**: Remove low-quality audio

## 🎯 Next Steps

### **After Training**
1. **Test model**: Evaluate on held-out test set
2. **Integration**: Update main.py to use trained model
3. **Deployment**: Deploy to production
4. **Monitoring**: Track performance in production

### **Continuous Improvement**
1. **Collect more data**: Ongoing data collection
2. **Fine-tune**: Periodic retraining
3. **A/B testing**: Compare with baseline
4. **User feedback**: Collect and incorporate feedback

---

**🎉 Success**: After completing this guide, you'll have a custom-trained Kinyarwanda ASR model that understands your specific use case and data much better than generic models! 🇷🇼
