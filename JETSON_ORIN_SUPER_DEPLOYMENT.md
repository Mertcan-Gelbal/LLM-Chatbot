# 🚀 Jetson Orin Nano Super Deployment Guide

## 📋 Sistem Bilgileri

- **Cihaz:** Jetson Orin Nano Super
- **IP Address:** 10.147.19.180
- **Username:** jetson-super
- **OS:** JetPack 6.2 L4T 36.4.3
- **CUDA:** 12.2+
- **PyTorch:** 2.3.0

## 1️⃣ SSH Bağlantısı

### İlk Bağlantı
```bash
# SSH ile bağlan
ssh jetson-super@10.147.19.180

# SSH key setup (opsiyonel)
ssh-keygen -t rsa -b 4096
ssh-copy-id jetson-super@10.147.19.180
```

### Bağlantı Testi
```bash
# Ping test
ping 10.147.19.180

# SSH test
ssh jetson-super@10.147.19.180 "echo 'Connection successful'"
```

## 2️⃣ Proje Kurulumu

### Repository Clone
```bash
# SSH bağlantısı
ssh jetson-super@10.147.19.180

# Home directory'ye git
cd ~

# Project clone
git clone https://github.com/Mertcan-Gelbal/LLM-Chatbot.git

# Project directory'ye git
cd LLM-Chatbot
```

### JetPack 6.2 Environment Setup
```bash
# Permission ver
chmod +x setup_jetson62.sh

# Kurulum başlat (20-30 dakika)
./setup_jetson62.sh

# Progress takip et
tail -f setup.log
```

## 3️⃣ Sistem Optimizasyonu

### Performance Mode
```bash
# Maximum performance
sudo nvpmodel -m 0

# Jetson clocks aktif
sudo jetson_clocks

# Fan control
echo 100 | sudo tee /sys/devices/pwm-fan/target_pwm
```

### Memory Configuration
```bash
# CUDA memory config
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Add to .bashrc
echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512' >> ~/.bashrc
```

## 4️⃣ BERT Dependencies

### Python Packages
```bash
# BERT özel requirements
pip install -r requirements_bert_jetpack62.txt

# Verify installation
python3 -c "
import torch
import transformers
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
"
```

### GPU Verification
```bash
# GPU status
nvidia-smi

# CUDA test
python3 -c "
import torch
print(f'CUDA Device: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

## 5️⃣ Dataset Preparation

### Büyük Dosyaları Manuel Transferi

**⚠️ DİKKAT:** `final_system/complete_index/chunks/all_chunks.json` dosyası (50MB) GitHub'da ignore edilmiş. Manuel transfer gerekli.

#### Seçenek 1: SCP ile Transfer
```bash
# Local machine'den Jetson'a
scp final_system/complete_index/chunks/all_chunks.json jetson-super@10.147.19.180:~/LLM-Chatbot/final_system/complete_index/chunks/
```

#### Seçenek 2: Compressed Transfer
```bash
# Local'de sıkıştır
tar czf chunks_data.tar.gz final_system/complete_index/chunks/

# Transfer et
scp chunks_data.tar.gz jetson-super@10.147.19.180:~/

# Jetson'da açık
ssh jetson-super@10.147.19.180
cd ~/LLM-Chatbot
tar xzf ~/chunks_data.tar.gz
```

### Dataset Generation
```bash
# Jetson'da dataset oluştur
cd ~/LLM-Chatbot

# 13,200 chunk'tan dataset üret
python3 agricultural_test_generator.py

# Verify datasets
ls -la agricultural_datasets/
```

## 6️⃣ BERT Training

### Pre-training Check
```bash
cd jetson_training

# System status
python3 -c "
from gpu_optimizer_jp62 import JetsonOptimizerJP62
opt = JetsonOptimizerJP62()
opt.monitor_jetson_gpu()
"
```

### Training Start
```bash
# Tarımsal BERT eğitimi
python3 bert_classification_trainer.py

# Background training
nohup python3 bert_classification_trainer.py > training.log 2>&1 &

# Progress monitoring
tail -f training.log
```

### Sadece Agricultural Experiments
```bash
# Sadece tarımsal veriler
python3 -c "
from bert_classification_trainer import JetsonBERTTrainer
trainer = JetsonBERTTrainer(mixed_precision=True, max_length=128)
trainer.run_agricultural_experiments()
"
```

## 7️⃣ Monitoring & Performance

### Real-time Monitoring
```bash
# Terminal 1: GPU monitoring
nvidia-smi -l 1

# Terminal 2: System stats
tegrastats

# Terminal 3: Training progress
tail -f jetson_training/training.log
```

### Temperature Monitoring
```bash
# Temperature check
watch -n 1 'cat /sys/class/thermal/thermal_zone*/temp'

# Fan status
cat /sys/devices/pwm-fan/cur_pwm
```

### Performance Metrics
```bash
# GPU utilization
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv -l 1

# Power consumption
tegrastats | grep "POM_5V_GPU"
```

## 8️⃣ Results & Models

### Training Results
```bash
# Results location
ls -la results/

# Model checkpoints
ls -la models/

# Best model
ls -la models/best_*
```

### Model Inference Test
```bash
cd jetson_training

# Quick inference test
python3 -c "
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model
tokenizer = BertTokenizer.from_pretrained('../models/best_bert-base-uncased_categorized/')
model = BertForSequenceClassification.from_pretrained('../models/best_bert-base-uncased_categorized/')

# Test inference
text = 'Tomato leaves showing brown spots'
inputs = tokenizer(text, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(f'Predictions: {predictions}')
"
```

## 9️⃣ Troubleshooting

### Common Issues

#### Memory Issues
```bash
# OOM Error Fix
# Edit bert_classification_trainer.py
# Line ~370: batch_size = 4  # Reduce from 8

# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()"
```

#### SSH Connection Issues
```bash
# Connection timeout
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=2 jetson-super@10.147.19.180

# Permission denied
chmod 600 ~/.ssh/id_rsa
```

#### Dataset Missing
```bash
# Check file existence
ls -la final_system/complete_index/chunks/all_chunks.json

# File size check
du -sh final_system/complete_index/chunks/all_chunks.json
# Should be ~50MB
```

### Performance Issues
```bash
# Check thermal throttling
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq

# Reset to max performance
sudo nvpmodel -m 0
sudo jetson_clocks
```

## 🔟 Advanced Operations

### Model Export
```bash
cd jetson_training

# ONNX export
python3 export_onnx.py \
    --model_path ../models/best_bert-base-uncased_categorized/ \
    --output_path ../models/agricultural_bert.onnx
```

### Batch Processing
```bash
# Batch inference script
python3 -c "
import pandas as pd
from bert_classification_trainer import JetsonBERTTrainer

# Load test data
df = pd.read_csv('../agricultural_datasets/test.csv')

# Batch predictions
trainer = JetsonBERTTrainer()
# trainer.batch_predict(df['text'].tolist())
"
```

### Remote Development
```bash
# VSCode remote development
# Install Remote-SSH extension
# Connect to jetson-super@10.147.19.180
```

## 📊 Expected Performance Metrics

### Training Performance
- **BERT-base:** 15-20 dakika (3 epochs)
- **BERT-large:** 25-35 dakika (3 epochs)
- **Memory Usage:** 3-5GB GPU
- **Temperature:** <80°C

### Model Accuracy
- **Agricultural Categorized:** 0.85-0.90 accuracy
- **Sentiment Analysis:** 0.82-0.87 accuracy
- **Inference Time:** <100ms per sample

---

## 🚀 Quick Start Commands

```bash
# Complete deployment sequence
ssh jetson-super@10.147.19.180
git clone https://github.com/Mertcan-Gelbal/LLM-Chatbot.git
cd LLM-Chatbot
chmod +x setup_jetson62.sh && ./setup_jetson62.sh
pip install -r requirements_bert_jetpack62.txt
python3 agricultural_test_generator.py
cd jetson_training && python3 bert_classification_trainer.py
```

**⚡ Jetson Orin Nano Super Agricultural BERT - Production Ready!** 🌾 