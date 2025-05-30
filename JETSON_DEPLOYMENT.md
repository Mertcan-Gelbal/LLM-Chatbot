# 🚀 Jetson Deployment Guide

## Jetson Orin Nano Super Deployment

### Sistem Gereksinimleri
- JetPack 6.2
- CUDA 12.2
- PyTorch 2.3
- Python 3.10+

### Kurulum Adımları

1. **Repository Clone**
```bash
git clone https://github.com/Mertcan-Gelbal/LLM-Chatbot.git
cd LLM-Chatbot
```

2. **Environment Setup**
```bash
chmod +x setup_jetson62.sh
./setup_jetson62.sh
```

3. **Dependencies**
```bash
pip install -r requirements_bert_jetpack62.txt
```

### SSH Bağlantısı
```bash
ssh jetson-super@10.147.19.180
```

### Performance Monitoring
```bash
# GPU monitoring
nvidia-smi -l 1

# System stats
tegrastats
``` 