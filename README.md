# 🌾 Agricultural BERT Classification System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JetPack 6.2](https://img.shields.io/badge/JetPack-6.2-green.svg)](https://developer.nvidia.com/embedded/jetpack)
[![CUDA 12.2](https://img.shields.io/badge/CUDA-12.2+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3.0-orange.svg)](https://pytorch.org/)
[![BERT](https://img.shields.io/badge/BERT-base%20%7C%20large-orange.svg)](https://huggingface.co/bert-base-uncased)

**Advanced expert chatbot on diseases in agricultural plants.** Jetson Orin Nano Super için optimize edilmiş, tarımsal hastalık ve crop management için gelişmiş BERT text classification sistemi.

## 🎯 Proje Özeti

Bu proje, **13,200 chunk tarımsal veri** kullanarak BERT modellerini eğitip tarımsal metinleri kategorize eden bir sistemdir:

- **🔬 13,200 İndekslenmiş Chunk:** PDF makaleler + sentetik tarımsal veriler
- **📊 6 Ana Kategori:** Plant Disease, Crop Management, Plant Genetics, Environmental Factors, Food Security, Technology
- **🤖 Dual Model Support:** BERT-base ve BERT-large
- **⚡ JetPack 6.2 Optimized:** Jetson Orin Nano Super için özel optimizasyonlar
- **💾 Auto Dataset Generation:** İndekslenmiş verilerden otomatik dataset üretimi

## 🚀 Jetson Orin Nano Super Deployment

### SSH Bağlantısı
```bash
ssh jetson-super@10.147.19.180
```

### Hızlı Kurulum
```bash
# Repository clone
git clone https://github.com/Mertcan-Gelbal/LLM-Chatbot.git
cd LLM-Chatbot

# JetPack 6.2 environment setup
chmod +x setup_jetson62.sh
./setup_jetson62.sh

# BERT dependencies
pip install -r requirements_bert_jetpack62.txt
```

### Tarımsal Dataset Oluşturma
```bash
# İndekslenmiş verilerden dataset oluştur
python3 agricultural_test_generator.py

# Sonuç: 
# - agricultural_datasets/train.csv (1,262 samples)
# - agricultural_datasets/val.csv (270 samples) 
# - agricultural_datasets/test.csv (271 samples)
# - agricultural_datasets/agricultural_sentiment.csv (780 samples)
```

### BERT Eğitimi Başlatma
```bash
cd jetson_training

# Tarımsal dataset ile eğitim
python3 bert_classification_trainer.py

# Sadece tarımsal veriler için
python3 -c "
from bert_classification_trainer import JetsonBERTTrainer
trainer = JetsonBERTTrainer()
trainer.run_agricultural_experiments()
"
```

## 📊 Dataset Kategorileri

| Kategori | Açıklama | Örnek Keywords |
|----------|----------|----------------|
| **Plant Disease** | Bitki hastalıkları | disease, pathogen, infection, blight, virus |
| **Crop Management** | Mahsul yönetimi | fertilizer, irrigation, planting, harvest |
| **Plant Genetics** | Bitki genetiği | variety, gene, resistance, hybrid |
| **Environmental Factors** | Çevre faktörleri | climate, weather, drought, temperature |
| **Food Security** | Gıda güvenliği | nutrition, supply, production, access |
| **Technology** | Tarım teknolojisi | AI, sensor, automation, precision |

## 🎯 Performance Expectations (Jetson Orin Nano Super)

### BERT-base-uncased
- **Training Time:** 15-20 dakika (3 epochs)
- **Memory Usage:** ~3GB GPU
- **Expected Accuracy:** 0.85-0.90

### BERT-large-uncased  
- **Training Time:** 25-35 dakika (3 epochs)
- **Memory Usage:** ~5GB GPU
- **Expected Accuracy:** 0.87-0.92

## 📁 Proje Yapısı

```
LLM-Chatbot/
├── agricultural_test_generator.py      # Dataset oluşturucu
├── agricultural_datasets/             # Oluşturulan datasets
│   ├── train.csv                      # Training data
│   ├── val.csv                        # Validation data
│   ├── test.csv                       # Test data
│   └── agricultural_sentiment.csv     # Sentiment data
├── jetson_training/                   # BERT training scripts
│   ├── bert_classification_trainer.py # Ana BERT trainer
│   ├── gpu_optimizer_jp62.py          # JetPack 6.2 optimizer
│   └── full_performance_trainer.py    # TAM performans trainer
├── final_system/                      # İndekslenmiş veriler
│   └── complete_index/                # 13,200 chunk veriler
├── setup_jetson62.sh                  # JetPack 6.2 kurulum
├── requirements_bert_jetpack62.txt     # BERT requirements
└── README_BERT.md                     # Detaylı BERT docs
```

## 🔧 JetPack 6.2 Optimizasyonları

- **CUDA 12.2** acceleration
- **PyTorch 2.3** compilation  
- **Mixed Precision** (FP16) training
- **Flash Attention** support
- **Memory-efficient** inference
- **TensorRT 10.x** integration

## 📈 Monitoring

```bash
# GPU monitoring
nvidia-smi -l 1

# System stats
tegrastats

# Training progress
tail -f results/training.log
```

## 🌾 Agricultural Use Cases

1. **Hastalık Teşhisi:** "leaf spots on tomato" → Plant Disease
2. **Mahsul Planlaması:** "irrigation schedule optimization" → Crop Management  
3. **Çeşit Seçimi:** "drought resistant wheat varieties" → Plant Genetics
4. **İklim Adaptasyonu:** "climate change impact on crops" → Environmental Factors
5. **Teknoloji Entegrasyonu:** "AI-powered crop monitoring" → Technology

## 🚨 Troubleshooting

### Memory Issues
```bash
# OOM hatası için batch size küçült
# bert_classification_trainer.py içinde batch_size = 4 yapın
```

### SSH Connection Issues
```bash
# Jetson cihaz IP kontrolü
ping 10.147.19.180

# SSH key setup
ssh-keygen -t rsa
ssh-copy-id jetson-super@10.147.19.180
```

## 📞 Support & Documentation

- **BERT Details:** [README_BERT.md](README_BERT.md)
- **Deployment Guide:** [JETSON_ORIN_SUPER_DEPLOYMENT.md](JETSON_ORIN_SUPER_DEPLOYMENT.md)
- **GitHub Issues:** Repository'de issue açın

---

**🤖 Advanced Agricultural BERT Classification - Production Ready on Jetson Orin Nano Super!** 🚀

### Quick Commands
```bash
# Complete setup and training on Jetson
ssh jetson-super@10.147.19.180
git clone https://github.com/Mertcan-Gelbal/LLM-Chatbot.git
cd LLM-Chatbot && ./setup_jetson62.sh
python3 agricultural_test_generator.py
cd jetson_training && python3 bert_classification_trainer.py
```
