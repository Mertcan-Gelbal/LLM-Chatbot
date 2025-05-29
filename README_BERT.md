# 🤖 BERT Classification - JetPack 6.2 Optimized

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JetPack 6.2](https://img.shields.io/badge/JetPack-6.2-green.svg)](https://developer.nvidia.com/embedded/jetpack)
[![CUDA 12.2](https://img.shields.io/badge/CUDA-12.2+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3.0-orange.svg)](https://pytorch.org/)
[![BERT](https://img.shields.io/badge/BERT-base%20%7C%20large-orange.svg)](https://huggingface.co/bert-base-uncased)

**JetPack 6.2 L4T 36.4.3** üzerinde optimize edilmiş BERT text classification sistemi. Custom dataset ve AG News dataset ile çoklu model eğitimi.

## 🎯 Özellikler

- **🤖 Dual BERT Models:** BERT-base ve BERT-large desteği
- **📊 Dual Datasets:** Custom dataset + AG News (120K+ samples)
- **⚡ JetPack 6.2 Optimization:** CUDA 12.2, PyTorch 2.3.0
- **🔧 Mixed Precision:** FP16 hızlandırma
- **📈 Advanced Metrics:** Accuracy, Precision, F1-Score, Classification Report
- **🎨 Visualization:** Karşılaştırmalı grafik analizi
- **💾 Auto Model Saving:** En iyi model otomatik kayıt

## 📊 Desteklenen Datasets

### 1. Custom Dataset (veri.csv)
- **Format:** `text,label`
- **Örnek:** Sentiment analysis (positive/negative)
- **Otomatik preprocessing:** Label encoding, train/val/test split

### 2. AG News Dataset
- **Kategoriler:** World, Sports, Business, Sci/Tech  
- **Size:** 120,000 training + 7,600 test samples
- **Otomatik download:** Internet'ten direkt yükleme

## 🚀 JetPack 6.2 Hızlı Başlangıç

### Kurulum
```bash
# Repository clone
git clone https://github.com/[USERNAME]/bert-classification-jetson.git
cd bert-classification-jetson

# JetPack 6.2 environment setup
chmod +x setup_jetson62.sh
./setup_jetson62.sh

# BERT özel paketler
pip install -r requirements_bert_jetpack62.txt

# Hızlı aktivasyon
./activate_rag62.sh
```

### 🎯 Eğitim Başlatma
```bash
cd jetson_training

# Custom dataset + AG News (her ikisi)
python bert_classification_trainer.py

# Sadece AG News
python -c "
from bert_classification_trainer import JetsonBERTTrainer
trainer = JetsonBERTTrainer()
trainer.run_ag_news_experiments()
"

# Sadece custom dataset
python -c "
from bert_classification_trainer import JetsonBERTTrainer
trainer = JetsonBERTTrainer()
trainer.run_custom_experiment('veri.csv')
"
```

## 📈 Beklenen Performans (JetPack 6.2)

### BERT-base-uncased
- **Training Time:** 15-20 dakika (3 epochs)
- **Memory Usage:** ~3GB GPU
- **Batch Size:** 8
- **Expected Accuracy:** 0.85-0.90

### BERT-large-uncased  
- **Training Time:** 25-35 dakika (3 epochs)
- **Memory Usage:** ~5GB GPU
- **Batch Size:** 4
- **Expected Accuracy:** 0.87-0.92

### System Resources
- **Temperature:** <75°C
- **Power:** ~18W
- **RAM Usage:** ~4GB

## 🔧 JetPack 6.2 Optimizasyonlar

### CUDA 12.2 Features
```python
# Otomatik aktif optimizasyonlar
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

### PyTorch 2.3 Compilation
```python
# Model compilation (otomatik)
model = torch.compile(model, mode="reduce-overhead")
```

### Mixed Precision Training
```python
# FP16 training (otomatik)
with autocast(enabled=True):
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
```

## 📊 Monitoring ve Analiz

### Real-time Monitoring
```bash
# GPU monitoring
nvidia-smi -l 1

# System monitoring
tegrastats

# Training progress
tail -f ../logs/training.log
```

### Sonuç Analizi
- **📁 Results:** `results/` klasöründe JSON/CSV formatında
- **📈 Graphs:** Otomatik karşılaştırma grafikleri
- **🎯 Best Models:** `models/best_*` klasörlerinde

## 📁 Proje Yapısı
```
bert-classification-jetson/
├── jetson_training/
│   ├── bert_classification_trainer.py    # Ana eğitim scripti
│   ├── gpu_optimizer_jp62.py             # JetPack 6.2 optimizer
│   └── ...
├── models/                               # Eğitilmiş modeller
│   ├── best_bert-base-uncased/
│   └── best_bert-large-uncased/
├── results/                              # Sonuçlar ve grafikler
│   ├── ag_news_results.json
│   ├── custom_dataset_results.json
│   └── model_comparison.png
├── veri.csv                              # Custom dataset
├── requirements_bert_jetpack62.txt       # BERT requirements
└── setup_jetson62.sh                     # JetPack 6.2 setup
```

## 📝 Custom Dataset Format

### veri.csv örneği:
```csv
text,label
"Bu film harika!","positive"
"Çok kötü bir deneyim","negative"
"Mükemmel hizmet","positive"
"Hiç beğenmedim","negative"
```

### Desteklenen formatlar:
- **Binary classification:** positive/negative, good/bad, yes/no
- **Multi-class:** category1, category2, category3, ...
- **Otomatik encoding:** String label'lar sayısal ID'lere çevrilir

## 🚨 Troubleshooting

### Memory Issues
```bash
# OOM hatası için batch size küçült
# bert_classification_trainer.py içinde:
batch_sizes = {'bert-base-uncased': 4, 'bert-large-uncased': 2}
```

### Dataset Issues
```bash
# Custom dataset kontrolü
python -c "
import pandas as pd
df = pd.read_csv('veri.csv')
print(df.head())
print(df['label'].value_counts())
"
```

### Performance Issues
```bash
# Maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Memory cleanup
python -c "import torch; torch.cuda.empty_cache()"
```

## 📞 Destek

- **GitHub Issues:** Repository'de issue açın
- **Custom Dataset:** veri.csv formatını kontrol edin
- **Performance:** `tegrastats` ve `nvidia-smi` çıktılarını paylaşın

---

**🤖 BERT Classification - Production Ready on JetPack 6.2!** 🚀

### Quick Commands
```bash
# Full setup and training
./setup_jetson62.sh && ./activate_rag62.sh
cd jetson_training && python bert_classification_trainer.py

# Monitor during training
tegrastats & nvidia-smi -l 1
``` 