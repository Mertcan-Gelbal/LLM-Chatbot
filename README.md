# 🌾 Agricultural RAG System - Jetson Orin Nano Optimized

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.4+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![Jetson](https://img.shields.io/badge/Jetson-Orin%20Nano-orange.svg)](https://developer.nvidia.com/embedded/jetson-orin)

Tarımsal hastalık tespiti için optimize edilmiş RAG (Retrieval-Augmented Generation) sistemi. Jetson Orin Nano Super kartında GPU hızlandırmalı model eğitimi için tasarlanmıştır.

## 🎯 Özellikler

- **13,200 chunk** indekslenmiş tarım verisi
- **182 gerçek PDF** + **1,000 sentetik makale**
- **GPU optimized** Jetson Orin Nano için
- **ONNX Runtime** desteği
- **FAISS** hızlı arama
- **Çoklu dil** desteği (TR/EN)

## 📊 Veri Seti
- **PDF Chunk:** 3,452 (Gerçek akademik makaleler)
- **Sentetik Chunk:** 9,748 (Yüksek kaliteli sentetik data)
- **Toplam Kelime:** 6,355,302
- **Embedding:** 384D (all-MiniLM-L6-v2)
- **Arama Skoru:** 0.52 ortalama (İyi seviye)

## 🚀 Jetson Kurulum

### Sistem Gereksinimleri
- **Jetson Orin Nano Super** (8GB+ RAM)
- **JetPack 5.1+** 
- **CUDA 11.4+**
- **Python 3.8+**
- **GPU Memory:** 4GB+

### Hızlı Başlangıç
```bash
# 1. Repo klonla
git clone https://github.com/[USERNAME]/agricultural-rag-jetson.git
cd agricultural-rag-jetson

# 2. Jetson environment setup
chmod +x setup_jetson.sh
./setup_jetson.sh

# 3. Model eğitimi başlat
python jetson_train.py --gpu --mixed-precision
```

## 📁 Proje Yapısı
```
agricultural-rag-jetson/
├── data_processing/           # Veri işleme
│   ├── real_papers/          # PDF makaleler
│   └── synthetic_papers/     # Sentetik veriler
├── final_system/             # İndekslenmiş veriler
│   └── complete_index/       # FAISS + embeddings
├── jetson_training/          # Jetson optimized training
│   ├── jetson_train.py       # Ana eğitim script
│   ├── gpu_optimizer.py      # GPU optimizasyonları
│   └── export_onnx.py        # ONNX export
├── models/                   # Eğitilmiş modeller
├── scripts/                  # Yardımcı scriptler
└── setup_jetson.sh          # Jetson kurulum
```

## 🔧 Jetson Optimizasyonları
- **TensorRT** hızlandırma
- **Mixed Precision** (FP16)
- **Memory optimization**
- **CUDA Graph** desteği
- **Dynamic batching**

## 📈 Performans
- **Training:** ~2-3 saat (Jetson Orin Nano)
- **Inference:** <100ms per query
- **Memory Usage:** ~3GB GPU, ~4GB RAM
- **Accuracy:** 0.85+ F1 score

## 🛠️ API Kullanımı
```python
from agricultural_rag import JetsonRAG

# Model yükle
rag = JetsonRAG.from_pretrained("models/rag_jetson_optimized")

# Soru sor
response = rag.query("Domates yaprak leke hastalığı nasıl tedavi edilir?")
print(response)
```

## 📝 Lisans
MIT License - Akademik ve ticari kullanım için ücretsiz.

## 🤝 Katkı
1. Fork yapın
2. Feature branch oluşturun
3. Commit yapın
4. Pull request gönderin

---
**🔗 Bağlantılar:**
- [Jetson Developer Guide](https://developer.nvidia.com/embedded/jetson-orin)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Agricultural Dataset Paper](link-to-paper) 