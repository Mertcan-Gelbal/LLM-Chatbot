# 🚀 Hızlı Başlangıç Rehberi

Bu rehber, Tarımsal AI projelerini hızlıca çalıştırmanız için hazırlanmıştır.

## ⚡ 5 Dakikada Çalıştırma

### 1. Ortam Hazırlama
```bash
# Repository'yi klonlayın
cd project_structure_reorganized

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 2. En İyi Modeli Test Etme (DistilBERT)
```bash
# BERT modelini eğitin
cd 02_models/bert_classification/
python simple_agricultural_bert.py

# Chatbot'u çalıştırın
cd ../../04_deployment/chatbots/
python simple_agricultural_chatbot.py
```

### 3. Tüm Modelleri Karşılaştırma
```bash
cd 03_training_results/
python compare_all_models.py
```

## 📊 Hangi Modeli Seçeyim?

### 🎯 Kullanım Amacınıza Göre:

| Amaç | Önerilen Model | Neden? |
|------|----------------|--------|
| **Hızlı Prototip** | BERT Simple | En hızlı kurulum |
| **Üretim Uygulaması** | DistilBERT | En iyi performans/verimlilik |
| **Araştırma Projesi** | RAG System | En kapsamlı bilgi |
| **Doğal Konuşma** | GPT-2 | En insan benzeri |

### 💻 Donanım Kısıtlarınıza Göre:

| Donanım | Önerilen Model | Memory | GPU |
|---------|----------------|--------|-----|
| **Jetson Nano** | BERT Simple | 2GB+ | İsteğe bağlı |
| **Jetson AGX** | DistilBERT | 4GB+ | Önerilen |
| **PC/Server** | GPT-2 / RAG | 8GB+ | CUDA |

## 🛠️ Model Eğitimi

### BERT Classification
```bash
cd 02_models/bert_classification/
python simple_agricultural_bert.py
```
**Süre:** ~10 dakika  
**Sonuç:** %86+ accuracy

### DistilBERT (En İyi)
```bash
cd 02_models/bert_classification/
python train_distilbert.py
```
**Süre:** ~15 dakika  
**Sonuç:** %96+ accuracy

## 💬 Chatbot Kullanımı

### Basit Chatbot
```bash
cd 04_deployment/chatbots/
python simple_agricultural_chatbot.py
```

**Örnek Sorular:**
- "Elmada erken yanıklığı nasıl tedavi edilir?"
- "Buğday ekim zamanı ne zaman?"
- "Toprak pH değeri neden önemli?"

### RAG Chatbot (Gelişmiş)
```bash
cd 02_models/rag_hybrid/
python advanced_agricultural_rag_chatbot.py
```

## 📈 Performans Testi

### Hızlı Test
```bash
cd 03_training_results/
python compare_all_models.py
```

### Detaylı Analiz
```bash
# Sonuçları görüntüle
cat 03_training_results/performance_metrics/comparison_results.json

# Model çıktılarını incele
cat 03_training_results/model_outputs/comparison_examples.md
```

## 🔧 Özelleştirme

### Yeni Veri Ekleme
```bash
# Veri dosyanızı buraya ekleyin
cd 01_data_preparation/original_data/

# Yeni kategori eklemek için model yeniden eğitin
cd ../../02_models/bert_classification/
python simple_agricultural_bert.py
```

### Model Parametreleri
```python
# simple_agricultural_bert.py dosyasında
BATCH_SIZE = 8      # Jetson için düşük tut
EPOCHS = 3          # Hızlı eğitim için
LEARNING_RATE = 2e-5  # Standart BERT lr
```

## 📊 Beklenen Sonuçlar

### Model Performansları
| Model | Accuracy | Hız | Bellek |
|-------|----------|-----|--------|
| BERT | 86% | ⚡⚡⚡ | 2GB |
| DistilBERT | **96%** | ⚡⚡ | 3GB |
| GPT-2 | N/A | ⚡ | 4GB |

### Yanıt Süreleri (Jetson AGX)
- **BERT:** ~50ms
- **DistilBERT:** ~80ms  
- **GPT-2:** ~200ms
- **RAG:** ~150ms

## ❓ Sorun Giderme

### Model Bulunamadı Hatası
```bash
# Model dizinini kontrol edin
ls 02_models/bert_classification/

# Eğer boşsa, modeli yeniden eğitin
python simple_agricultural_bert.py
```

### Bellek Hatası
```python
# Batch size'ı azaltın
BATCH_SIZE = 4  # Jetson Nano için
BATCH_SIZE = 8  # Jetson AGX için
```

### CUDA Hatası
```python
# CPU modunda çalıştırın
device = torch.device('cpu')
```

## 📚 Dokümantasyon

- **Teknik Rapor:** [05_documentation/technical_report.md](05_documentation/technical_report.md)
- **Model Çıktıları:** [03_training_results/model_outputs/](03_training_results/model_outputs/)
- **Ana README:** [README.md](README.md)

## 🎯 Sonraki Adımlar

1. **Model Eğitin:** En iyi modeli (DistilBERT) eğitin
2. **Test Edin:** Karşılaştırma scriptini çalıştırın  
3. **Özelleştirin:** Kendi verilerinizi ekleyin
4. **Deploy Edin:** Production için chatbot'u hazırlayın

---

**🌾 Başarılar! Tarımsal AI projenizde size yardımcı olmaktan memnuniyet duyarız.** 