# 🌾 Tarımsal AI Proje Koleksiyonu

Bu proje, **küçük dil modellerinin tarımsal uygulamalarda** kullanılması üzerine kapsamlı bir araştırma ve geliştirme çalışmasıdır.

## 📊 Proje Özeti

### 🎯 Amaç
Farklı AI yaklaşımlarını tarımsal danışmanlık alanında karşılaştırmak ve en uygun çözümü bulmak.

### 🧪 Test Edilen Yaklaşımlar
1. **BERT Fine-tuning** - Sınıflandırma tabanlı
2. **GPT-2 Fine-tuning** - Text generation tabanlı  
3. **RAG (Retrieval-Augmented Generation)** - Hibrit yaklaşım
4. **Template-based Systems** - Kural tabanlı

### 📈 Ana Bulgular
- **En İyi Performans**: DistilBERT (%96.3 accuracy)
- **En Doğal Konuşma**: GPT-2 Fine-tuned
- **En Kapsamlı**: RAG sistemi
- **En Hızlı**: Template-based

## 📁 Proje Yapısı

```
project_structure_reorganized/
├── README.md                           # Bu dosya
├── 01_data_preparation/               # Veri hazırlama
│   ├── original_data/                 # Ham veriler
│   ├── processed_data/               # İşlenmiş veriler
│   └── synthetic_data/               # Sentetik veriler
├── 02_models/                        # Model implementasyonları
│   ├── bert_classification/          # BERT sınıflandırma
│   ├── gpt2_generation/             # GPT-2 text generation
│   ├── rag_hybrid/                  # RAG sistemi
│   └── template_based/              # Template sistemler
├── 03_training_results/             # Eğitim sonuçları
│   ├── performance_metrics/         # Performans metrikleri
│   ├── model_outputs/              # Model çıktıları
│   └── comparative_analysis/       # Karşılaştırmalı analiz
├── 04_deployment/                   # Deployment dosyaları
│   ├── chatbots/                   # Chatbot implementasyonları
│   └── apis/                       # API servisleri
└── 05_documentation/               # Dokümantasyon
    ├── technical_report.md         # Teknik rapor
    ├── methodology.md              # Metodoloji
    └── results_analysis.md         # Sonuç analizi
```

## 🔬 Araştırma Metodolojisi

### 1. Veri Toplama ve Hazırlama
- **Ham Veri**: 1,800+ tarımsal metin
- **Kategoriler**: 6 ana kategori (hastalık, yetiştirme, çevre, vb.)
- **Sentetik Veri**: GPT destekli veri artırımı
- **Ön İşleme**: Tokenization, normalizasyon

### 2. Model Seçimi ve Eğitimi
- **Baseline**: Template-based responses
- **Classical ML**: BERT/DistilBERT fine-tuning
- **Modern NLP**: GPT-2 generation
- **Hybrid**: RAG with embeddings

### 3. Değerlendirme Kriterleri
- **Objektif Metrikler**: Accuracy, F1-score, Precision
- **Subjektif Metrikler**: Doğallık, Yararlılık, Tutarlılık
- **Performans**: Hız, Bellek kullanımı, GPU ihtiyacı

## 📊 Performans Karşılaştırması

| Model | Accuracy | F1-Score | Doğallık | Hız | Bellek |
|-------|----------|----------|----------|-----|--------|
| BERT-small | 86.3% | 86.5% | ⭐⭐⭐ | 🚀🚀🚀 | 🟢 |
| DistilBERT | 96.3% | 96.2% | ⭐⭐⭐⭐ | 🚀🚀 | 🟡 |
| GPT-2 | N/A* | N/A* | ⭐⭐⭐⭐⭐ | 🚀 | 🔴 |
| RAG | 85%** | 85%** | ⭐⭐⭐⭐ | 🚀🚀 | 🟡 |
| Template | 70%*** | 70%*** | ⭐⭐ | 🚀🚀🚀🚀 | 🟢 |

*Generation task için classification metrics uygulanamaz  
**Retrieval accuracy  
***Rule-based accuracy estimation

## 🎯 Sonuçlar ve Öneriler

### En İyi Seçenekler:
1. **Genel Kullanım**: DistilBERT (yüksek accuracy + makul performans)
2. **Doğal Sohbet**: GPT-2 fine-tuned (en insan benzeri)
3. **Hızlı Deploy**: Template-based (minimum kaynak)
4. **Kapsamlı Bilgi**: RAG sistemi (geniş bilgi tabanı)

### Jetson Önerileri:
- **Üretim**: DistilBERT + Template hybrid
- **Geliştirme**: GPT-2 small model
- **Edge**: BERT-small optimized

## 📚 Teknik Detaylar

Detaylı teknik bilgiler için:
- [Teknik Rapor](05_documentation/technical_report.md)
- [Metodoloji](05_documentation/methodology.md) 
- [Sonuç Analizi](05_documentation/results_analysis.md)

## 🚀 Hızlı Başlangıç

```bash
# 1. En iyi model (DistilBERT)
cd 02_models/bert_classification/
python train_distilbert.py

# 2. Chatbot test
cd 04_deployment/chatbots/
python distilbert_chatbot.py

# 3. Karşılaştırmalı test
cd 03_training_results/
python compare_all_models.py
```

## 📝 Katkıda Bulunanlar

Bu proje, tarımsal AI uygulamaları araştırması kapsamında geliştirilmiştir. 