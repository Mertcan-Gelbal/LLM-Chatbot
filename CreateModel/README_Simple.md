# 🌾 Basit Tarımsal BERT Fine-tuning

Bu proje, arkadaşının AG News koduna benzer şekilde, **BERT modelini tarımsal verilerle fine-tuning** yapan basit bir sistemdir.

## 🎯 Özellikler

- **Basit BERT Fine-tuning**: Arkadaşının AG News koduna benzer yapı
- **Tarımsal Kategoriler**: 3 kategori (hastalık, yetiştirme, çevre)
- **Clean Code**: Anlaşılır ve düzenli kod yapısı
- **Jetson Uyumlu**: Küçük batch size ve optimize parametreler

## 🚀 Hızlı Kullanım

### 1. Requirements yükle:
```bash
pip install -r requirements_simple.txt
```

### 2. Modeli eğit:
```bash
python simple_agricultural_bert.py
```

### 3. Chatbot'u test et:
```bash
python simple_agricultural_chatbot.py
```

## 📊 Kod Yapısı

### AG News Benzeri Yapı:
```python
# Veri yükleme fonksiyonu
def load_agricultural_data():
    # Manuel tarımsal veri seti
    texts = ["Elmada erken yanıklığı...", ...]
    labels = ["plant_disease", ...]
    
# Dataset sınıfı
class AgriculturalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        # AG News Dataset'e benzer

# Eğitim fonksiyonu  
def train(model, data_loader, optimizer, scheduler, device):
    # Aynı eğitim döngüsü

# Ana experiment
def run_agricultural_experiment():
    # AG News benzeri ana fonksiyon
```

## 🔧 Parametreler

- **Model**: bert-base-uncased
- **Epochs**: 3 
- **Batch Size**: 8 (Jetson için)
- **Learning Rate**: 2e-5
- **Max Length**: 128

## 📈 Sonuçlar

Model 3 kategoride sınıflandırma yapar:
- `plant_disease`: Bitki hastalıkları
- `crop_management`: Yetiştirme teknikleri  
- `environmental_factors`: Çevre faktörleri

## 💬 Chatbot Kullanımı

```bash
python simple_agricultural_chatbot.py
```

**Örnek Sorular:**
- "Elmada erken yanıklığı nasıl tedavi edilir?"
- "Buğday ekim zamanı ne zaman?"
- "Toprak pH değeri neden önemli?"

**Örnek Cevap:**
```
Soru: Elmada erken yanıklığı nasıl tedavi edilir?
Cevap: **plant_disease** (Güven: 0.95)

Elmada erken yanıklığı bakteriyel bir hastalıktır. 
Hasta dalları kesin, sterilize edin, antibiyotik sprey uygulayın.
```

## 📁 Dosya Yapısı

```
CreateModel/
├── simple_agricultural_bert.py      # Ana eğitim kodu
├── simple_agricultural_chatbot.py   # Chatbot kodu
├── requirements_simple.txt          # Kütüphane listesi
├── README_Simple.md                 # Bu dosya
└── agricultural_bert_base_uncased/  # Eğitilmiş model (otomatik)
```

## 🎉 Başarı

Bu sistem:
✅ Arkadaşının koduna benzer basitlik
✅ Tarımsal domain'e özel
✅ Jetson optimizasyonu
✅ Test edilebilir chatbot
✅ Clean code yapısı

**Amacına uygun, basit ve çalışan bir BERT fine-tuning sistemi!** 