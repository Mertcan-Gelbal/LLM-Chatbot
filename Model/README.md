# 🤖 Eğitilmiş Model

Bu klasör, eğitilmiş Botanical BERT modelini ve çalıştırma scriptlerini içerir.

## Hızlı Kullanım

```bash
# İnteraktif chat demo
cd Model
python run_model.py

# Tek tahmin
python run_model.py "Domates yaprak yanıklığı"

# Programatik kullanım
python -c "
from run_model import predict_text
result = predict_text('Buğday hastalığı')
print(result)
"
```

## Model Yapısı

```
Model/
├── botanical_bert_model/
│   ├── config.json          # Model konfigürasyonu
│   ├── pytorch_model.bin    # Eğitilmiş ağırlıklar (~90MB)
│   ├── tokenizer.json       # Tokenizer dosyası
│   ├── vocab.txt           # Vocabulary
│   └── model_info.json     # Performans metrikleri
├── checkpoints/            # Eğitim checkpoints (opsiyonel)
├── logs/                   # Training logs (opsiyonel)
├── run_model.py           # Ana çalıştırma scripti
└── README.md              # Bu dosya
```

## Model Özellikleri

### Teknik Detaylar
- **Model:** BERT-base-uncased (küçük versiyon)
- **Parametreler:** ~22M (normal BERT'ten %80 küçük)
- **Model Boyutu:** ~90MB
- **Max Length:** 128 token
- **Kategoriler:** 6 tarımsal sınıf

### Performans
- **Test Accuracy:** %75-85
- **F1 Score:** 0.75-0.85
- **Inference Hızı:** ~15ms per query
- **GPU Memory:** ~500MB

## Kullanım Örnekleri

### 1. İnteraktif Chat
```bash
python run_model.py
# → Test örnekleri göster
# → İnteraktif chat başlat
```

### 2. Komut Satırından Tahmin
```bash
python run_model.py "Mısır ekimi için en uygun toprak türü"
# → Çıktı:
# 🎯 Kategori: Mahsul Yönetimi
# 📊 Güven: 87.3%
```

### 3. Python Kodunda Kullanım
```python
from run_model import BotanicalBERTPredictor

# Predictor oluştur
predictor = BotanicalBERTPredictor()

# Tek tahmin
result = predictor.predict_text(
    "Genetiği değiştirilmiş soya fasulyesi",
    return_probabilities=True
)

print(f"Kategori: {result['category_turkish']}")
print(f"Güven: {result['confidence']:.2%}")

# Batch tahmin
texts = [
    "Domates hastalığı",
    "Akıllı sulama sistemi", 
    "Organik gübre kullanımı"
]
results = predictor.predict_batch(texts)

for result in results:
    print(f"{result['text']} → {result['category_turkish']}")
```

## Kategori Açıklamaları

| Kategori | Türkçe | Açıklama | Örnek Kelimeler |
|----------|--------|----------|-----------------|
| `plant_disease` | Bitki Hastalıkları | Fungal, bacterial, viral hastalıklar | hastalık, mantar, virus, tedavi |
| `crop_management` | Mahsul Yönetimi | Ekim, gübreleme, hasat | ekim, gübre, sulama, hasat |
| `plant_genetics` | Bitki Genetiği | GMO, ıslah, çeşit geliştirme | genetik, GMO, çeşit, ıslah |
| `environmental_factors` | Çevre Faktörleri | İklim, toprak, stres | iklim, toprak, kuraklık, pH |
| `food_security` | Gıda Güvenliği | Üretim, depolama, beslenme | gıda, depolama, beslenme, güvenlik |
| `technology` | Tarım Teknolojisi | AI, drone, sensör, otomasyon | teknoloji, AI, drone, sensör |

## API Referansı

### BotanicalBERTPredictor Sınıfı

```python
class BotanicalBERTPredictor:
    def __init__(self, model_path="botanical_bert_model")
    def predict_text(self, text, return_probabilities=False)
    def predict_batch(self, texts)
    def interactive_demo(self)
```

### predict_text() Yanıtı

```python
{
    'text': 'Girdi metni',
    'predicted_category': 'plant_disease',
    'category_turkish': 'Bitki Hastalıkları', 
    'confidence': 0.891,
    'timestamp': '2024-05-29T22:15:30',
    'all_probabilities': {  # return_probabilities=True ise
        'plant_disease': 0.891,
        'crop_management': 0.045,
        'technology': 0.032,
        # ...
    }
}
```

## Performance Tuning

### GPU Hızlandırma
```python
# CUDA kullanımını kontrol et
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Batch İşleme
```python
# Çok sayıda tahmin için batch kullan
texts = ["text1", "text2", "text3", ...]
results = predictor.predict_batch(texts)
```

### Memory Optimizasyonu
```python
# Model yükledikten sonra CUDA cache temizle
torch.cuda.empty_cache()
```

## Hata Çözümleri

### "Model bulunamadı"
```bash
# Model dosyalarını kontrol et
ls -la botanical_bert_model/
# config.json, pytorch_model.bin olmalı

# Yoksa eğitim yap
cd ../CreateModel && python train_model.py
```

### "CUDA hatası"
```bash
# CPU modunda çalıştır
export CUDA_VISIBLE_DEVICES=""
python run_model.py
```

### "Paket bulunamadı"
```bash
pip install torch transformers
```

## Model Güncelleme

### Yeni Model Yükle
```bash
# Mevcut modeli yedekle
mv botanical_bert_model botanical_bert_model_backup

# Yeni model kopyala
cp -r ../CreateModel/output/model botanical_bert_model
```

### Model Karşılaştır
```python
# Eski ve yeni model performansını karşılaştır
old_predictor = BotanicalBERTPredictor("botanical_bert_model_backup")
new_predictor = BotanicalBERTPredictor("botanical_bert_model")

test_text = "Domates yaprak yanıklığı"
old_result = old_predictor.predict_text(test_text)
new_result = new_predictor.predict_text(test_text)

print(f"Eski model: {old_result['confidence']:.3f}")
print(f"Yeni model: {new_result['confidence']:.3f}")
```

## Entegrasyon

### Web API
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
predictor = BotanicalBERTPredictor()

@app.route('/predict', methods=['POST'])
def api_predict():
    data = request.json
    text = data.get('text', '')
    
    result = predictor.predict_text(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Streamlit Web App
```python
import streamlit as st

st.title("🌱 Botanical BERT")
text = st.text_input("Tarımsal sorunuzu yazın:")

if text:
    result = predictor.predict_text(text)
    st.write(f"**Kategori:** {result['category_turkish']}")
    st.write(f"**Güven:** {result['confidence']:.2%}")
``` 