# 🚀 Model Eğitimi

Bu klasör, Botanical BERT modelini eğitmek için gerekli dosyaları içerir.

## Hızlı Başlangıç

```bash
# Model eğitimini başlat
cd CreateModel
python train_model.py
```

## Eğitim Süreci

### 1. Ön Hazırlık
- Dataset'ler `../Data/` klasöründe olmalı
- GPU/CUDA kurulu olmalı (tercihen)
- Gerekli Python paketleri kurulu olmalı

### 2. Model Konfigürasyonu
- **Model:** BERT-base-uncased (küçük versiyonu)
- **Parametreler:** ~22M (normal BERT'ten %80 küçük)
- **Kategoriler:** 6 tarımsal sınıf
- **Max Length:** 128 token

### 3. Eğitim Parametreleri
- **Epochs:** 3 (özelleştirilebilir)
- **Batch Size:** 8 (GPU memory'ye göre)
- **Learning Rate:** Otomatik (Hugging Face default)
- **Mixed Precision:** FP16 (GPU varsa)

## Çıktılar

Eğitim sonunda `../Model/` klasöründe oluşturulacaklar:

```
Model/
├── botanical_bert_model/
│   ├── config.json          # Model konfigürasyonu
│   ├── pytorch_model.bin    # Eğitilmiş ağırlıklar
│   ├── tokenizer.json       # Tokenizer
│   ├── vocab.txt           # Vocabulary
│   └── model_info.json     # Performans bilgileri
├── checkpoints/            # Eğitim checkpoints
└── logs/                   # Training logs
```

## Performance Beklentileri

### Jetson Orin Nano Super (8GB)
- **Eğitim Süresi:** ~10-15 dakika
- **Memory Kullanımı:** ~3-4GB GPU
- **Beklenen Accuracy:** %75-85

### Normal GPU (GTX 1080+)
- **Eğitim Süresi:** ~5-10 dakika  
- **Memory Kullanımı:** ~2-3GB GPU
- **Beklenen Accuracy:** %80-90

## Özelleştirme

### Eğitim Parametrelerini Değiştir

```python
# train_model.py içinde main() fonksiyonunda:

# Daha uzun eğitim
trainer = trainer_obj.train_model(train_df, val_df, epochs=5, batch_size=4)

# Daha büyük model
trainer_obj.prepare_model("bert-large-uncased")
```

### Batch Size Ayarlama

```python
# GPU memory yetersizse küçült
batch_size=4  # veya 2

# Güçlü GPU varsa artır  
batch_size=16  # veya 32
```

## Hata Çözümleri

### "CUDA out of memory"
```bash
# Batch size küçült
# train_model.py'de batch_size=4 yap
```

### "Dataset bulunamadı"
```bash
# Data klasörünü kontrol et
ls ../Data/
# train.csv, val.csv, test.csv olmalı
```

### "Package bulunamadı"
```bash
# Gerekli paketleri kur
pip install torch transformers sklearn pandas matplotlib seaborn
```

## Model İzleme

```bash
# GPU kullanımını izle
nvidia-smi -l 1

# Eğitim loglarını izle  
tail -f ../Model/logs/*/events*
```

## İleri Seviye

### Custom Dataset
```python
# Kendi dataset'inizi kullanın
train_df = pd.read_csv("custom_train.csv")
# Format: text,label
```

### Hiperparametre Tuning
```python
# Farklı learning rate
training_args.learning_rate = 2e-5

# Farklı warmup
training_args.warmup_steps = 200
```

## Kullanım Sonrası

Eğitim bittikten sonra:

```bash
# Model'i test et
cd ../Model
python run_model.py

# Örnek tahmin
python -c "
from run_model import predict_text
result = predict_text('Domates yaprak yanıklığı')
print(result)
"
``` 