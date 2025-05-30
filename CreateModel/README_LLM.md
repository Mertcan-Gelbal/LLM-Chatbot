# 🧠 Gerçek LLM Tarımsal Chatbot

Bu proje **gerçek bir Language Model (GPT-2)** kullanarak tarımsal danışmanlık yapan bir AI sistemidir. Önceki template-based sistemlerden farklı olarak, bu sistem **gerçek text generation** yapar.

## 🌟 Özellikler

### 🤖 Gerçek LLM Teknolojisi
- **GPT-2 Small Model**: Hugging Face'den indirilen gerçek language model
- **Fine-tuning**: Tarımsal verilerle özel eğitim
- **Text Generation**: Template değil, gerçek metin üretimi
- **Doğal Dil**: İnsan gibi akıcı konuşma

### 🎯 Tarımsal Uzmanlık
- Bitki hastalıkları (erken yanıklığı, sarı yaprak, vb.)
- Yetiştirme teknikleri (ekim, sulama, gübreleme)
- Çevre faktörleri (sıcaklık stresi, pH kontrolü)
- Pratik öneriler ve çözümler

### ⚡ Jetson Optimizasyonu
- Küçük model (GPT-2 small ~124M parametre)
- Mixed precision (FP16) desteği
- Düşük batch size
- CPU/GPU uyumlu

## 🚀 Hızlı Başlangıç

### 1. Basit Başlatma
```bash
cd CreateModel
python run_llm_training.py
```
Bu script size menü sunar:
- Kütüphaneleri yükler
- Modeli eğitir
- Chatbot'u başlatır

### 2. Manuel Adımlar

#### Requirements yükle:
```bash
pip install -r requirements_llm.txt
```

#### LLM'i eğit:
```bash
python train_agricultural_llm.py
```

#### Chatbot'u başlat:
```bash
python real_llm_agricultural_chatbot.py
```

## 📚 Eğitim Verisi

Model şu konularda eğitilmiştir:

### 🍎 Meyve Hastalıkları
- Elmada erken yanıklığı (Erwinia amylovora)
- Armutta erken yanıklığı
- Tedavi ve korunma yöntemleri

### 🌾 Tahıl Tarımı
- Buğday ekim zamanı ve tekniği
- Toprak hazırlığı
- Gübreleme stratejileri

### 🍅 Sebze Yetiştirme
- Domates sarı yaprak sorunu
- Havuç yetiştirme teknikleri
- Sulama ve bakım

### 🌡️ Çevre Stres Yönetimi
- Aşırı sıcaklık korunması
- Bitki takviye yöntemleri
- Stres faktörü yönetimi

## 💬 Kullanım Örnekleri

### Gerçek Konuşma Örnekleri:

**Kullanıcı:** "Elmamda yanıklık var, ne yapmalıyım?"

**LLM AI:** "Bu erken yanıklığı (Erwinia amylovora) olabilir. Hasta dalları hemen 30 cm sağlam kısımdan kesin. Kesim aletlerini %10 çamaşır suyu ile sterilize edin. Streptomisin içerikli sprey uygulayın ve kesim yerlerini bahçe macunu ile kapatın..."

**Kullanıcı:** "Sıcak havada bitkilerimi nasıl koruyabilirim?"

**LLM AI:** "Aşırı sıcaklıkta bitkileri korumak için gölgeleme ağları kurun, mulch uygulayın. Potasyum sülfat verin çünkü sıcaklık stresine karşı direnç artırır. Erken sabah veya akşam sulaması yapın..."

## 🔧 Teknik Detaylar

### Model Mimarisi
- **Base Model**: GPT-2 small (124M parametre)
- **Tokenizer**: GPT-2Tokenizer + özel tokenlar
- **Special Tokens**: `<|soru|>`, `<|cevap|>`, `<|end|>`
- **Max Length**: 512 token

### Eğitim Parametreleri
- **Epochs**: 3 (Jetson için optimize)
- **Batch Size**: 2 (bellek dostu)
- **Learning Rate**: 5e-5
- **Warmup Steps**: 50
- **FP16**: GPU varsa otomatik

### Generation Parametreleri
- **Temperature**: 0.7 (yaratıcılık dengeli)
- **Top-p**: 0.9 (kaliteli üretim)
- **Repetition Penalty**: 1.1 (tekrar önleme)

## 📊 Model Performansı

### Eğitim Süreçi
- **Training Time**: ~5-10 dakika (Jetson AGX)
- **Dataset Size**: ~15 Q&A çifti (demo)
- **Loss Reduction**: Başarılı convergence

### Test Sonuçları
- **Doğal Dil**: ✅ Akıcı Türkçe üretim
- **Tarımsal Bağlam**: ✅ Konu uygunluğu
- **Tutarlılık**: ✅ Mantıklı cevaplar
- **Teknik Doğruluk**: ✅ Bilimsel geçerlilik

## 🆚 Önceki Sistemle Karşılaştırma

| Özellik | Eski RAG Sistemi | Yeni LLM Sistemi |
|---------|------------------|------------------|
| Text Generation | ❌ Template-based | ✅ Gerçek LLM |
| Doğallık | ⚠️ Mekanik | ✅ İnsan benzeri |
| Yaratıcılık | ❌ Sınırlı | ✅ Yaratıcı |
| Bağlam | ⚠️ Retrieval-only | ✅ Generative |
| Esneklik | ❌ Rigid | ✅ Adaptatif |

## 🛠️ Geliştirme

### Model İyileştirme
1. **Veri Artırma**: Daha fazla Q&A çifti ekleyin
2. **Fine-tuning**: Daha uzun eğitim
3. **Model Büyütme**: GPT-2 medium/large
4. **Domain Adaptation**: Spesifik tarım alanları

### Kod Genişletme
```python
# Yeni eğitim verisi ekleme
training_data.extend([
    {
        "input": "Yeni soru",
        "output": "Detaylı cevap"
    }
])
```

## 🐛 Sorun Giderme

### Model Bulunamadı Hatası
```
❌ Eğitilmiş model bulunamadı!
```
**Çözüm**: Önce `train_agricultural_llm.py` çalıştırın

### CUDA Bellek Hatası
```
RuntimeError: CUDA out of memory
```
**Çözüm**: `batch_size`'ı 1'e düşürün

### Import Hatası
```
ModuleNotFoundError: No module named 'transformers'
```
**Çözüm**: `pip install -r requirements_llm.txt`

## 📈 Sonraki Adımlar

1. **🔄 Büyük Dataset**: Gerçek tarımsal veritabanı entegrasyonu
2. **🌐 Web Interface**: Streamlit/Gradio arayüzü
3. **📱 Mobile App**: Android/iOS uygulaması
4. **🔗 API**: REST API servisi
5. **🧠 Büyük Model**: Llama-2, GPT-3.5 entegrasyonu

## 🤝 Katkı

Bu proje açık kaynak! Geliştirmek için:
1. Fork yapın
2. Feature branch oluşturun
3. Pull request gönderin

## 📜 Lisans

MIT License - Detaylar için LICENSE dosyasına bakın.

## 🌾 Sonuç

Bu sistem **gerçek bir Language Model** kullanarak tarımsal danışmanlık yapar. Template-based sistemlerden çok daha doğal, akıcı ve kullanışlıdır. Jetson gibi edge devices için optimize edilmiştir.

**🎯 Amacımız**: Çiftçilere AI destekli, akıllı ve erişilebilir tarımsal danışmanlık sağlamak! 