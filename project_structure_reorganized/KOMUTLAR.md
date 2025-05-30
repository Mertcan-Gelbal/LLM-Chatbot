# ⚡ Hızlı Komut Kartı

## 🚀 TEK KOMUTLA TEST

```bash
# Otomatik test scripti (30 dakika)
./test_all.sh
```

## 📋 KOPYALA-YAPIŞTIR KOMUTLARI

### 1️⃣ Hızlı Başlangıç (5 dakika)
```bash
cd project_structure_reorganized
pip install -r requirements.txt
cd 02_models/bert_classification/
python simple_agricultural_bert.py
cd ../../04_deployment/chatbots/
python simple_agricultural_chatbot.py
```

### 2️⃣ En İyi Modeli Test Et (15 dakika)
```bash
cd project_structure_reorganized/02_models/bert_classification/
python train_distilbert.py
cd ../../04_deployment/chatbots/
python simple_agricultural_chatbot.py
```

### 3️⃣ Performans Karşılaştırması
```bash
cd project_structure_reorganized/03_training_results/
python compare_all_models.py
cat performance_metrics/comparison_results.json
```

### 4️⃣ RAG Sistem Test
```bash
cd project_structure_reorganized/02_models/rag_hybrid/
python advanced_agricultural_rag_chatbot.py
```

## 🔍 SONUÇ KONTROL

### Eğitilmiş Modelleri Listele
```bash
find project_structure_reorganized/ -name "*.bin" -o -name "pytorch_model.*"
```

### Performans Sonuçlarını Gör
```bash
cat project_structure_reorganized/03_training_results/performance_metrics/comparison_results.json
```

### Proje Boyutunu Kontrol Et
```bash
du -sh project_structure_reorganized/02_models/*/
```

## 🧪 TEST SORULARI

### Chatbot'ta Dene:
```
Elmada erken yanıklığı nasıl tedavi edilir?
Buğday ekim zamanı ne zaman?
Toprak pH değeri neden önemli?
Aşırı sıcaklıkta bitkileri nasıl koruruz?
```

## ❓ SORUN GİDERME

### Model Bulunamadı:
```bash
cd project_structure_reorganized/02_models/bert_classification/
python simple_agricultural_bert.py
```

### Bağımlılık Hatası:
```bash
pip install torch transformers pandas numpy scikit-learn
```

### Bellek Hatası:
```bash
# simple_agricultural_bert.py içinde batch_size=4 yap
```

## 📊 BEKLENEN SONUÇLAR

- ✅ BERT: %86+ accuracy
- ✅ DistilBERT: %96+ accuracy  
- ✅ Chatbot yanıt veriyor
- ✅ JSON sonuç dosyası oluştu

**Toplam süre:** 30 dakika  
**Disk alanı:** ~3GB  
**RAM ihtiyacı:** 4GB+ 