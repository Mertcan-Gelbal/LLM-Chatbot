# AgriRAG-Multimodal-Ingestor

Bitki patolojisi için multimodal RAG (Retrieval-Augmented Generation) veri seti hazırlama sistemi.

## 🌱 Proje Açıklaması

Bu proje, bitki hastalıkları konusunda uzmanlaşmış bir RAG sistemi için veri toplama ve işleme pipeline'ı sağlar. Sistem şu bileşenleri birleştirir:

1. **Etiketli bitki hastalık görüntü veri seti** (Kaggle)
2. **Akademik literatür** (tezler, makaleler)
3. **Multimodal vektör indeksi** (metin + görüntü)

## 🚀 Özellikler

- ✅ Bitki-hastalık taksonomisi yönetimi
- ✅ Kaggle veri seti otomatik indirme
- ✅ Akademik literatür toplama (Google Scholar, YÖK Tez)
- ✅ PDF metin çıkarma ve temizleme
- ✅ Akıllı metin parçalama (chunking)
- ✅ Çok dilli embedding (Türkçe destekli)
- ✅ FAISS vektör indeksleme
- ✅ Görüntü metadata işleme
- ✅ Hata yönetimi ve logging
- ✅ JSON çıktı formatı

## 📋 Desteklenen Bitkiler ve Hastalıklar

| Bitki | Latin Adı | Hastalıklar |
|-------|-----------|-------------|
| Domates | Solanum lycopersicum | Erken yanıklık, Geç yanıklık, Mildiyö, Bakteriyel lekeler |
| Patates | Solanum tuberosum | Erken yanıklık, Geç yanıklık, Alternaria kararma, Virüs enfeksiyonları |
| Elma | Malus domestica | Elma pası, Karaleke, Külleme, Bakteriyel kanser |
| Üzüm | Vitis vinifera | Külleme, Mildiyö, Botrytis çürüklüğü, Kök çürüklüğü |
| Buğday | Triticum aestivum | Pas hastalıkları, Külleme, Septoria lekesi |
| Çay | Camellia sinensis | Küf, Külleme, Bakteriyel lekeler |
| Mısır | Zea mays | Yaprak lekeleri, Fusarium, Viral mozaik |

## 🛠️ Kurulum

### 1. Sanal Ortam Oluşturma

```bash
python3.10 -m venv agri_rag_env
source agri_rag_env/bin/activate  # Linux/Mac
# veya
agri_rag_env\Scripts\activate  # Windows
```

### 2. Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

### 3. Kaggle API Kurulumu (Opsiyonel)

Kaggle veri setini otomatik indirmek için:

1. [Kaggle hesabınızdan](https://www.kaggle.com/account) API anahtarı indirin
2. `~/.kaggle/kaggle.json` dosyasına yerleştirin
3. Dosya izinlerini ayarlayın: `chmod 600 ~/.kaggle/kaggle.json`

## 🎯 Kullanım

### Temel Kullanım

```python
import asyncio
from agri_rag_ingestor import AgriRAGIngestor

async def main():
    # Ingestor'ı başlat
    ingestor = AgriRAGIngestor(output_dir="my_agri_rag")
    
    # Pipeline'ı çalıştır
    result = await ingestor.run_pipeline()
    
    # Sonuçları görüntüle
    print(f"İşlenen tez sayısı: {result['statistics']['num_theses']}")
    print(f"Metin parçası sayısı: {result['statistics']['num_text_chunks']}")
    print(f"Görüntü sayısı: {result['statistics']['num_images']}")

# Çalıştır
asyncio.run(main())
```

### Komut Satırından Çalıştırma

```bash
python agri_rag_ingestor.py
```

## 📁 Çıktı Yapısı

```
agri_rag_output/
├── images/                 # İndirilen görüntüler
├── pdfs/                   # İndirilen PDF'ler
├── texts/                  # Çıkarılan metinler
├── vectors/                # FAISS indeksleri
│   └── agri_rag_index.faiss
└── agri_rag_results.json   # Ana sonuç dosyası
```

## 📊 JSON Çıktı Formatı

```json
{
  "taxonomy": [
    {
      "bitki": "Domates",
      "latin": "Solanum lycopersicum",
      "hastalık": "Erken yanıklık"
    }
  ],
  "image_dataset": {
    "source": "kaggle://mertcangelbal/botanix-dataset",
    "num_images": 1500
  },
  "documents": [
    {
      "tez_başlığı": "Domates Erken Yanıklık Hastalığı...",
      "yazar": "Dr. Ahmet Yılmaz",
      "yıl": 2023,
      "bitki": "Domates",
      "hastalık": "Erken yanıklık",
      "chunks": ["uuid1", "uuid2", "uuid3"]
    }
  ],
  "vector_index": "faiss://path/to/index",
  "statistics": {
    "num_theses": 75,
    "num_text_chunks": 2340,
    "num_image_embeddings": 1500,
    "num_images": 1500,
    "processing_errors": []
  }
}
```

## 🔧 Yapılandırma

### Embedding Modeli Değiştirme

```python
ingestor = AgriRAGIngestor()
ingestor.initialize_embedding_model("sentence-transformers/distiluse-base-multilingual-cased")
```

### Chunk Boyutu Ayarlama

```python
chunks = ingestor.chunk_text(text, chunk_size=500, overlap=100)
```

## 🧪 Test Etme

```bash
# Temel test
python -c "import agri_rag_ingestor; print('✅ Import başarılı')"

# Pipeline test (kısa)
python agri_rag_ingestor.py
```

## 📈 Performans

- **Embedding Modeli**: Çok dilli Sentence-BERT
- **Vektör Arama**: FAISS (cosine similarity)
- **Chunk Boyutu**: 400 kelime (varsayılan)
- **Overlap**: 50 kelime
- **Desteklenen Formatlar**: PDF, JPG, PNG, JPEG

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🆘 Sorun Giderme

### Kaggle API Hatası
```
Kaggle API anahtarı bulunamadı
```
**Çözüm**: `~/.kaggle/kaggle.json` dosyasını oluşturun ve API anahtarınızı ekleyin.

### Memory Hatası
```
CUDA out of memory
```
**Çözüm**: Daha küçük batch size kullanın veya CPU modeline geçin.

### PDF İşleme Hatası
```
PDF metin çıkarma hatası
```
**Çözüm**: PDF dosyasının bozuk olmadığından emin olun.

## 📞 İletişim

- **Geliştirici**: Mert Can Gelbal
- **E-posta**: mertcangelbal@example.com
- **GitHub**: [@mertcangelbal](https://github.com/mertcangelbal)

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın! 