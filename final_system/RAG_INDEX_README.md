# 🌾 Agricultural RAG System - İndekslenmiş Veriler

## 📊 Veri Seti Özeti
- **Toplam Chunk:** 13,200 metin parçası
- **PDF Chunk:** 3,452 (182 gerçek akademik makale)
- **Sentetik Chunk:** 9,748 (1,000 sentetik makale)
- **Toplam Kelime:** 6,355,302
- **Embedding Boyutu:** 384D (all-MiniLM-L6-v2)

## 📁 Dosya Yapısı
```
complete_index/
├── chunks/
│   └── all_chunks.json          # 50MB - Tüm metin chunk'ları
├── embeddings/
│   └── embeddings.npy           # 19MB - Numpy embedding vektörleri
├── indices/
│   └── faiss_index.bin          # 19MB - FAISS arama indeksi
└── stats.json                   # 156B - İstatistikler
```

## 🚀 Hızlı Başlangıç

### Python'da Yükleme:
```python
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Chunk'ları yükle
with open('complete_index/chunks/all_chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Embeddings yükle
embeddings = np.load('complete_index/embeddings/embeddings.npy')

# FAISS indeksi yükle
index = faiss.read_index('complete_index/indices/faiss_index.bin')

# Model yükle
model = SentenceTransformer('all-MiniLM-L6-v2')
```

### Arama Örneği:
```python
# Sorgu
query = "tomato disease detection"
query_embedding = model.encode([query])
faiss.normalize_L2(query_embedding)

# Arama (en iyi 5 sonuç)
scores, indices = index.search(query_embedding.astype('float32'), 5)

# Sonuçları göster
for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
    chunk = chunks[idx]
    print(f"{i+1}. Skor: {score:.3f}")
    print(f"   Kaynak: {chunk['source']}")
    print(f"   Metin: {chunk['text'][:100]}...")
```

## 📋 Chunk Yapısı
Her chunk şu bilgileri içerir:
```json
{
  "text": "Metin içeriği...",
  "source": "pdf|synthetic",
  "filename": "dosya_adi.pdf",     // PDF için
  "paper_id": "synthetic_0001",    // Sentetik için
  "chunk_id": 1234
}
```

## 🔍 Performans
- **Ortalama arama skoru:** 0.52 (İyi seviye)
- **En iyi kategoriler:** AI/ML konuları (0.58+)
- **Dil desteği:** İngilizce (yüksek), Türkçe (orta)

## 📦 İndirme Seçenekleri
1. **Tam paket:** `agricultural_rag_index.tar.gz` (41MB)
2. **Sadece chunk'lar:** `chunks/all_chunks.json` (50MB)
3. **Sadece embeddings:** `embeddings/embeddings.npy` (19MB)
4. **Sadece indeks:** `indices/faiss_index.bin` (19MB)

## 🛠️ Gereksinimler
```bash
pip install sentence-transformers faiss-cpu numpy
```

## 📝 Lisans
Bu veri seti akademik araştırma amaçlı oluşturulmuştur.

---
📧 Sorularınız için: [Geliştirici]
📅 Oluşturma Tarihi: 29 Mayıs 2024 