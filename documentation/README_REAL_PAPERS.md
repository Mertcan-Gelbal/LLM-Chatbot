# 🌾 Gerçek Akademik Makale RAG Sistemi

Bu sistem, **gerçek akademik makaleleri** PubMed ve arXiv gibi kaynaklardan indirerek bitki patolojisi için multimodal RAG (Retrieval-Augmented Generation) sistemi oluşturur.

## 🎯 Özellikler

- ✅ **Gerçek akademik makaleler** (sentetik değil!)
- 📚 **PubMed ve arXiv entegrasyonu**
- 📄 **PDF indirme ve işleme**
- 🧠 **Çok dilli embedding** (Türkçe destekli)
- 🔍 **FAISS vektör indeksi**
- 🤖 **İnteraktif RAG sistemi**
- 📊 **Kapsamlı raporlama**

## 📋 Gereksinimler

### Python Paketleri
```bash
pip install -r requirements_real_papers.txt
```

### Ana Paketler
- `aiohttp` - Async HTTP istekleri
- `PyMuPDF` - PDF işleme
- `sentence-transformers` - Embedding
- `faiss-cpu` - Vektör indeksi
- `numpy`, `pandas` - Veri işleme

## 🚀 Hızlı Başlangıç

### Seçenek 1: Otomatik Pipeline (Önerilen)
```bash
python run_complete_pipeline.py
```

### Seçenek 2: Adım Adım
```bash
# 1. Makaleleri indir
python real_paper_downloader.py

# 2. PDF'leri indeksle
python pdf_indexer.py

# 3. RAG sistemini çalıştır
python complete_rag_system.py
```

## 📁 Dosya Yapısı

```
📦 Proje Dizini
├── 📄 real_paper_downloader.py      # Makale indirici
├── 📄 pdf_indexer.py                # PDF işleyici
├── 📄 complete_rag_system.py        # RAG sistemi
├── 📄 run_complete_pipeline.py      # Master script
├── 📄 requirements_real_papers.txt  # Gereksinimler
├── 📄 README_REAL_PAPERS.md         # Bu dosya
├── 📁 real_papers/                  # İndirilen makaleler
│   ├── 📁 pdfs/                     # PDF dosyaları
│   └── 📁 metadata/                 # Makale metadata'sı
└── 📁 indexed_papers/               # İşlenmiş veriler
    ├── 📁 chunks/                   # Metin parçaları
    ├── 📁 embeddings/               # Vektör indeksi
    ├── 📁 images/                   # Çıkarılan görüntüler
    └── 📁 metadata/                 # İşleme metadata'sı
```

## 🔧 Detaylı Kullanım

### 1. Makale İndirme (`real_paper_downloader.py`)

**Özellikler:**
- PubMed API entegrasyonu
- arXiv API entegrasyonu
- Otomatik PDF URL tespiti
- Rate limiting
- Hata yönetimi

**Arama Terimleri:**
- Bitki + hastalık kombinasyonları
- Tarımsal terimler
- Patoloji terimleri

**Çıktılar:**
- `real_papers/pdfs/` - İndirilen PDF'ler
- `real_papers/metadata/collected_papers_metadata.json` - Makale metadata'sı

### 2. PDF İndeksleme (`pdf_indexer.py`)

**Özellikler:**
- PDF metin çıkarma
- Görüntü çıkarma
- Chunk'lara bölme (500 kelime, 100 overlap)
- Çok dilli embedding
- FAISS indeksi oluşturma

**Çıktılar:**
- `indexed_papers/chunks/all_chunks.json` - Tüm metin parçaları
- `indexed_papers/embeddings/faiss_index.bin` - FAISS indeksi
- `indexed_papers/embeddings/embeddings.npy` - Embedding vektörleri
- `indexed_papers/metadata/chunk_metadata.json` - Chunk metadata'sı

### 3. RAG Sistemi (`complete_rag_system.py`)

**Özellikler:**
- Semantik arama
- Bağlam tabanlı cevap oluşturma
- İnteraktif sohbet modu
- Kaynak takibi
- Güven skoru

**Kullanım:**
```python
from complete_rag_system import AgriculturalRAGSystem

rag = AgriculturalRAGSystem()
result = rag.ask("Domates hastalıkları nasıl önlenir?")
print(result["answer"])
```

## 📊 API Referansı

### RealPaperDownloader

```python
downloader = RealPaperDownloader()
result = await downloader.collect_and_download_papers()
```

**Metodlar:**
- `search_pubmed(query, max_results)` - PubMed'den ara
- `search_arxiv(query, max_results)` - arXiv'den ara
- `download_pdf(paper, session)` - PDF indir

### PDFIndexer

```python
indexer = PDFIndexer()
result = indexer.process_all_pdfs()
```

**Metodlar:**
- `extract_text_from_pdf(pdf_path)` - PDF'den metin çıkar
- `create_chunks(text, metadata)` - Chunk'lara böl
- `create_embeddings(chunks)` - Embedding oluştur
- `build_faiss_index(embeddings)` - FAISS indeksi oluştur

### AgriculturalRAGSystem

```python
rag = AgriculturalRAGSystem()
result = rag.ask(question, top_k=5)
```

**Metodlar:**
- `search(query, top_k)` - Semantik arama
- `ask(question, top_k)` - Soru-cevap
- `get_statistics()` - Sistem istatistikleri
- `interactive_chat()` - İnteraktif mod

## 🔍 Arama Stratejileri

### Bitki Hastalıkları
```python
# Spesifik hastalık
result = rag.ask("tomato early blight treatment")

# Genel önleme
result = rag.ask("fungal disease prevention methods")

# İklim etkisi
result = rag.ask("climate change impact on plant diseases")
```

### Türkçe Sorgular
```python
result = rag.ask("Domates hastalıkları nasıl önlenir?")
result = rag.ask("Fungal enfeksiyonlar için hangi yöntemler kullanılır?")
result = rag.ask("İklim değişikliği bitki hastalıklarını nasıl etkiler?")
```

## 📈 Performans Optimizasyonu

### Embedding Modeli
- **Varsayılan:** `paraphrase-multilingual-MiniLM-L12-v2`
- **Alternatif:** `all-MiniLM-L6-v2` (daha hızlı)
- **Gelişmiş:** `all-mpnet-base-v2` (daha doğru)

### Chunk Parametreleri
```python
# Daha küçük chunk'lar (daha hassas)
indexer.chunk_size = 300
indexer.chunk_overlap = 50

# Daha büyük chunk'lar (daha hızlı)
indexer.chunk_size = 800
indexer.chunk_overlap = 150
```

### FAISS İndeksi
```python
# Daha hızlı arama için
index = faiss.IndexFlatIP(dimension)  # Mevcut

# Daha az bellek için
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
```

## 🐛 Sorun Giderme

### Yaygın Hatalar

**1. Paket Eksik**
```bash
pip install -r requirements_real_papers.txt
```

**2. PDF İndirme Başarısız**
- İnternet bağlantısını kontrol edin
- Rate limiting nedeniyle bekleyin
- VPN kullanmayı deneyin

**3. Embedding Modeli Yüklenmiyor**
```bash
# Manuel indirme
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

**4. FAISS Hatası**
```bash
# CPU versiyonu
pip install faiss-cpu

# GPU versiyonu (opsiyonel)
pip install faiss-gpu
```

### Log Seviyeleri
```python
import logging
logging.basicConfig(level=logging.DEBUG)  # Detaylı loglar
logging.basicConfig(level=logging.INFO)   # Normal loglar
logging.basicConfig(level=logging.WARNING) # Sadece uyarılar
```

## 📊 İstatistikler ve Raporlama

### Pipeline Raporu
```bash
python run_complete_pipeline.py
# Çıktı: pipeline_summary_report.json
```

### Sistem İstatistikleri
```python
rag = AgriculturalRAGSystem()
stats = rag.get_statistics()
print(f"Toplam dosya: {stats['unique_files']}")
print(f"Toplam chunk: {stats['total_chunks']}")
print(f"Toplam kelime: {stats['total_words']:,}")
```

## 🔮 Gelecek Geliştirmeler

### Planlanan Özellikler
- [ ] **Google Scholar entegrasyonu**
- [ ] **Crossref API desteği**
- [ ] **DOAJ (Directory of Open Access Journals) entegrasyonu**
- [ ] **Görüntü analizi** (PDF'lerden çıkarılan görseller)
- [ ] **Çok dilli cevap oluşturma**
- [ ] **Atıf ağı analizi**
- [ ] **Trend analizi**
- [ ] **Web arayüzü**

### API Entegrasyonları
- [ ] **PubMed Central (PMC)**
- [ ] **Semantic Scholar**
- [ ] **Microsoft Academic**
- [ ] **YÖK Tez Merkezi**
- [ ] **TÜBİTAK ULAKBİM**

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📞 İletişim

Sorularınız için issue açabilir veya email gönderebilirsiniz.

---

**⚠️ Önemli Not:** Bu sistem gerçek akademik makalelerle çalışır. Telif hakkı kurallarına uygun kullanım yapınız.

**🎯 Hedef:** Bitki patolojisi araştırmalarında akademisyenlere ve araştırmacılara yardımcı olmak. 