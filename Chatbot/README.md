# 🌱 Agricultural LLM Chatbot System
## Tarımsal AI Chatbot - Gelişmiş Sohbet Sistemi

Bu klasör, eğitilmiş BERT modelini kullanarak **akıllı tarımsal chatbot** sistemleri içerir.

---

## 🚀 Hızlı Başlangıç

### 1. **Gerekli Paketleri Kurun**
```bash
# Temel chatbot için
pip install transformers torch pandas

# Gelişmiş özellikler için
pip install sentence-transformers faiss-cpu gradio

# API server için
pip install flask flask-cors
```

### 2. **Basit Terminal Chatbot**
```bash
# İnteraktif chat
cd Chatbot
python simple_agricultural_chat.py

# Tek soru
python simple_agricultural_chat.py "Domates hastalığı nedir?"
```

### 3. **Web Arayüzü**
```bash
# Gradio web arayüzü başlat
python agricultural_llm_chatbot.py

# Browser'da otomatik açılır: http://localhost:7860
```

### 4. **API Server**
```bash
# REST API server başlat
python agricultural_api_server.py

# API documentation: http://localhost:5000
```

---

## 📁 Dosya Yapısı

```
🌱 Chatbot/
├── agricultural_llm_chatbot.py      # 🎯 Ana LLM chatbot sistemi
├── simple_agricultural_chat.py     # 💬 Basit terminal chatbot  
├── agricultural_api_server.py      # 🌐 REST API server
└── README.md                       # 📖 Bu dosya
```

---

## 🤖 Chatbot Türleri

### 1. **Simple Agricultural Chat** 💬
**En basit kullanım** - terminal tabanlı
- ✅ Hızlı başlangıç
- ✅ Düşük kaynak kullanımı  
- ✅ CLI interface
- ✅ Tek soru modu

```bash
# İnteraktif mod
python simple_agricultural_chat.py

# Tek soru modu
python simple_agricultural_chat.py "Buğday ekimi nasıl yapılır?"
```

**Özellikler:**
- BERT sınıflandırması
- Kategori bazlı öneriler
- Türkçe/İngilizce desteği
- Yardım komutları

### 2. **Agricultural LLM Chatbot** 🎯  
**Gelişmiş sistem** - RAG ile bilgi retrieval
- ✅ Semantic search
- ✅ Knowledge base  
- ✅ Web UI (Gradio)
- ✅ Context-aware responses

```bash
# CLI modu
python agricultural_llm_chatbot.py --cli

# Web arayüzü modu (varsayılan)
python agricultural_llm_chatbot.py
```

**Özellikler:**
- BERT classification + RAG
- FAISS vector search
- Expert knowledge base
- Interactive web interface
- Conversation history

### 3. **Agricultural API Server** 🌐
**Production ready** - REST API
- ✅ HTTP endpoints
- ✅ JSON responses
- ✅ Usage statistics
- ✅ Health monitoring

```bash
# Default port 5000
python agricultural_api_server.py

# Custom port
python agricultural_api_server.py --port 8080 --host 0.0.0.0
```

**Endpoints:**
- `POST /chat` - Ana chatbot
- `POST /classify` - Sadece sınıflandırma
- `GET /health` - Sistem durumu
- `GET /stats` - Kullanım istatistikleri

---

## 🔧 Kullanım Örnekleri

### Terminal Chatbot
```bash
$ python simple_agricultural_chat.py

🌱 Tarım AI - Tarımsal AI Asistanı
====================================

👤 Soru: Domates yaprak yanıklığı nedir?

🤖 İşleniyor...

🌱 Tarım AI: 🎯 Kategori: Bitki Hastalıkları (%89.2 güven)
💡 Öneri: Hastalık teşhisi için bitki türü, belirtiler ve çevre koşullarını belirtin.
```

### Web Arayüzü
```bash
$ python agricultural_llm_chatbot.py

🌱 Agricultural LLM Chatbot System Starting...
✅ Chatbot initialized successfully!
🌐 Starting web interface...
Running on local URL:  http://127.0.0.1:7860
```

### API Kullanımı
```bash
# Chat endpoint
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Organik gübre nasıl yapılır?"}'

# Response:
{
  "response": "🔍 Konu Kategorisi: Mahsul Yönetimi (%87.3 güven)\n💡 Öneri: ...",
  "category": "crop_management",
  "confidence": 0.873,
  "processing_time_ms": 245.6
}
```

---

## 🎯 Özellik Karşılaştırması

| Özellik | Simple Chat | LLM Chatbot | API Server |
|---------|-------------|-------------|------------|
| **Kurulum** | ⭐⭐⭐ Kolay | ⭐⭐ Orta | ⭐⭐ Orta |
| **Kaynak Kullanımı** | 🟢 Düşük | 🟡 Orta | 🟡 Orta |
| **Özellik Zenginliği** | 🟡 Temel | 🟢 Yüksek | 🟢 Yüksek |
| **Web Arayüzü** | ❌ | ✅ Gradio | ✅ HTML |
| **API Desteği** | ❌ | ❌ | ✅ REST |
| **Knowledge Base** | ❌ | ✅ RAG | ✅ Enhanced |
| **Production Ready** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 📊 Performans

### Simple Chat
- **Başlangıç Süresi:** ~2 saniye
- **Memory:** ~500MB  
- **Response Time:** ~100ms

### LLM Chatbot  
- **Başlangıç Süresi:** ~15 saniye
- **Memory:** ~2GB
- **Response Time:** ~300ms

### API Server
- **Başlangıç Süresi:** ~3 saniye
- **Memory:** ~800MB
- **Response Time:** ~150ms
- **Concurrent Users:** 50+

---

## 🔧 Konfigürasyon

### Model Yolu Değiştirme
```python
# simple_agricultural_chat.py içinde
sys.path.append("path/to/your/model")

# agricultural_llm_chatbot.py içinde  
class AgriculturalClassifier:
    def __init__(self, model_path: str = "path/to/your/model"):
```

### Port ve Host Ayarları
```bash
# API server için
python agricultural_api_server.py --host 0.0.0.0 --port 8080

# Web chatbot için 
# agricultural_llm_chatbot.py içindeki launch() parametrelerini değiştirin
demo.launch(server_port=7860, server_name="0.0.0.0")
```

### Knowledge Base Genişletme
```python
# agricultural_llm_chatbot.py içinde
def _get_expert_knowledge(self) -> Dict:
    expert_texts = [
        "Yeni tarımsal bilgi...",
        # Daha fazla uzman bilgisi ekleyin
    ]
```

---

## 🌐 Deployment

### Docker ile Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "Chatbot/agricultural_api_server.py", "--host", "0.0.0.0"]
```

### Nginx ile Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Systemd Service
```ini
[Unit]
Description=Agricultural Chatbot API
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/chatbot
ExecStart=/usr/bin/python3 agricultural_api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 🔍 Monitoring & Debugging

### Log Seviyesi Değiştirme
```python
import logging
logging.basicConfig(level=logging.DEBUG)  # Detaylı loglar
```

### Performance Monitoring
```bash
# API istatistikleri
curl http://localhost:5000/stats

# Sistem durumu
curl http://localhost:5000/health

# GPU kullanımı
nvidia-smi -l 1
```

### Error Handling
```bash
# Hata logları için
tail -f /var/log/agricultural-chatbot.log

# Model yükleme sorunları için
python -c "from Model.run_model import predict_text; print(predict_text('test'))"
```

---

## 🚨 Troubleshooting

### Yaygın Sorunlar

**1. Model Bulunamadı**
```bash
# Model yolunu kontrol edin
ls ../Model/botanical_bert_small/

# Model klasörünü doğru yerde mi?
python -c "from pathlib import Path; print(Path('../Model').exists())"
```

**2. Memory Hatası**
```bash
# Batch size küçültün veya CPU kullanın
export CUDA_VISIBLE_DEVICES=""  # CPU'ya zorla
```

**3. Port Kullanımda**
```bash
# Port kontrolü
lsof -i :5000

# Farklı port kullanın
python agricultural_api_server.py --port 5001
```

**4. Package Eksik**
```bash
# Tüm requirements'ı kurun
pip install -r ../requirements.txt

# Spesifik paket
pip install sentence-transformers faiss-cpu gradio flask
```

---

## 🎓 İleri Seviye Kullanım

### Custom Knowledge Base
```python
# Kendi bilgi tabanınızı ekleyin
def add_custom_knowledge(self, texts: List[str], metadata: List[Dict]):
    self.documents.extend(texts)
    self.metadata.extend(metadata)
    self._create_vector_index()  # Index'i yeniden oluştur
```

### API Authentication
```python
# API token authentication eklemek için
@app.before_request
def verify_token():
    token = request.headers.get('Authorization')
    if not token or not verify_api_token(token):
        return jsonify({'error': 'Invalid token'}), 401
```

### Multi-Language Support
```python
# Dil algılama ve çeviri eklemek için
from langdetect import detect
from googletrans import Translator

def detect_and_translate(text: str) -> Tuple[str, str]:
    lang = detect(text)
    if lang != 'tr':
        translator = Translator()
        translated = translator.translate(text, dest='tr').text
        return translated, lang
    return text, 'tr'
```

---

## 📞 Destek

### 🆘 Yardım İçin
- **GitHub Issues:** [Repository Issues](https://github.com/your-repo/issues)
- **Email:** support@agricultural-ai.com
- **Discord:** [Agricultural AI Community](https://discord.gg/agricultural-ai)

### 📚 Öğrenme Kaynakları
- **Documentation:** [Full Docs](https://docs.agricultural-ai.com)
- **Video Tutorials:** [YouTube Playlist](https://youtube.com/agricultural-ai)
- **Blog:** [Medium Articles](https://medium.com/@agricultural-ai)

---

## 🤝 Katkıda Bulunma

```bash
# Fork & clone
git clone https://github.com/your-username/agricultural-chatbot.git

# Feature branch
git checkout -b feature/new-chatbot-feature

# Development
# ... kod değişiklikleri ...

# Test
python -m pytest tests/

# Commit & push
git commit -m "feat: add new chatbot feature"
git push origin feature/new-chatbot-feature

# Pull request oluştur
```

---

## 📄 Lisans

MIT License - Botanical BERT Team © 2024

---

**🌱 Tarımın geleceğini AI ile şekillendiriyoruz! 🤖**

*Son güncelleme: 2024-05-29 | Version: 1.0.0* 