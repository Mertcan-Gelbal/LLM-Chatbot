# 🚀 Nasıl Çalıştırılır? - Hızlı Kılavuz

## ⚡ Tek Komutla Çalıştırma (Önerilen)

```bash
python3 quick_start.py
```

Bu komut:
- ✅ Sistem gereksinimlerini kontrol eder
- ✅ Eksik paketleri otomatik kurar  
- ✅ Web uygulamasını tarayıcıda açar

---

## 📋 Manuel Adım-Adım Kurulum

### 1. Bağımlılıkları Kurun
```bash
python3 -m pip install streamlit pandas plotly numpy
```

### 2. Uygulamayı Başlatın
```bash
python3 -m streamlit run demo_app.py
```

### 3. Tarayıcıda Açın
```
http://localhost:8501
```

---

## 🎯 Ne Yapacaksınız?

1. **Demo uygulaması açılacak** - Modern web arayüzü
2. **Tarımsal sorular sorun** - "Domates hastalıkları nelerdir?"
3. **AI yanıtları alın** - Kategori ve güven puanı ile
4. **İstatistikleri görün** - Soru dağılımı grafikleri

---

## 🐛 Sorun mu Yaşıyorsunuz?

### Hata: "pip command not found"
```bash
# Çözüm:
python3 -m pip install streamlit
```

### Hata: "streamlit command not found"  
```bash
# Çözüm:
python3 -m streamlit run demo_app.py
```

### Hata: "Port already in use"
```bash
# Çözüm: Farklı port kullanın
python3 -m streamlit run demo_app.py --server.port 8502
```

---

## 🔧 Alternatif Çalıştırma Yöntemleri

### A) Terminal Chatbot
```bash
cd Chatbot
python3 simple_agricultural_chat.py
```

### B) API Server
```bash
cd Chatbot  
python3 agricultural_api_server.py
```

### C) Tek Soru Modu
```bash
python3 Chatbot/simple_agricultural_chat.py "Buğday ne zaman ekilir?"
```

---

## 📱 Web Arayüzü Kullanımı

1. **Soru Sorun**: Alt kısımdaki chat kutusuna yazın
2. **Örnek Seçin**: Sol menüden hazır örnekleri tıklayın  
3. **Sonuçları Görün**: AI yanıtı ve güven puanı
4. **İstatistikleri İnceleyin**: Sağ panelde grafikler

### 💡 Örnek Sorular:
- "Domates hastalıkları nelerdir?"
- "Buğday ne zaman ekilir?"
- "Organik gübre çeşitleri neler?"
- "Sulama sistemleri nasıl çalışır?"

---

## ✅ Başarı Kontrol Listesi

- [ ] Python 3.9+ yüklü
- [ ] `python3 quick_start.py` çalıştırıldı
- [ ] Tarayıcıda uygulama açıldı
- [ ] Soru-cevap testi yapıldı

---

## 📞 Yardım Gerekli mi?

```bash
# Yardım menüsü:
python3 quick_start.py --help

# Sistem bilgileri:
python3 --version
python3 -m pip list
```

**🎉 Bu kadar! Tarımsal AI Chatbot kullanıma hazır!** 