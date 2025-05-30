#!/bin/bash
# 🌾 Tarımsal AI Projesi - Otomatik Test Scripti

echo "🌾 Tarımsal AI Projesi Test Başlıyor..."
echo "========================================"

# Ana dizine git
cd "$(dirname "$0")"

# 1. Bağımlılık kontrolü
echo "📦 Bağımlılıklar kontrol ediliyor..."
pip install -r requirements.txt

# 2. Basit BERT modeli eğit
echo "🧠 BERT modeli eğitiliyor..."
cd 02_models/bert_classification/
python simple_agricultural_bert.py

# 3. Chatbot hızlı test (opsiyonel - script modunda)
echo "💬 Chatbot test ediliyor..."
cd ../../04_deployment/chatbots/
echo "quit" | python simple_agricultural_chatbot.py &
CHATBOT_PID=$!
sleep 5
kill $CHATBOT_PID 2>/dev/null

# 4. Performans karşılaştırması
echo "📊 Performans test ediliyor..."
cd ../../03_training_results/
python compare_all_models.py

# 5. Sonuçları kontrol et
echo "📋 Sonuçlar kontrol ediliyor..."
if [ -f "performance_metrics/comparison_results.json" ]; then
    echo "✅ Test sonuçları başarıyla oluşturuldu!"
    echo "📁 Sonuç dosyası: performance_metrics/comparison_results.json"
else
    echo "❌ Test sonuçları oluşturulamadı!"
    exit 1
fi

echo "🎉 Tüm testler tamamlandı!"
echo "📚 Detaylı rapor için: 05_documentation/technical_report.md" 