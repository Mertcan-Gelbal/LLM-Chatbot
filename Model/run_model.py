#!/usr/bin/env python3
"""
🌱 Botanical BERT Model Çalıştırma
Eğitilmiş modeli kullanarak tahmin yapma scripti
"""

import os
import sys
import json
import torch
import warnings
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification

warnings.filterwarnings('ignore')

class BotanicalBERTPredictor:
    """Botanical BERT Tahmin Sistemi"""
    
    def __init__(self, model_path="botanical_bert_model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Kategori mappings
        self.label2id = {
            'plant_disease': 0,
            'crop_management': 1, 
            'plant_genetics': 2,
            'environmental_factors': 3,
            'food_security': 4,
            'technology': 5
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # Türkçe kategori isimleri
        self.turkish_categories = {
            'plant_disease': 'Bitki Hastalıkları',
            'crop_management': 'Mahsul Yönetimi',
            'plant_genetics': 'Bitki Genetiği', 
            'environmental_factors': 'Çevre Faktörleri',
            'food_security': 'Gıda Güvenliği',
            'technology': 'Tarım Teknolojisi'
        }
        
        self.load_model()
    
    def load_model(self):
        """Model ve tokenizer'ı yükle"""
        print(f"🤖 Model yükleniyor: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            print(f"❌ Model bulunamadı: {self.model_path}")
            print("💡 Önce modeli eğittiğinizden emin olun:")
            print("   cd ../CreateModel && python train_model.py")
            return False
        
        try:
            # Tokenizer yükle
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            
            # Model yükle
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Model bilgilerini yükle
            info_path = os.path.join(self.model_path, "model_info.json")
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    self.model_info = json.load(f)
                print(f"✅ Model yüklendi!")
                print(f"   📊 Accuracy: {self.model_info.get('test_accuracy', 'N/A'):.4f}")
                print(f"   📈 F1 Score: {self.model_info.get('test_f1_score', 'N/A'):.4f}")
                print(f"   💾 Model boyutu: {self.model_info.get('model_size_mb', 'N/A'):.1f}MB")
            else:
                print("✅ Model yüklendi! (Bilgi dosyası bulunamadı)")
                self.model_info = {}
            
            print(f"🎮 Device: {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            return False
    
    def predict_text(self, text, return_probabilities=False):
        """Tek bir text için tahmin yap"""
        if self.model is None:
            return None
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        # Prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            # En yüksek skorlu kategori
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            predicted_label = self.id2label[predicted_id]
            confidence = probabilities[0][predicted_id].item()
        
        result = {
            'text': text,
            'predicted_category': predicted_label,
            'category_turkish': self.turkish_categories[predicted_label],
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        if return_probabilities:
            all_probs = {}
            for i, prob in enumerate(probabilities[0]):
                category = self.id2label[i]
                all_probs[category] = prob.item()
            result['all_probabilities'] = all_probs
        
        return result
    
    def predict_batch(self, texts):
        """Birden çok text için batch prediction"""
        results = []
        for text in texts:
            result = self.predict_text(text)
            if result:
                results.append(result)
        return results
    
    def interactive_demo(self):
        """İnteraktif demo"""
        print("\n🌱" + "="*50 + "🌱")
        print("     BOTANİK BERT İNTERAKTİF DEMO")
        print("🌱" + "="*50 + "🌱")
        print()
        print("💬 Tarımsal sorularınızı yazın (çıkmak için 'exit'):")
        print("📝 Örnek: 'Domates yaprak yanıklığı nasıl tedavi edilir?'")
        print()
        
        conversation_count = 0
        
        while True:
            try:
                user_input = input("\n🌱 Siz: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'çık', 'çıkış']:
                    print("\n👋 Botanical BERT ile görüştüğünüz için teşekkürler!")
                    break
                
                if not user_input:
                    continue
                
                print("🤔 Analiz ediliyor...")
                
                # Prediction
                result = self.predict_text(user_input, return_probabilities=True)
                
                if result:
                    conversation_count += 1
                    
                    print(f"\n🤖 Botanical AI:")
                    print(f"   🎯 Kategori: {result['category_turkish']}")
                    print(f"   📊 Güven: {result['confidence']*100:.1f}%")
                    
                    # Uzman yanıtı
                    expert_responses = {
                        'plant_disease': "🦠 Bitki hastalıkları konusunda detaylı inceleme gerekiyor. Belirtileri, yayılma şeklini ve etkili tedavi yöntemlerini değerlendirmek önemli.",
                        'crop_management': "🌾 Mahsul yönetiminde doğru zamanlama ve metot seçimi kritik. Toprak analizi, iklim koşulları ve sürdürülebilir pratikleri göz önünde bulundurun.",
                        'plant_genetics': "🧬 Bitki genetiği alanında çeşit seçimi ve ıslah programları önemli. Dayanıklılık, verim ve kalite özelliklerini dengelemek gerekiyor.",
                        'environmental_factors': "🌡️ Çevre faktörleri bitkiler üzerinde büyük etki yapar. İklim değişikliği, toprak kalitesi ve su kaynaklarını değerlendirin.",
                        'food_security': "🍽️ Gıda güvenliği için üretimden tüketime kadar tüm süreçleri optimize etmek gerekiyor. Kayıpları minimize edin ve kaliteyi koruyun.",
                        'technology': "🚁 Tarım teknolojileri verimliliği artırır. Sensörlər, AI sistemleri və otomasyon çözümleri ile modern tarıma geçiş yapın."
                    }
                    
                    expert_response = expert_responses.get(result['predicted_category'], 
                                                         "🤖 Genel tarım konusunda daha detaylı bilgi gerekebilir.")
                    print(f"   💡 {expert_response}")
                    
                    # Top 3 kategoriler
                    if result.get('all_probabilities'):
                        print(f"\n   📊 Diğer kategori olasılıkları:")
                        sorted_probs = sorted(result['all_probabilities'].items(), 
                                            key=lambda x: x[1], reverse=True)[:3]
                        for i, (category, prob) in enumerate(sorted_probs, 1):
                            if prob > 0.1:  # %10'dan fazla olanları göster
                                turkish_cat = self.turkish_categories[category]
                                print(f"      {i}. {turkish_cat}: {prob*100:.1f}%")
                else:
                    print("❌ Tahmin yapılamadı!")
                
            except KeyboardInterrupt:
                print("\n\n👋 Çıkış yapılıyor...")
                break
            except Exception as e:
                print(f"❌ Hata: {e}")
        
        print(f"\n📊 Toplam {conversation_count} soru soruldu.")
        print("🌱 Tarım alanında AI destekli çözümler için teşekkürler!")

def predict_text(text):
    """Basit tahmin fonksiyonu (dış kullanım için)"""
    predictor = BotanicalBERTPredictor()
    if predictor.model is None:
        return None
    return predictor.predict_text(text)

def main():
    """Ana fonksiyon"""
    print("🌱 Botanical BERT Model Runner")
    print("="*40)
    
    # Predictor oluştur
    predictor = BotanicalBERTPredictor()
    
    if predictor.model is None:
        print("\n❌ Model yüklenemedi!")
        sys.exit(1)
    
    # Komut satırı argümanlarını kontrol et
    if len(sys.argv) > 1:
        # Direkt tahmin modu
        text = " ".join(sys.argv[1:])
        print(f"\n📝 Text: {text}")
        
        result = predictor.predict_text(text, return_probabilities=True)
        if result:
            print(f"🎯 Kategori: {result['category_turkish']}")
            print(f"📊 Güven: {result['confidence']*100:.1f}%")
            print(f"🔤 İngilizce: {result['predicted_category']}")
        else:
            print("❌ Tahmin yapılamadı!")
    else:
        # Test örnekleri
        test_samples = [
            "Domates bitkilerinde yaprak yanıklığı hastalığı nasıl tedavi edilir?",
            "Buğday ekimi için en uygun toprak hazırlığı nedir?",
            "Genetiği değiştirilmiş mısır çeşitleri hakkında bilgi",
            "İklim değişikliği tarıma nasıl etki ediyor?",
            "Gıda güvenliği için depolama önlemleri",
            "Tarımda drone teknolojisi kullanımı"
        ]
        
        print("\n🧪 Test örnekleri:")
        print("-" * 40)
        
        for i, text in enumerate(test_samples, 1):
            result = predictor.predict_text(text)
            if result:
                print(f"{i}. {text[:40]}...")
                print(f"   → {result['category_turkish']} ({result['confidence']*100:.1f}%)")
            
        # İnteraktif demo başlat
        predictor.interactive_demo()

if __name__ == "__main__":
    main() 