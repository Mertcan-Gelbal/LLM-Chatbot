#!/usr/bin/env python3
"""
Basit Tarımsal BERT Chatbot
Eğitilmiş modeli kullanarak basit soru-cevap
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from pathlib import Path
import json

class SimpleAgriculturalChatbot:
    def __init__(self, model_path="agricultural_bert_base_uncased"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        
        # Kategori bilgileri
        self.categories = {
            0: "plant_disease",
            1: "crop_management", 
            2: "environmental_factors"
        }
        
        # Kategori cevapları
        self.responses = {
            "plant_disease": {
                "elma": "Elmada erken yanıklığı bakteriyel bir hastalıktır. Hasta dalları kesin, sterilize edin, antibiyotik sprey uygulayın.",
                "domates": "Domates sarı yaprak sorunu beslenme eksikliği olabilir. Azotlu gübre uygulayın, sulama düzenini kontrol edin.",
                "genel": "Bitki hastalıkları için erken teşhis önemlidir. Hasta kısımları temizleyin ve koruyucu ilaçlama yapın."
            },
            "crop_management": {
                "buğday": "Buğday ekim zamanı toprak sıcaklığına bağlıdır. Kışlık buğday Ekim-Kasım, yazlık buğday Mart-Nisan aylarında ekilir.",
                "havuç": "Havuç için derin, gevşek toprak gerekir. Ekim derinliği 1-2 cm, düzenli sulama yapın.",
                "sulama": "Düzenli sulama önemlidir. Aşırı su vermeyin, toprak nemini kontrol edin.",
                "genel": "Doğru ekim zamanı, toprak hazırlığı ve gübreleme başarının anahtarıdır."
            },
            "environmental_factors": {
                "ph": "Toprak pH'sı 6.0-7.0 arasında ideal. Asit topraklar kireçleme, alkalin topraklar sülfür ile düzeltilir.",
                "sıcaklık": "Aşırı sıcaklık bitki stresine neden olur. Gölgeleme, mulch ve sık sulama uygulayın.",
                "genel": "Çevre faktörleri bitki gelişimini doğrudan etkiler. Kontrollü koşullar sağlamaya çalışın."
            }
        }
        
        self.load_model()
    
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        try:
            if self.model_path.exists():
                print(f"📁 Model yükleniyor: {self.model_path}")
                
                # Tokenizer yükle
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                
                # Model config yükle
                config = BertConfig.from_pretrained(self.model_path)
                
                # Model oluştur ve state dict yükle
                self.model = BertForSequenceClassification(config)
                state_dict = torch.load(self.model_path / "pytorch_model.bin", map_location=self.device)
                self.model.load_state_dict(state_dict)
                
                self.model.to(self.device)
                self.model.eval()
                print("✅ Model başarıyla yüklendi!")
            else:
                print(f"❌ Model bulunamadı: {self.model_path}")
                print("Lütfen önce simple_agricultural_bert.py çalıştırın.")
                raise FileNotFoundError
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            raise
    
    def predict_category(self, text):
        """Metin kategorisini tahmin et"""
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Tahmin
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item()
        
        category = self.categories[pred_id]
        return category, confidence
    
    def generate_response(self, question):
        """Soruya cevap üret"""
        # Kategori tahmin et
        category, confidence = self.predict_category(question)
        
        # Anahtar kelime ara
        question_lower = question.lower()
        responses = self.responses[category]
        
        # Spesifik cevap ara
        for keyword, response in responses.items():
            if keyword != "genel" and keyword in question_lower:
                return f"**{category}** (Güven: {confidence:.2f})\n\n{response}"
        
        # Genel cevap ver
        general_response = responses["genel"]
        return f"**{category}** (Güven: {confidence:.2f})\n\n{general_response}"
    
    def chat(self):
        """Sohbet döngüsü"""
        print("\n🌾 Tarımsal BERT Chatbot'a Hoş Geldiniz!")
        print("💬 Tarımsal sorularınızı sorun (çıkmak için 'quit' yazın)")
        print("-" * 50)
        
        while True:
            question = input("\n❓ Soru: ").strip()
            
            if question.lower() in ['quit', 'exit', 'çıkış']:
                print("👋 Hoşçakalın!")
                break
            
            if not question:
                continue
            
            try:
                response = self.generate_response(question)
                print(f"\n🤖 Cevap: {response}")
            except Exception as e:
                print(f"❌ Hata: {e}")

def main():
    """Ana fonksiyon"""
    try:
        chatbot = SimpleAgriculturalChatbot()
        chatbot.chat()
    except Exception as e:
        print(f"❌ Chatbot başlatılamadı: {e}")

if __name__ == "__main__":
    main() 