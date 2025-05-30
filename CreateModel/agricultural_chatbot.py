#!/usr/bin/env python3
"""
Tarımsal AI Sohbet Botu
Eğitilmiş BERT-small ve DistilBERT modellerini test etmek için interaktif sohbet arayüzü
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from transformers import (
    BertTokenizer, BertForSequenceClassification, BertConfig,
    DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.text import Text
from rich import print as rprint

console = Console()

class AgriculturalChatbot:
    """Tarımsal AI Sohbet Botu"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model paths
        self.bert_small_path = Path("bert_small_agricultural")
        self.distilbert_path = Path("distilbert_agricultural")
        
        # Models and tokenizers
        self.bert_small_model = None
        self.bert_small_tokenizer = None
        self.distilbert_model = None
        self.distilbert_tokenizer = None
        
        # Label mappings
        self.label_mapping = None
        
        # Kategori açıklamaları
        self.category_info = {
            'crop_management': {
                'name': 'Mahsul Yönetimi',
                'description': 'Ekim, dikim, budama, hasat gibi tarımsal işlemler',
                'examples': 'Tohum ekimi, sulama, gübreleme, hasat zamanı',
                'tips': '🌱 Doğru ekim zamanı ve toprak hazırlığı çok önemlidir'
            },
            'plant_disease': {
                'name': 'Bitki Hastalıkları', 
                'description': 'Bitkileri etkileyen hastalık ve zararlılar',
                'examples': 'Fungal enfeksiyonlar, böcek zararları, viral hastalıklar',
                'tips': '🔍 Erken teşhis ve önleyici tedbirler kritiktir'
            },
            'plant_genetics': {
                'name': 'Bitki Genetiği',
                'description': 'Bitki ıslahı, genetik çeşitlilik ve kalıtım',
                'examples': 'Hibrit çeşitler, gen modifikasyonu, tohum ıslahı',
                'tips': '🧬 Genetik çeşitlilik sürdürülebilir tarım için önemlidir'
            },
            'environmental_factors': {
                'name': 'Çevresel Faktörler',
                'description': 'İklim, toprak, su ve çevre koşulları',
                'examples': 'Toprak pH\'ı, iklim değişikliği, su kaynakları',
                'tips': '🌍 Çevre koşullarına uyum sağlamak gereklidir'
            },
            'food_security': {
                'name': 'Gıda Güvenliği',
                'description': 'Gıda üretimi, dağıtımı ve erişilebilirliği',
                'examples': 'Gıda üretim kapasitesi, beslenme, gıda kayıpları',
                'tips': '🍽️ Sürdürülebilir gıda sistemleri gelecek için kritiktir'
            },
            'technology': {
                'name': 'Tarım Teknolojisi',
                'description': 'Modern tarımda kullanılan teknolojiler',
                'examples': 'Drone\'lar, sensörler, akıllı sulama, GPS',
                'tips': '💻 Teknoloji tarımsal verimliliği artırır'
            },
            'general_agriculture': {
                'name': 'Genel Tarım',
                'description': 'Tarımla ilgili genel konular ve uygulamalar',
                'examples': 'Tarım politikaları, geleneksel yöntemler',
                'tips': '🚜 Temel tarım bilgisi her çiftçi için önemlidir'
            }
        }
        
        console.print("🤖 Tarımsal AI Sohbet Botu başlatılıyor...", style="bold green")
        self.load_models()
    
    def load_models(self):
        """Eğitilmiş modelleri yükle"""
        console.print("📂 Modeller yükleniyor...", style="bold blue")
        
        try:
            # Label mapping yükle
            with open(self.bert_small_path / 'label_mapping.json', 'r') as f:
                self.label_mapping = json.load(f)
            
            # BERT-small model yükle
            console.print("🤖 BERT-small modeli yükleniyor...")
            
            # Config dosyasından model config'i yükle
            with open(self.bert_small_path / 'config.json', 'r') as f:
                bert_config_dict = json.load(f)
            bert_config = BertConfig.from_dict(bert_config_dict)
            
            self.bert_small_model = BertForSequenceClassification(bert_config)
            
            # State dict yükle
            state_dict = torch.load(self.bert_small_path / 'pytorch_model.bin', 
                                  map_location=self.device, weights_only=False)
            self.bert_small_model.load_state_dict(state_dict)
            self.bert_small_model.to(self.device)
            self.bert_small_model.eval()
            
            self.bert_small_tokenizer = BertTokenizer.from_pretrained(str(self.bert_small_path))
            
            console.print("✅ BERT-small yüklendi!", style="green")
            
            # DistilBERT model yükle
            console.print("🚀 DistilBERT modeli yükleniyor...")
            
            # DistilBERT için config yükle
            with open(self.distilbert_path / 'config.json', 'r') as f:
                distil_config_dict = json.load(f)
            
            distil_config = DistilBertConfig.from_dict(distil_config_dict)
            
            self.distilbert_model = DistilBertForSequenceClassification(distil_config)
            
            # State dict yükle
            distil_state_dict = torch.load(self.distilbert_path / 'pytorch_model.bin', 
                                         map_location=self.device, weights_only=False)
            self.distilbert_model.load_state_dict(distil_state_dict)
            self.distilbert_model.to(self.device)
            self.distilbert_model.eval()
            
            self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained(str(self.distilbert_path))
            
            console.print("✅ DistilBERT yüklendi!", style="green")
            console.print("🎉 Tüm modeller hazır!", style="bold green")
            
        except Exception as e:
            console.print(f"❌ Model yükleme hatası: {e}", style="bold red")
            console.print("🔧 Model dosyalarını kontrol edin", style="yellow")
            raise
    
    def predict_category(self, text: str) -> Dict:
        """Her iki model ile kategori tahmini yap"""
        results = {}
        
        # BERT-small prediction
        try:
            inputs = self.bert_small_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_small_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                predicted_id = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_id].item()
            
            predicted_label = self.label_mapping['id_to_label'][str(predicted_id)]
            
            results['bert_small'] = {
                'category': predicted_label,
                'confidence': confidence,
                'probabilities': probs[0].cpu().numpy()
            }
        except Exception as e:
            results['bert_small'] = {'error': str(e)}
        
        # DistilBERT prediction
        try:
            inputs = self.distilbert_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.distilbert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                predicted_id = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_id].item()
            
            predicted_label = self.label_mapping['id_to_label'][str(predicted_id)]
            
            results['distilbert'] = {
                'category': predicted_label,
                'confidence': confidence,
                'probabilities': probs[0].cpu().numpy()
            }
        except Exception as e:
            results['distilbert'] = {'error': str(e)}
        
        return results
    
    def create_results_table(self, text: str, results: Dict) -> Table:
        """Sonuçları tablo formatında göster"""
        table = Table(title=f"🔍 Analiz Sonuçları: '{text[:50]}...'", show_header=True)
        table.add_column("Model", style="cyan", width=15)
        table.add_column("Kategori", style="green", width=20)
        table.add_column("Güven", style="yellow", width=10)
        table.add_column("Açıklama", style="white", width=40)
        
        for model_name, result in results.items():
            if 'error' in result:
                table.add_row(
                    model_name.upper(),
                    "❌ Hata",
                    "0%",
                    result['error'][:40]
                )
            else:
                category = result['category']
                confidence = f"{result['confidence']*100:.1f}%"
                description = self.category_info.get(category, {}).get('name', category)
                
                table.add_row(
                    model_name.upper(),
                    description,
                    confidence,
                    self.category_info.get(category, {}).get('description', '')[:40]
                )
        
        return table
    
    def get_category_advice(self, category: str) -> str:
        """Kategoriye göre tavsiye ver"""
        if category in self.category_info:
            info = self.category_info[category]
            advice = f"""
🌟 **{info['name']}** hakkında bilgiler:

📋 **Açıklama:** {info['description']}

💡 **Örnekler:** {info['examples']}

🎯 **Tavsiye:** {info['tips']}
"""
            return advice
        return "Bu kategori hakkında detaylı bilgi bulunamadı."
    
    def chat_loop(self):
        """Ana sohbet döngüsü"""
        console.print(Panel.fit(
            "🌾 Tarımsal AI Sohbet Botu'na Hoş Geldiniz! 🌾\n\n"
            "• Tarımsal konularda sorularınızı sorabilirsiniz\n"
            "• Her iki model de sorunuzu analiz edecek\n"
            "• 'çıkış' yazarak programı kapatabilirsiniz\n"
            "• 'help' yazarak yardım alabilirsiniz",
            style="bold green"
        ))
        
        while True:
            try:
                # Kullanıcı girişi al
                user_input = Prompt.ask(
                    "\n🌱 Tarımsal sorunuzu yazın",
                    default="",
                ).strip()
                
                if not user_input:
                    continue
                
                # Özel komutlar
                if user_input.lower() in ['çıkış', 'exit', 'quit', 'q']:
                    console.print("👋 Görüşmek üzere! İyi tarımlar!", style="bold green")
                    break
                
                if user_input.lower() in ['help', 'yardım']:
                    self.show_help()
                    continue
                
                if user_input.lower() in ['kategoriler', 'categories']:
                    self.show_categories()
                    continue
                
                # Model predictions
                console.print("\n🔄 Modeller analiz ediyor...", style="bold yellow")
                
                results = self.predict_category(user_input)
                
                # Sonuçları göster
                table = self.create_results_table(user_input, results)
                console.print(table)
                
                # En yüksek güvenli tahmini bul
                best_result = None
                best_confidence = 0
                
                for model_name, result in results.items():
                    if 'error' not in result and result['confidence'] > best_confidence:
                        best_confidence = result['confidence']
                        best_result = result
                
                if best_result:
                    console.print(f"\n📊 En güvenli tahmin: **{self.category_info[best_result['category']]['name']}** "
                                f"(Güven: {best_confidence*100:.1f}%)", style="bold blue")
                    
                    # Kategori tavsiyeleri
                    advice = self.get_category_advice(best_result['category'])
                    console.print(Panel(advice, title="💡 Kategori Bilgileri", style="blue"))
                
            except KeyboardInterrupt:
                console.print("\n👋 Görüşmek üzere!", style="bold green")
                break
            except Exception as e:
                console.print(f"❌ Hata: {e}", style="bold red")
    
    def show_help(self):
        """Yardım bilgilerini göster"""
        help_text = """
🔧 **Komutlar:**
• Normal metin yazın → Model analizi yapar
• 'kategoriler' → Mevcut kategorileri listeler  
• 'help' veya 'yardım' → Bu yardım mesajını gösterir
• 'çıkış' veya 'exit' → Programdan çıkar

💡 **İpuçları:**
• Açık ve net sorular sorun
• Tarımsal terimler kullanın
• Hem Türkçe hem İngilizce desteklenir
"""
        console.print(Panel(help_text, title="🆘 Yardım", style="cyan"))
    
    def show_categories(self):
        """Kategorileri göster"""
        table = Table(title="🗂️ Tarımsal Kategoriler", show_header=True)
        table.add_column("Kategori", style="green", width=20)
        table.add_column("Açıklama", style="white", width=50)
        
        for key, info in self.category_info.items():
            table.add_row(info['name'], info['description'])
        
        console.print(table)

def main():
    """Ana fonksiyon"""
    try:
        chatbot = AgriculturalChatbot()
        chatbot.chat_loop()
    except Exception as e:
        console.print(f"❌ Program başlatılamadı: {e}", style="bold red")

if __name__ == "__main__":
    main() 