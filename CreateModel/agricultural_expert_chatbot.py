#!/usr/bin/env python3
"""
Gerçek Tarımsal Uzman Sohbet Botu
Sorunuza göre detaylı, pratik tavsiyeler veren akıllı tarım uzmanı
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import random
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

class AgriculturalExpertChatbot:
    """Gerçek Tarımsal Uzman Sohbet Botu"""
    
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
        
        # Uzman bilgi bankası
        self.expert_knowledge = {
            'crop_management': {
                'name': 'Mahsul Yönetimi Uzmanı',
                'greeting': "Merhaba! Ben tarımsal mahsul yönetimi uzmanınızım. 🌱",
                'responses': {
                    'genel': [
                        "Mahsul yönetimi için temel prensipler şunlardır:",
                        "✅ Doğru tohum seçimi ve kaliteli tohum kullanımı",
                        "✅ Toprak analizi yaparak uygun gübreleme programı",
                        "✅ Düzenli sulama ve nem kontrolü",
                        "✅ Zamanında ekim, bakım ve hasat",
                        "✅ Hastalık ve zararlı takibi"
                    ],
                    'ekim': [
                        "Ekim zamanlaması kritik önem taşır:",
                        "🌡️ Toprak sıcaklığının 8-10°C olması gerekir",
                        "💧 Toprak neminin %60-70 olması ideal",
                        "📅 Yerel iklim koşullarına uygun ekim takvimi",
                        "🌱 Tohum derinliği tohum boyutunun 2-3 katı olmalı"
                    ],
                    'sulama': [
                        "Sulama stratejiniz şu şekilde olmalı:",
                        "💧 Toprak nem seviyesini düzenli kontrol edin",
                        "⏰ Sabah erken saatlerde sulama yapın",
                        "🌱 Bitki büyüme dönemlerine göre su ihtiyacını ayarlayın",
                        "💡 Damla sulama sistemi en verimli yöntemdir"
                    ]
                }
            },
            'plant_disease': {
                'name': 'Bitki Hastalıkları Uzmanı',
                'greeting': "Merhaba! Ben bitki hastalıkları ve zararlılar uzmanınızım. 🔍",
                'responses': {
                    'genel': [
                        "Bitki hastalıklarında erken teşhis çok önemlidir:",
                        "🔍 Günlük bitki kontrolü yapın",
                        "🌡️ Nem ve sıcaklık takibi yapın",
                        "🧪 Gerektiğinde laboratuvar analizi yaptırın",
                        "💊 Organik önlemler tercih edin",
                        "⚠️ Kimyasal mücadeleyi son çare olarak görün"
                    ],
                    'fungal': [
                        "Mantar hastalıkları için önlemler:",
                        "🌪️ Hava sirkülasyonunu iyileştirin",
                        "💧 Yaprakları ıslatmamaya dikkat edin",
                        "🧄 Organik fungisitler kullanın (neem yağı, bakır sülfat)",
                        "🗑️ Hasta bitki parçalarını derhal uzaklaştırın"
                    ],
                    'yapraksarı': [
                        "Sarı yapraklar için kontrol edilecekler:",
                        "💧 Aşırı veya az sulama kontrolü",
                        "🌱 Azot eksikliği - yaprak gübresi uygulayın",
                        "🔍 Kök çürüklüğü kontrol edin",
                        "🐛 Zararlı kontrolü yapın",
                        "⚗️ Toprak pH seviyesini ölçün (6.0-7.0 ideal)"
                    ]
                }
            },
            'plant_genetics': {
                'name': 'Bitki Genetiği Uzmanı',
                'greeting': "Merhaba! Ben bitki genetiği ve ıslah uzmanınızım. 🧬",
                'responses': {
                    'genel': [
                        "Bitki ıslahı modern tarımın temelidir:",
                        "🧬 Genetik çeşitlilik korunmalıdır",
                        "🌱 Yerel çeşitler değerli genetik kaynaklardır",
                        "🔬 Hibrit çeşitler verim artışı sağlar",
                        "🌍 İklim değişikliğine uyumlu çeşitler geliştirilmelidir"
                    ],
                    'hibrit': [
                        "Hibrit çeşit geliştirme süreci:",
                        "👥 Ebeveyn hatların seçimi ve karakterizasyonu",
                        "💑 Kontrollü melezleme çalışmaları",
                        "🧪 F1 hibrit performans testleri",
                        "📊 Çok çevreli verim denemeleri",
                        "✅ Tescil ve üretim süreçleri"
                    ]
                }
            },
            'environmental_factors': {
                'name': 'Çevre Uzmanı',
                'greeting': "Merhaba! Ben tarımsal çevre koşulları uzmanınızım. 🌍",
                'responses': {
                    'genel': [
                        "Çevre faktörleri tarımsal başarının anahtarıdır:",
                        "🌡️ Sıcaklık ve nem takibi yapın",
                        "🌧️ Yağış durumunu izleyin",
                        "🌱 Toprak analizlerini düzenli yaptırın",
                        "💨 Rüzgar ve erozyon önlemleri alın"
                    ],
                    'toprak': [
                        "Toprak sağlığı için temel uygulamalar:",
                        "⚗️ pH seviyesi 6.0-7.5 arasında tutun",
                        "🌿 Organik madde %2-4 seviyesinde olmalı",
                        "💧 Drenaj sistemi uygun olmalı",
                        "🔄 Ekim nöbeti uygulayın",
                        "🌱 Toprak örtüsü kullanın"
                    ]
                }
            },
            'food_security': {
                'name': 'Gıda Güvenliği Uzmanı',
                'greeting': "Merhaba! Ben gıda güvenliği uzmanınızım. 🍽️",
                'responses': {
                    'genel': [
                        "Gıda güvenliği kapsamlı bir yaklaşım gerektirir:",
                        "📈 Verim artırıcı sürdürülebilir yöntemler",
                        "🗑️ Hasat sonrası kayıpları minimize edin",
                        "❄️ Uygun depolama koşulları sağlayın",
                        "🚚 Soğuk zincir sistemi kurın",
                        "🧪 Kalite kontrol sistemleri uygulayın"
                    ],
                    'depolama': [
                        "Ürün depolama için kritik faktörler:",
                        "🌡️ Uygun sıcaklık (ürüne göre 0-15°C)",
                        "💧 Nem kontrolü (%85-95 relatif nem)",
                        "🌪️ Hava sirkülasyonu sağlayın",
                        "🧽 Temiz ve hijyenik ortam",
                        "🔍 Düzenli kontrol ve ayıklama"
                    ]
                }
            },
            'technology': {
                'name': 'Tarım Teknolojisi Uzmanı',
                'greeting': "Merhaba! Ben tarım teknolojileri uzmanınızım. 💻",
                'responses': {
                    'genel': [
                        "Modern tarım teknolojileri verimliliği artırır:",
                        "🚁 Drone ile alan taraması ve analiz",
                        "📡 IoT sensörleri ile gerçek zamanlı takip",
                        "🤖 Otomasyon sistemleri",
                        "📱 Akıllı tarım uygulamaları",
                        "🛰️ GPS ile hassas tarım"
                    ],
                    'drone': [
                        "Drone kullanımının faydaları:",
                        "📸 Yüksek çözünürlüklü alan görüntüleme",
                        "🌱 Bitki sağlığı analizi (NDVI)",
                        "💧 Su stresi tespiti",
                        "🐛 Hastalık ve zararlı erken uyarı",
                        "💊 Hassas ilaçlama uygulaması"
                    ]
                }
            },
            'general_agriculture': {
                'name': 'Genel Tarım Uzmanı',
                'greeting': "Merhaba! Ben genel tarım uygulamaları uzmanınızım. 🚜",
                'responses': {
                    'genel': [
                        "Başarılı tarım için temel prensipler:",
                        "📅 Tarım takvimi ve planlama",
                        "💰 Maliyet-fayda analizi",
                        "🌱 Sürdürülebilir tarım uygulamaları",
                        "📚 Sürekli eğitim ve gelişim",
                        "🤝 Tarımsal danışmanlık hizmetleri"
                    ],
                    'organik': [
                        "Organik tarım uygulamaları:",
                        "🚫 Kimyasal gübre ve ilaç kullanmayın",
                        "🌿 Organik gübreler tercih edin",
                        "🐛 Biyolojik mücadele yöntemleri",
                        "🔄 Ekim nöbeti sistemi",
                        "📜 Organik sertifikasyon süreçleri"
                    ]
                }
            }
        }
        
        console.print("🧑‍🌾 Tarımsal Uzman AI başlatılıyor...", style="bold green")
        self.load_models()
    
    def load_models(self):
        """Eğitilmiş modelleri yükle"""
        console.print("📂 AI modelleri yükleniyor...", style="bold blue")
        
        try:
            # Label mapping yükle
            with open(self.bert_small_path / 'label_mapping.json', 'r') as f:
                self.label_mapping = json.load(f)
            
            # BERT-small model yükle
            console.print("🤖 Hızlı analiz modeli yükleniyor...")
            
            with open(self.bert_small_path / 'config.json', 'r') as f:
                bert_config_dict = json.load(f)
            bert_config = BertConfig.from_dict(bert_config_dict)
            
            self.bert_small_model = BertForSequenceClassification(bert_config)
            state_dict = torch.load(self.bert_small_path / 'pytorch_model.bin', 
                                  map_location=self.device, weights_only=False)
            self.bert_small_model.load_state_dict(state_dict)
            self.bert_small_model.to(self.device)
            self.bert_small_model.eval()
            
            self.bert_small_tokenizer = BertTokenizer.from_pretrained(str(self.bert_small_path))
            
            # DistilBERT model yükle
            console.print("🚀 Uzman analiz modeli yükleniyor...")
            
            with open(self.distilbert_path / 'config.json', 'r') as f:
                distil_config_dict = json.load(f)
            distil_config = DistilBertConfig.from_dict(distil_config_dict)
            
            self.distilbert_model = DistilBertForSequenceClassification(distil_config)
            distil_state_dict = torch.load(self.distilbert_path / 'pytorch_model.bin', 
                                         map_location=self.device, weights_only=False)
            self.distilbert_model.load_state_dict(distil_state_dict)
            self.distilbert_model.to(self.device)
            self.distilbert_model.eval()
            
            self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained(str(self.distilbert_path))
            
            console.print("✅ AI Uzman Sistemi hazır!", style="bold green")
            
        except Exception as e:
            console.print(f"❌ Model yükleme hatası: {e}", style="bold red")
            raise
    
    def analyze_question(self, text: str) -> Dict:
        """Soruyu analiz et ve en iyi kategoriyi bul"""
        try:
            # DistilBERT ile analiz (daha güvenilir)
            inputs = self.distilbert_tokenizer(
                text, return_tensors="pt", truncation=True, 
                padding=True, max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.distilbert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                predicted_id = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_id].item()
            
            predicted_category = self.label_mapping['id_to_label'][str(predicted_id)]
            
            return {
                'category': predicted_category,
                'confidence': confidence,
                'all_probs': probs[0].cpu().numpy()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_expert_response(self, text: str, category: str) -> str:
        """Kategoriye göre uzman cevabı üret"""
        if category not in self.expert_knowledge:
            return "Üzgünüm, bu konuda yeterli bilgim yok. Lütfen daha spesifik bir soru sorun."
        
        expert = self.expert_knowledge[category]
        
        # Soruya göre uygun cevap tipini belirle
        text_lower = text.lower()
        
        if 'drone' in text_lower and category == 'technology':
            responses = expert['responses'].get('drone', expert['responses']['genel'])
        elif any(word in text_lower for word in ['sarı', 'yellow', 'yaprak']) and category == 'plant_disease':
            responses = expert['responses'].get('yapraksarı', expert['responses']['genel'])
        elif any(word in text_lower for word in ['ekim', 'tohum', 'plant']) and category == 'crop_management':
            responses = expert['responses'].get('ekim', expert['responses']['genel'])
        elif any(word in text_lower for word in ['sulama', 'water', 'su']) and category == 'crop_management':
            responses = expert['responses'].get('sulama', expert['responses']['genel'])
        elif any(word in text_lower for word in ['mantar', 'fungal', 'küf']) and category == 'plant_disease':
            responses = expert['responses'].get('fungal', expert['responses']['genel'])
        elif any(word in text_lower for word in ['hibrit', 'hybrid', 'ıslah']) and category == 'plant_genetics':
            responses = expert['responses'].get('hibrit', expert['responses']['genel'])
        elif any(word in text_lower for word in ['toprak', 'soil']) and category == 'environmental_factors':
            responses = expert['responses'].get('toprak', expert['responses']['genel'])
        elif any(word in text_lower for word in ['depo', 'storage', 'sakla']) and category == 'food_security':
            responses = expert['responses'].get('depolama', expert['responses']['genel'])
        elif any(word in text_lower for word in ['organik', 'organic']) and category == 'general_agriculture':
            responses = expert['responses'].get('organik', expert['responses']['genel'])
        else:
            responses = expert['responses']['genel']
        
        return '\n'.join(responses)
    
    def chat_loop(self):
        """Ana sohbet döngüsü"""
        console.print(Panel.fit(
            "🧑‍🌾 Tarımsal Uzman AI'ya Hoş Geldiniz! 🌾\n\n"
            "Ben deneyimli bir tarım uzmanıyım. Size şu konularda yardımcı olabilirim:\n"
            "🌱 Mahsul yönetimi ve yetiştiricilik\n"
            "🔍 Bitki hastalıkları ve zararlılar\n"
            "🧬 Bitki genetiği ve ıslahı\n"
            "🌍 Çevresel faktörler ve toprak\n"
            "🍽️ Gıda güvenliği ve depolama\n"
            "💻 Tarım teknolojileri\n"
            "🚜 Genel tarım uygulamaları\n\n"
            "Sorularınızı doğal dilde sorabilirsiniz. 'çıkış' yazarak ayrılabilirsiniz.",
            style="bold green"
        ))
        
        while True:
            try:
                # Kullanıcı girişi al
                user_input = Prompt.ask(
                    "\n🌱 Tarımsal sorunuz nedir",
                    default="",
                ).strip()
                
                if not user_input:
                    continue
                
                # Özel komutlar
                if user_input.lower() in ['çıkış', 'exit', 'quit', 'q', 'bye']:
                    console.print("👋 İyi tarımlar! Başarılı hasatlar dilerim! 🌾", style="bold green")
                    break
                
                if user_input.lower() in ['help', 'yardım']:
                    self.show_help()
                    continue
                
                # Soruyu analiz et
                console.print("\n🔬 Sorunuzu analiz ediyorum...", style="bold yellow")
                
                analysis = self.analyze_question(user_input)
                
                if 'error' in analysis:
                    console.print("❌ Analiz hatası oluştu. Lütfen sorunuzu tekrar ifade edin.", style="red")
                    continue
                
                category = analysis['category']
                confidence = analysis['confidence']
                
                # Uzman cevabı üret
                expert_info = self.expert_knowledge[category]
                expert_response = self.get_expert_response(user_input, category)
                
                # Uzman kimliği ve selamlama
                console.print(f"\n👨‍🔬 {expert_info['name']}", style="bold cyan")
                console.print(expert_info['greeting'], style="cyan")
                
                # Ana cevap
                console.print(Panel(
                    expert_response,
                    title=f"💡 Uzman Tavsiyesi (Güven: %{confidence*100:.1f})",
                    style="blue",
                    padding=(1, 2)
                ))
                
                # Ek bilgiler
                if confidence > 0.8:
                    console.print("✅ Bu tavsiyeleri güvenle uygulayabilirsiniz.", style="bold green")
                elif confidence > 0.6:
                    console.print("⚠️  Bu konuda daha detaylı bilgi almak isteyebilirsiniz.", style="yellow")
                else:
                    console.print("🤔 Sorunuzu daha net ifade ederseniz daha iyi yardımcı olabilirim.", style="orange3")
                
            except KeyboardInterrupt:
                console.print("\n👋 İyi tarımlar! Başarılı hasatlar dilerim! 🌾", style="bold green")
                break
            except Exception as e:
                console.print(f"❌ Hata: {e}", style="bold red")
    
    def show_help(self):
        """Yardım bilgilerini göster"""
        help_text = """
🆘 **Nasıl Kullanılır:**

✍️  **Normal soru sorun:** 
   "Domates bitkimde sarı yapraklar var, ne yapmalıyım?"
   "Buğday ekimi için toprak nasıl hazırlanır?"

🔍 **Spesifik konular:**
   • Hastalık belirtileri
   • Ekim ve yetiştiricilik
   • Teknoloji kullanımı
   • Toprak sorunları

💡 **İpuçları:**
   • Detaylı sorular daha iyi cevaplar alır
   • Bitki türünü belirtin
   • Semptomları açıklayın
   • Bölgenizi belirtebilirsiniz

🚪 **Çıkış:** 'çıkış' yazın
"""
        console.print(Panel(help_text, title="🆘 Yardım", style="cyan"))

def main():
    """Ana fonksiyon"""
    try:
        chatbot = AgriculturalExpertChatbot()
        chatbot.chat_loop()
    except Exception as e:
        console.print(f"❌ Program başlatılamadı: {e}", style="bold red")

if __name__ == "__main__":
    main() 