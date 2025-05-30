#!/usr/bin/env python3
"""
Gerçek Konuşkan Tarımsal Chatbot
Bağlamsal ve akıcı sohbet yapabilen gerçek AI tarım uzmanı
"""

import os
import json
import torch
import numpy as np
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
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

class ConversationalAgriculturalBot:
    """Gerçek Konuşkan Tarımsal AI Uzmanı"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model paths
        self.bert_small_path = Path("bert_small_agricultural")
        self.distilbert_path = Path("distilbert_agricultural")
        
        # AI Models
        self.distilbert_model = None
        self.distilbert_tokenizer = None
        self.label_mapping = None
        
        # Konuşma hafızası
        self.conversation_history = []
        self.current_context = None
        self.user_preferences = {}
        self.session_start = datetime.now()
        
        # Bot kişiliği
        self.bot_personality = {
            'name': 'Tarım AI',
            'personality': 'samimi, bilgili, yardımsever',
            'expertise': 'tarım, ziraat, botanik',
            'tone': 'arkadaşça ama profesyonel'
        }
        
        # Gelişmiş bilgi bankası
        self.knowledge_base = self._initialize_knowledge_base()
        
        # Sohbet kalıpları
        self.conversation_patterns = self._initialize_conversation_patterns()
        
        console.print("🤖 Gerçek Tarımsal Sohbet AI yükleniyor...", style="bold green")
        self.load_models()
        
        # Hoş geldin mesajı
        self._greet_user()
    
    def _initialize_knowledge_base(self):
        """Detaylı tarımsal bilgi bankası"""
        return {
            'crop_management': {
                'keywords': ['ekim', 'tohum', 'sulama', 'gübre', 'mahsul', 'hasat', 'yetiştiricilik'],
                'context_responses': {
                    'ekim': {
                        'casual': [
                            "Ekim zamanı gerçekten kritik! Hangi ürünü ekmeyi planlıyorsun?",
                            "Ah, ekim mevsimi yaklaşıyor! Toprak hazırlığını yaptın mı?",
                            "Ekim konusunda yardımcı olabilirim. Hangi bölgedesin, onu öğrenebilir miyim?"
                        ],
                        'detailed': [
                            "Ekim için ideal koşullar şunlar: Toprak sıcaklığı 8-10°C, nem %60-70. Sen hangi koşullarda ekiyorsun?",
                            "Tohum kalitesi çok önemli. Sertifikalı tohum kullanıyor musun? Hangi çeşidi tercih ediyorsun?",
                            "Ekim derinliği tohum boyutunun 2-3 katı olmalı. Bu konuda sorun yaşıyor musun?"
                        ],
                        'followup': [
                            "Bu bilgiler yardımcı oldu mu? Başka hangi konuda merak ettiğin var?",
                            "Ekim sonrası bakım hakkında da konuşalım mı?",
                            "Toprak analizi yaptırdın mı? Bu çok önemli."
                        ]
                    },
                    'sulama': {
                        'casual': [
                            "Sulama sistemi kurmuş musun? Damla sulama harika bir seçenek!",
                            "Su çok değerli. Akıllı sulama yapıyor musun?",
                            "Hangi saatlerde sulama yapıyorsun? Sabah erken saatler ideal."
                        ],
                        'detailed': [
                            "Toprak nem seviyesini nasıl kontrol ediyorsun? Nem sensörü kullanman çok faydalı olur.",
                            "Bitki türüne göre su ihtiyacı değişir. Hangi bitkilerle çalışıyorsun?",
                            "Damla sulama %30-50 su tasarrufu sağlar. Maliyet analizi yapmış mıydın?"
                        ]
                    }
                }
            },
            'plant_disease': {
                'keywords': ['hastalık', 'zararlı', 'yaprak', 'sarı', 'leke', 'mantar', 'böcek'],
                'context_responses': {
                    'hastalık': {
                        'urgent': [
                            "Hastalık belirtileri acil müdahale gerektirir! Ne tür belirtiler görüyorsun?",
                            "Hemen fotoğraf çekip inceleyebilir misin? Erken teşhis çok önemli!",
                            "Hangi bitkide sorun var? Belirtileri detayca anlat bakalım."
                        ],
                        'diagnostic': [
                            "Yapraklarda leke var mı? Rengi nasıl - sarı, kahverengi, siyah?",
                            "Böyle belirtiler genelde nem fazlalığından olur. Hava sirkülasyonu nasıl?",
                            "Bu mantar hastalığı olabilir. Organik fungisit denedin mi?"
                        ],
                        'solution': [
                            "Hasta yaprakları hemen topla ve imha et. Bulaşmayı önlemek için!",
                            "Neem yağı çok etkili organik bir çözüm. Denedin mi hiç?",
                            "Önleyici olarak bakır sülfat kullanabilirsin. Ama dikkatli ol, dozajı önemli."
                        ]
                    }
                }
            },
            'technology': {
                'keywords': ['drone', 'sensör', 'teknoloji', 'akıllı', 'otomasyon', 'gps'],
                'context_responses': {
                    'teknoloji': {
                        'interested': [
                            "Teknoloji meraklısısın galiba! Ben de çok seviyorum bu konuları.",
                            "Hangi tarım teknolojileri ilgini çekiyor? Drone, sensör, AI?",
                            "Bütçen ne kadar? Bazı teknolojiler artık çok uygun fiyatlı."
                        ],
                        'practical': [
                            "Drone kullanmayı düşünüyor musun? Hangi amaçla - analiz, ilaçlama?",
                            "IoT sensörleri şimdi çok ucuzladı. Toprak nemi, sıcaklık ölçebiliyorlar.",
                            "GPS'li traktör kullanıyor musun? Hassas tarım için çok önemli."
                        ]
                    }
                }
            }
        }
    
    def _initialize_conversation_patterns(self):
        """Sohbet kalıpları ve akışları"""
        return {
            'greetings': [
                "Merhaba! Ben tarım uzmanı AI'ınım. Nasılsın bugün?",
                "Selam! Tarımsal konularda sana yardımcı olmak için buradayım.",
                "Hey! Bugün hangi tarımsal konularda konuşacağız?"
            ],
            'acknowledgments': [
                "Anlıyorum", "Tabii", "Evet, haklısın", "Doğru söylüyorsun", 
                "İyi bir gözlem", "Mantıklı"
            ],
            'clarifications': [
                "Daha detaylı anlatabilir misin?",
                "Hangi kısmını merak ediyorsun?",
                "Bu konuda spesifik bir sorun mu var?",
                "Daha açık ifade eder misin?"
            ],
            'transitions': [
                "Bu arada,", "Bir de şunu sorabilir miyim?", "Peki,", 
                "Aklıma şu geldi:", "Bu konudan bahsetmişken,"
            ],
            'empathy': [
                "Anlıyorum, zor bir durum",
                "Bu gerçekten can sıkıcı olabilir",
                "Böyle problemlerle karşılaşmak normal",
                "Endişelenme, çözümü var"
            ]
        }
    
    def load_models(self):
        """AI modellerini yükle"""
        try:
            # Label mapping
            with open(self.bert_small_path / 'label_mapping.json', 'r') as f:
                self.label_mapping = json.load(f)
            
            console.print("🧠 DistilBERT analiz modeli yükleniyor...")
            
            # DistilBERT model
            with open(self.distilbert_path / 'config.json', 'r') as f:
                config_dict = json.load(f)
            config = DistilBertConfig.from_dict(config_dict)
            
            self.distilbert_model = DistilBertForSequenceClassification(config)
            state_dict = torch.load(self.distilbert_path / 'pytorch_model.bin', 
                                  map_location=self.device, weights_only=False)
            self.distilbert_model.load_state_dict(state_dict)
            self.distilbert_model.to(self.device)
            self.distilbert_model.eval()
            
            self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained(str(self.distilbert_path))
            
            console.print("✅ AI modeli hazır!", style="bold green")
            
        except Exception as e:
            console.print(f"❌ Model yükleme hatası: {e}", style="bold red")
            raise
    
    def _greet_user(self):
        """Kullanıcıyı karşıla"""
        greeting = random.choice(self.conversation_patterns['greetings'])
        
        welcome_panel = Panel.fit(
            f"{greeting}\n\n"
            "🌾 Ben senin kişisel tarım uzmanınım! \n\n"
            "💬 Benimle normal konuşma gibi sohbet edebilirsin\n"
            "🧠 Geçmiş konuşmalarımızı hatırlıyorum\n"
            "🎯 Spesifik sorularına detaylı cevaplar veriyorum\n"
            "🤝 Samimi ve arkadaşça bir üslubum var\n\n"
            "Hangi konularda sohbet etmek istersin? 🌱",
            title="🤖 Tarım AI Uzmanın",
            style="bold green"
        )
        console.print(welcome_panel)
    
    def analyze_intent(self, text: str) -> Dict:
        """Kullanıcının niyetini ve bağlamını analiz et"""
        try:
            # Model ile kategori analizi
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
            
            # Soru tipini belirle
            question_type = self._detect_question_type(text)
            
            # Aciliyet seviyesi
            urgency = self._detect_urgency(text)
            
            # Duygusal ton
            emotional_tone = self._detect_emotional_tone(text)
            
            return {
                'category': predicted_category,
                'confidence': confidence,
                'question_type': question_type,
                'urgency': urgency,
                'emotional_tone': emotional_tone,
                'keywords': self._extract_keywords(text, predicted_category)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_question_type(self, text: str) -> str:
        """Soru tipini belirle"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['nasıl', 'how', 'ne şekilde']):
            return 'how_to'
        elif any(word in text_lower for word in ['neden', 'why', 'niçin']):
            return 'why'
        elif any(word in text_lower for word in ['ne zaman', 'when', 'hangi zaman']):
            return 'when'
        elif any(word in text_lower for word in ['nerede', 'where', 'hangi yer']):
            return 'where'
        elif any(word in text_lower for word in ['ne', 'what', 'hangi']):
            return 'what'
        elif '?' in text:
            return 'general_question'
        else:
            return 'statement'
    
    def _detect_urgency(self, text: str) -> str:
        """Aciliyet seviyesini belirle"""
        text_lower = text.lower()
        
        urgent_words = ['acil', 'urgent', 'hemen', 'çabuk', 'ölüyor', 'dying', 'kurtarın']
        if any(word in text_lower for word in urgent_words):
            return 'urgent'
        
        concern_words = ['endişe', 'worry', 'sorun', 'problem', 'korku']
        if any(word in text_lower for word in concern_words):
            return 'concerned'
        
        return 'normal'
    
    def _detect_emotional_tone(self, text: str) -> str:
        """Duygusal tonu algıla"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['teşekkür', 'thank', 'sağol', 'memnun']):
            return 'grateful'
        elif any(word in text_lower for word in ['üzgün', 'sad', 'kötü', 'berbat']):
            return 'sad'
        elif any(word in text_lower for word in ['mutlu', 'happy', 'harika', 'süper']):
            return 'happy'
        elif any(word in text_lower for word in ['kızgın', 'angry', 'sinir', 'öfke']):
            return 'angry'
        else:
            return 'neutral'
    
    def _extract_keywords(self, text: str, category: str) -> List[str]:
        """Metinden anahtar kelimeleri çıkar"""
        text_lower = text.lower()
        keywords = []
        
        if category in self.knowledge_base:
            category_keywords = self.knowledge_base[category]['keywords']
            for keyword in category_keywords:
                if keyword in text_lower:
                    keywords.append(keyword)
        
        return keywords
    
    def generate_response(self, user_input: str) -> str:
        """Bağlamsal ve doğal cevap üret"""
        # Analiz yap
        analysis = self.analyze_intent(user_input)
        
        if 'error' in analysis:
            return self._handle_error()
        
        # Konuşma geçmişine ekle
        self.conversation_history.append({
            'user': user_input,
            'timestamp': datetime.now(),
            'analysis': analysis
        })
        
        # Bağlamsal cevap üret
        response = self._create_contextual_response(user_input, analysis)
        
        # Bot cevabını da geçmişe ekle
        self.conversation_history[-1]['bot_response'] = response
        
        return response
    
    def _create_contextual_response(self, user_input: str, analysis: Dict) -> str:
        """Bağlamsal cevap oluştur"""
        category = analysis['category']
        confidence = analysis['confidence']
        question_type = analysis['question_type']
        urgency = analysis['urgency']
        emotional_tone = analysis['emotional_tone']
        keywords = analysis['keywords']
        
        # Cevap bileşenlerini hazırla
        response_parts = []
        
        # 1. Duygusal yanıt
        emotional_response = self._get_emotional_response(emotional_tone, urgency)
        if emotional_response:
            response_parts.append(emotional_response)
        
        # 2. Ana cevap
        main_response = self._get_main_response(category, keywords, question_type, user_input)
        response_parts.append(main_response)
        
        # 3. Takip sorusu/önerisi
        if confidence > 0.7:
            followup = self._get_followup_question(category, keywords, user_input)
            if followup:
                response_parts.append(followup)
        
        # 4. Kişisel dokunuş
        personal_touch = self._add_personal_touch(user_input, analysis)
        if personal_touch:
            response_parts.append(personal_touch)
        
        return '\n\n'.join(response_parts)
    
    def _get_emotional_response(self, tone: str, urgency: str) -> str:
        """Duygusal yanıt ver"""
        if urgency == 'urgent':
            return random.choice([
                "Anlıyorum, acil bir durum! Hemen yardımcı olalım.",
                "Tamam, bu gerçekten acil görünüyor. Adım adım çözelim.",
                "Endişelenme, böyle durumlarda ne yapmak gerektiğini biliyorum."
            ])
        
        if tone == 'grateful':
            return random.choice([
                "Rica ederim! Yardımcı olabildiğime sevindim.",
                "Ne demek! Bu tür konuları konuşmayı seviyorum.",
                "Teşekkürler güzel! Daha ne konuşmak istersin?"
            ])
        
        if tone == 'sad':
            return random.choice(self.conversation_patterns['empathy'])
        
        return None
    
    def _get_main_response(self, category: str, keywords: List[str], question_type: str, user_input: str) -> str:
        """Ana cevabı üret"""
        # Kategori bazlı özel cevaplar
        if category == 'plant_disease' and any(k in keywords for k in ['sarı', 'yaprak']):
            return self._generate_plant_disease_response(user_input, keywords)
        elif category == 'crop_management' and 'ekim' in keywords:
            return self._generate_planting_response(user_input, question_type)
        elif category == 'technology' and 'drone' in keywords:
            return self._generate_technology_response(user_input)
        
        # Genel kategorik cevap
        return self._generate_general_response(category, question_type, user_input)
    
    def _generate_plant_disease_response(self, user_input: str, keywords: List[str]) -> str:
        """Bitki hastalığı özel cevabı"""
        responses = [
            f"Sarı yapraklar birkaç nedenden olabilir. Önce şunu kontrol edelim:",
            f"1. 💧 Sulama düzeni nasıl? Aşırı su veya susuzluk sararmaya neden olur",
            f"2. 🌱 Azot eksikliği olabilir - yaprak gübresi denedin mi?",
            f"3. 🔍 Kök bölgesinde çürüklük var mı kontrol et",
            f"",
            f"Hangi bitkide bu problem var? Ve ne kadar süredir böyle?"
        ]
        return '\n'.join(responses)
    
    def _generate_planting_response(self, user_input: str, question_type: str) -> str:
        """Ekim özel cevabı"""
        if question_type == 'when':
            return (
                "Ekim zamanı çok önemli! Hangi ürünü ekmeyi planlıyorsun?\n\n"
                "Genel olarak:\n"
                "🌡️ Toprak sıcaklığı en az 8-10°C olmalı\n"
                "📅 Son don tarihinden 2-3 hafta sonra güvenli\n"
                "💧 Toprak nemi %60-70 arasında ideal\n\n"
                "Hangi bölgedesin? Ona göre daha spesifik tavsiye verebilirim."
            )
        elif question_type == 'how_to':
            return (
                "Ekim işlemi şöyle yapılır:\n\n"
                "1. 🌱 Toprak hazırlığı - kök ve taş temizliği\n"
                "2. ⚗️ Toprak analizi ve gerekirse düzeltme\n"
                "3. 📏 Doğru ekim derinliği (tohum boyutunun 2-3 katı)\n"
                "4. 💧 Ekim sonrası hafif sulama\n\n"
                "Hangi aşamada takıldın? Detayına inelim!"
            )
        
        return "Ekim konusunda ne öğrenmek istiyorsun? Soru tipini anlayamadım, biraz daha açık sorabilir misin?"
    
    def _generate_technology_response(self, user_input: str) -> str:
        """Teknoloji özel cevabı"""
        return (
            "Drone teknolojisi tarımda devrim yaratıyor! 🚁\n\n"
            "Kullanım alanları:\n"
            "📸 Alan haritalama ve analiz\n"
            "🌱 Bitki sağlığı takibi (NDVI analizi)\n"
            "💊 Hassas ilaçlama ve gübreleme\n"
            "📊 Verim tahminleri\n\n"
            "Hangi amaçla kullanmayı düşünüyorsun? Bütçen ne kadar?"
        )
    
    def _generate_general_response(self, category: str, question_type: str, user_input: str) -> str:
        """Genel kategorik cevap"""
        category_responses = {
            'crop_management': "Mahsul yönetimi konusunda yardımcı olabilirim! Spesifik olarak neyi merak ediyorsun?",
            'environmental_factors': "Çevre faktörleri çok önemli! Toprak, iklim, hangi konuda konuşalım?",
            'food_security': "Gıda güvenliği kritik bir konu. Hangi açıdan yaklaşmak istersin?",
            'general_agriculture': "Genel tarım konularında deneyimliyim. Ne öğrenmek istiyorsun?",
            'plant_genetics': "Bitki genetiği fascinant bir alan! Hangi yönüyle ilgileniyorsun?",
            'technology': "Tarım teknolojilerini seviyorum! Hangi teknolojiler ilgini çekiyor?"
        }
        
        return category_responses.get(category, "Bu konuda yardımcı olabilirim. Biraz daha detaya inelim mi?")
    
    def _get_followup_question(self, category: str, keywords: List[str], user_input: str) -> str:
        """Takip sorusu üret"""
        followups = [
            "Bu konuda başka merak ettiğin var mı?",
            "Daha detayına inmek ister misin?",
            "Pratik uygulamalar konusunda konuşalım mı?",
            "Kendi deneyimlerini de paylaşır mısın?"
        ]
        
        # Konuya özel takip soruları
        if category == 'crop_management':
            return "Bu bilgiler işine yaradı mı? Pratik uygulama sırasında takıldığın bir yer olursa söyle!"
        elif category == 'plant_disease':
            return "Fotoğraf paylaşabiliyor musun? Görselle daha net teşhis yapabilirim."
        elif category == 'technology':
            return "Hangi teknolojileri merak ediyorsun? Modern çiftçilikte çok seçenek var!"
        
        return random.choice(followups)
    
    def _add_personal_touch(self, user_input: str, analysis: Dict) -> str:
        """Kişisel dokunuş ekle"""
        # Konuşma sayısına göre
        conversation_count = len(self.conversation_history)
        
        if conversation_count == 1:
            return "Bu arada, ben burada senin için varım. İstediğin zaman soru sorabilirsin! 😊"
        elif conversation_count == 5:
            return "Güzel sohbet ediyoruz! Tarım konularını konuşmayı gerçekten seviyorum."
        elif conversation_count == 10:
            return "Vay be! Epey sohbet ettik bugün. Başka konular da var mı aklında?"
        
        return None
    
    def _handle_error(self) -> str:
        """Hata durumunu yakala"""
        return (
            "Hmm, sorununu tam anlayamadım. 🤔\n\n"
            "Biraz daha açık ifade eder misin? Örneğin:\n"
            "• Hangi bitki/ürün hakkında?\n"
            "• Ne tür bir sorun yaşıyorsun?\n"
            "• Hangi konuda yardım istiyorsun?\n\n"
            "Böylece daha iyi yardımcı olabilirim!"
        )
    
    def chat_loop(self):
        """Ana sohbet döngüsü"""
        while True:
            try:
                # Kullanıcı girişi
                user_input = Prompt.ask(
                    f"\n💬 [bold green]Sen[/bold green]",
                    default=""
                ).strip()
                
                if not user_input:
                    continue
                
                # Çıkış komutları
                if user_input.lower() in ['çıkış', 'exit', 'quit', 'bye', 'görüşürüz']:
                    self._farewell()
                    break
                
                # Yardım
                if user_input.lower() in ['help', 'yardım']:
                    self._show_help()
                    continue
                
                # Sohbet geçmişi
                if user_input.lower() in ['geçmiş', 'history']:
                    self._show_conversation_history()
                    continue
                
                # Ana cevap üretimi
                console.print("\n🤔 Düşünüyorum...", style="italic yellow")
                
                response = self.generate_response(user_input)
                
                # Bot cevabını göster
                console.print(f"\n🤖 [bold cyan]Tarım AI[/bold cyan]: {response}")
                
            except KeyboardInterrupt:
                self._farewell()
                break
            except Exception as e:
                console.print(f"\n❌ Bir hata oluştu: {e}", style="bold red")
                console.print("Tekrar dener misin? 😊", style="yellow")
    
    def _farewell(self):
        """Vedalaşma"""
        session_duration = datetime.now() - self.session_start
        conversation_count = len(self.conversation_history)
        
        farewell_msg = (
            f"👋 Hoşçakal! {session_duration.seconds//60} dakika boyunca "
            f"{conversation_count} mesaj ile güzel sohbet ettik.\n\n"
            "🌾 İyi tarımlar! Başarılı hasatlar dilerim!\n"
            "🤝 İhtiyacın olduğunda tekrar gel, hep buradayım!"
        )
        
        console.print(Panel(farewell_msg, title="👋 Görüşmek Üzere!", style="bold green"))
    
    def _show_help(self):
        """Yardım göster"""
        help_text = (
            "🆘 **Nasıl Kullanabilirim?**\n\n"
            "💬 Normal konuşma gibi sorularını sor:\n"
            "   'Domates bitkimde sarı yapraklar var'\n"
            "   'Buğday ne zaman ekilir?'\n"
            "   'Drone teknolojisi hakkında bilgi ver'\n\n"
            "🎯 **Özel Komutlar:**\n"
            "   • 'geçmiş' - Sohbet geçmişini göster\n"
            "   • 'yardım' - Bu mesajı göster\n"
            "   • 'çıkış' - Sohbeti bitir\n\n"
            "💡 **İpuçları:**\n"
            "   • Detaylı sorular daha iyi cevaplar alır\n"
            "   • Bağlamsal konuşma yapabilirim\n"
            "   • Geçmiş konuşmaları hatırlıyorum"
        )
        
        console.print(Panel(help_text, title="🆘 Yardım", style="cyan"))
    
    def _show_conversation_history(self):
        """Sohbet geçmişini göster"""
        if not self.conversation_history:
            console.print("Henüz hiç konuşmadık! 😊", style="yellow")
            return
        
        console.print(f"\n📜 Sohbet Geçmişi ({len(self.conversation_history)} mesaj):", style="bold blue")
        
        for i, conv in enumerate(self.conversation_history[-5:], 1):  # Son 5 mesaj
            time_str = conv['timestamp'].strftime("%H:%M")
            console.print(f"\n{i}. [{time_str}] Sen: {conv['user'][:100]}...")
            if 'bot_response' in conv:
                console.print(f"   🤖: {conv['bot_response'][:100]}...")

def main():
    """Ana fonksiyon"""
    try:
        bot = ConversationalAgriculturalBot()
        bot.chat_loop()
    except Exception as e:
        console.print(f"❌ Program başlatılamadı: {e}", style="bold red")

if __name__ == "__main__":
    main() 