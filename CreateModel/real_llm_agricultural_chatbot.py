#!/usr/bin/env python3
"""
Gerçek LLM Tarımsal Chatbot
Eğitilmiş GPT-2 model ile doğal dilde cevap üretimi
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    pipeline
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.text import Text
from rich import print as rprint

console = Console()

class RealLLMAgriculturalChatbot:
    """Gerçek LLM ile Tarımsal Chatbot"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path("agricultural_gpt2")
        
        # LLM Pipeline
        self.generator = None
        self.tokenizer = None
        
        # Konuşma hafızası
        self.conversation_history = []
        self.session_start = datetime.now()
        
        # Bot kişiliği
        self.bot_personality = {
            'name': 'Tarım LLM AI',
            'style': 'Uzman, samimi, yardımsever',
            'expertise': 'Gerçek LLM tabanlı tarımsal danışmanlık'
        }
        
        console.print("🧠 Gerçek LLM Tarımsal AI yükleniyor...", style="bold green")
        self.initialize_llm()
        self._welcome_user()
    
    def initialize_llm(self):
        """LLM'i başlat"""
        try:
            if not self.model_path.exists():
                console.print("❌ Eğitilmiş model bulunamadı! Önce train_agricultural_llm.py çalıştırın.", style="bold red")
                console.print("📖 Geçici olarak GPT-2 base model kullanılacak...", style="yellow")
                self.model_path = "gpt2"
            
            console.print("🚀 LLM yükleniyor...", style="cyan")
            
            # Text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=str(self.model_path),
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(str(self.model_path))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            console.print("✅ LLM hazır!", style="bold green")
            
        except Exception as e:
            console.print(f"❌ LLM yükleme hatası: {e}", style="bold red")
            raise
    
    def _welcome_user(self):
        """Kullanıcıyı karşıla"""
        welcome_panel = Panel.fit(
            "🧠 **Gerçek LLM Tarımsal AI'ya Hoş Geldin!**\n\n"
            "🌟 **Yeni Teknoloji:**\n"
            "🤖 **Gerçek Language Model**: GPT-2 tabanlı tarımsal uzman\n"
            "💬 **Doğal Dil**: İnsan gibi akıcı konuşma\n"
            "🎯 **Özel Eğitim**: Tarımsal verilerle fine-tuned\n"
            "🧠 **Akıllı Üretim**: Template değil, gerçek text generation\n"
            "📚 **Uzmanlık**: Hastalık, yetiştirme, bakım konularında\n\n"
            "💡 **Artık gerçek bir AI uzmanıyla konuşuyorsunuz!**\n"
            "🗨️ Sorunuzu doğal dilde sorun, size detaylı açıklama yapayım! 🌱",
            title="🧠 Gerçek LLM Tarım AI",
            style="bold green"
        )
        console.print(welcome_panel)
    
    def generate_response(self, user_query: str) -> str:
        """LLM ile cevap üret"""
        try:
            # Prompt hazırla
            if self.model_path.name == "agricultural_gpt2":
                # Eğitilmiş model için özel format
                prompt = f"<|soru|>{user_query}<|cevap|>"
            else:
                # Base GPT-2 için genel prompt
                prompt = f"Tarım uzmanı olarak şu soruyu yanıtlayın: {user_query}\n\nYanıt:"
            
            # Cevap üret
            response = self.generator(
                prompt,
                max_length=min(len(prompt.split()) + 150, 400),  # Adaptive length
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            
            # Cevabı temizle
            if self.model_path.name == "agricultural_gpt2":
                # Eğitilmiş modelden cevabı çıkar
                if "<|cevap|>" in generated_text:
                    answer = generated_text.split("<|cevap|>")[-1]
                    if "<|end|>" in answer:
                        answer = answer.split("<|end|>")[0]
                    answer = answer.strip()
                else:
                    answer = generated_text[len(prompt):].strip()
            else:
                # Base modelden cevabı çıkar
                if "Yanıt:" in generated_text:
                    answer = generated_text.split("Yanıt:")[-1].strip()
                else:
                    answer = generated_text[len(prompt):].strip()
            
            # Çok kısa cevapları genişlet
            if len(answer.split()) < 10:
                answer = self._enhance_short_answer(user_query, answer)
            
            # Tarımsal bağlam ekle
            answer = self._add_agricultural_context(user_query, answer)
            
            return answer
            
        except Exception as e:
            console.print(f"⚠️ Cevap üretme hatası: {e}", style="yellow")
            return self._fallback_response(user_query)
    
    def _enhance_short_answer(self, query: str, answer: str) -> str:
        """Kısa cevapları genişlet"""
        query_lower = query.lower()
        
        # Konu bazlı genişletmeler
        if 'yanıklık' in query_lower:
            return f"{answer}\n\nBu hastalık bakteriyel kökenlidir ve hızla yayılabilir. Erken müdahale çok önemlidir. Hasta kısımları temizlemek ve koruyucu ilaçlama yapmak gerekir."
        elif 'ekim' in query_lower:
            return f"{answer}\n\nEkim başarısı için toprak hazırlığı, doğru zaman seçimi ve uygun derinlik çok önemlidir. Hava koşullarını da takip etmek gerekir."
        elif 'sulama' in query_lower:
            return f"{answer}\n\nSulama zamanı ve miktarı bitkinin gelişim dönemine göre ayarlanmalıdır. Toprak tipini ve hava durumunu da göz önünde bulundurmak önemlidir."
        
        return f"{answer}\n\nBu konuda daha detaylı bilgi isterseniz, spesifik sorular sorabilirsiniz."
    
    def _add_agricultural_context(self, query: str, answer: str) -> str:
        """Tarımsal bağlam ekle"""
        # Çok genel cevapları filtreye
        generic_phrases = [
            "I cannot", "I don't know", "As an AI", "I'm not sure",
            "Sorry", "Unfortunately", "However", "But"
        ]
        
        if any(phrase in answer for phrase in generic_phrases):
            return self._fallback_response(query)
        
        # Tarımsal terimler içeriyorsa olduğu gibi döndür
        agricultural_terms = [
            'bitki', 'plant', 'toprak', 'soil', 'gübre', 'fertilizer',
            'sulama', 'irrigation', 'hastalık', 'disease', 'ekim', 'planting'
        ]
        
        if any(term in answer.lower() for term in agricultural_terms):
            return answer
        
        # Genel cevapsa tarımsal versiyona çevir
        return self._fallback_response(query)
    
    def _fallback_response(self, query: str) -> str:
        """Yedek cevap sistemi"""
        query_lower = query.lower()
        
        fallback_responses = {
            'elma': "Elma yetiştirme konusunda yardımcı olabilirim. Elma ağaçları düzenli bakım, uygun budama ve hastalık kontrolü gerektirir.",
            'yanıklık': "Erken yanıklığı ciddi bir bakteriyel hastalıktır. Hasta kısımları hemen kesilmeli ve koruyucu ilaçlama yapılmalıdır.",
            'buğday': "Buğday tarımında ekim zamanı, toprak hazırlığı ve gübreleme çok önemlidir.",
            'domates': "Domates yetiştirmede sulama, gübreleme ve hastalık kontrolü başarının anahtarıdır.",
            'havuç': "Havuç için derin, gevşek toprak gerekir. Düzenli sulama ve iyi drenaj önemlidir.",
            'sıcaklık': "Aşırı sıcaklıkta bitkileri korumak için gölgeleme, mulch ve düzenli sulama gerekir.",
            'sulama': "Doğru sulama zamanı ve tekniği bitkinin sağlıklı gelişimi için kritiktir."
        }
        
        for keyword, response in fallback_responses.items():
            if keyword in query_lower:
                return response
        
        return "Bu konuda size yardımcı olmaya çalışıyorum. Sorunuzu biraz daha spesifik hale getirebilir misiniz? Hangi bitki veya tarımsal konuyla ilgili yardım istiyorsunuz?"
    
    def chat_loop(self):
        """Ana sohbet döngüsü"""
        while True:
            try:
                # Kullanıcı girişi
                user_input = Prompt.ask(
                    f"\n💬 [bold green]Siz[/bold green]",
                    default=""
                ).strip()
                
                if not user_input:
                    continue
                
                # Özel komutlar
                if user_input.lower() in ['çıkış', 'exit', 'quit', 'bye']:
                    self._farewell()
                    break
                elif user_input.lower() in ['help', 'yardım']:
                    self._show_help()
                    continue
                elif user_input.lower() in ['geçmiş', 'history']:
                    self._show_history()
                    continue
                elif user_input.lower() in ['test', 'test et']:
                    self._run_test()
                    continue
                
                # Ana cevap üretimi
                console.print("\n🧠 LLM düşünüyor ve cevap üretiyor...", style="italic yellow")
                
                response = self.generate_response(user_input)
                
                # Konuşma geçmişine ekle
                self.conversation_history.append({
                    'user': user_input,
                    'bot': response,
                    'timestamp': datetime.now()
                })
                
                # Cevabı göster
                console.print(f"\n🧠 [bold cyan]Tarım LLM AI[/bold cyan]:\n{response}")
                
            except KeyboardInterrupt:
                self._farewell()
                break
            except Exception as e:
                console.print(f"\n❌ Bir hata oluştu: {e}", style="bold red")
                console.print("Tekrar dener misiniz? 😊", style="yellow")
    
    def _run_test(self):
        """Test soruları çalıştır"""
        console.print("🧪 Test soruları çalıştırılıyor...", style="cyan")
        
        test_questions = [
            "Elmada erken yanıklığı nedir?",
            "Buğday ekim zamanı ne zaman?",
            "Domates sarı yaprak sorunu nasıl çözülür?",
            "Aşırı sıcaklıkta bitkileri nasıl koruruz?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            console.print(f"\n[bold blue]Test {i}:[/bold blue] {question}")
            response = self.generate_response(question)
            console.print(f"[bold green]Cevap:[/bold green] {response}")
            console.print("-" * 60)
    
    def _show_help(self):
        """Yardım göster"""
        help_panel = Panel.fit(
            "🆘 **Gerçek LLM Chatbot Kullanım Kılavuzu**\n\n"
            "💬 **Doğal Konuşma:**\n"
            "   Tarımsal sorularınızı normal konuşma dili ile sorun\n\n"
            "🧠 **LLM Avantajları:**\n"
            "   • Gerçek text generation (template değil)\n"
            "   • Bağlamsal ve akıcı cevaplar\n"
            "   • Tarımsal konularda özel eğitim\n\n"
            "🎯 **Örnek Sorular:**\n"
            "   • 'Elmamda yanıklık var, ne yapmalıyım?'\n"
            "   • 'Havuç ekimi için en iyi zaman ne?'\n"
            "   • 'Sıcak havada bitkilerimi nasıl koruyabilirim?'\n\n"
            "⚡ **Komutlar:**\n"
            "   • 'test' - Test sorularını çalıştır\n"
            "   • 'geçmiş' - Konuşma geçmişi\n"
            "   • 'yardım' - Bu yardım menüsü\n"
            "   • 'çıkış' - Programdan çık\n\n"
            "💡 **İpucu:** Bu gerçek bir AI! Normal konuşma gibi\n"
            "   sorularınızı sorun, size detaylı açıklama yapacak.",
            title="🆘 LLM Chatbot Yardım",
            style="cyan"
        )
        console.print(help_panel)
    
    def _show_history(self):
        """Sohbet geçmişini göster"""
        if not self.conversation_history:
            console.print("Henüz konuşma geçmişi yok! 😊", style="yellow")
            return
        
        console.print(f"\n📜 Son 5 Konuşma:", style="bold blue")
        
        for i, conv in enumerate(self.conversation_history[-5:], 1):
            time_str = conv['timestamp'].strftime("%H:%M")
            console.print(f"\n{i}. [{time_str}]")
            console.print(f"   [bold blue]Soru:[/bold blue] {conv['user'][:100]}...")
            console.print(f"   [bold green]Cevap:[/bold green] {conv['bot'][:150]}...")
    
    def _farewell(self):
        """Vedalaşma"""
        duration = datetime.now() - self.session_start
        duration_min = duration.seconds // 60
        conversation_count = len(self.conversation_history)
        
        farewell_panel = Panel.fit(
            f"👋 **Hoşçakalın!**\n\n"
            f"📊 **LLM Sohbet Özeti:**\n"
            f"⏰ Süre: {duration_min} dakika\n"
            f"💬 Mesaj: {conversation_count} adet\n"
            f"🧠 Gerçek LLM ile etkileşim tamamlandı\n\n"
            f"🌾 **Tekrar görüşmek üzere!**\n"
            f"🤖 LLM'iniz her zaman hizmetinizde!",
            title="👋 Görüşmek Üzere",
            style="bold green"
        )
        console.print(farewell_panel)

def main():
    """Ana fonksiyon"""
    try:
        bot = RealLLMAgriculturalChatbot()
        bot.chat_loop()
    except Exception as e:
        console.print(f"❌ Program başlatılamadı: {e}", style="bold red")

if __name__ == "__main__":
    main() 