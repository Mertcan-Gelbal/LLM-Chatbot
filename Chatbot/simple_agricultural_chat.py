#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌱 Simple Agricultural Chat - Terminal Based
Lightweight chatbot for quick agricultural questions

Usage:
    python simple_agricultural_chat.py
    python simple_agricultural_chat.py "Your question here"
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from Model.run_model import predict_text

class SimpleAgriculturalChat:
    """Simple terminal-based agricultural chatbot"""
    
    def __init__(self):
        self.name = "🌱 Tarım AI"
        self.categories = {
            "plant_disease": "🦠 Bitki Hastalıkları",
            "crop_management": "🌾 Mahsul Yönetimi",
            "plant_genetics": "🧬 Bitki Genetiği",
            "environmental_factors": "🌡️ Çevre Faktörleri",
            "food_security": "🍽️ Gıda Güvenliği",
            "technology": "🚁 Tarım Teknolojisi"
        }
        
        self.examples = [
            "Domates yaprak yanıklığı nedir?",
            "Buğday ekimi nasıl yapılır?",
            "Organik gübre çeşitleri nelerdir?",
            "Kuraklık stresi nasıl önlenir?",
            "Akıllı sulama sistemleri nasıl çalışır?",
            "GMO bitkilerin avantajları nelerdir?"
        ]
    
    def welcome_message(self):
        """Show welcome message"""
        print("=" * 60)
        print(f"🌱 {self.name} - Tarımsal AI Asistanı")
        print("=" * 60)
        print()
        print("📋 Uzman olduğum konular:")
        for category, name in self.categories.items():
            print(f"   • {name}")
        print()
        print("💬 Örnek sorular:")
        for example in self.examples[:3]:
            print(f"   • {example}")
        print()
        print("ℹ️  Çıkmak için 'çıkış', 'quit' veya 'q' yazın")
        print("ℹ️  Yardım için 'yardım' veya 'help' yazın")
        print("=" * 60)
    
    def help_message(self):
        """Show help message"""
        print("\n📚 Yardım Menüsü:")
        print("=" * 40)
        print("🔧 Komutlar:")
        print("   • 'yardım' / 'help' - Bu mesajı göster")
        print("   • 'örnekler' - Örnek soruları listele")
        print("   • 'kategoriler' - Konu kategorilerini göster")
        print("   • 'çıkış' / 'quit' / 'q' - Programdan çık")
        print()
        print("💡 İpuçları:")
        print("   • Soruları açık ve net şekilde sorun")
        print("   • Bitki adı, hastalık belirtisi gibi detaylar verin")
        print("   • Türkçe veya İngilizce sorabilirsiniz")
        print("=" * 40)
    
    def show_examples(self):
        """Show example questions"""
        print("\n💬 Örnek Sorular:")
        print("=" * 40)
        for i, example in enumerate(self.examples, 1):
            print(f"{i}. {example}")
        print("=" * 40)
    
    def show_categories(self):
        """Show categories"""
        print("\n📋 Konu Kategorileri:")
        print("=" * 40)
        for category, name in self.categories.items():
            print(f"   {name}")
        print("=" * 40)
    
    def process_query(self, query: str) -> str:
        """Process user query and return response"""
        try:
            # Use existing model prediction
            result = predict_text(query)
            
            if 'error' in result:
                return f"❌ Hata: {result['error']}"
            
            # Format response
            category_tr = result.get('category_turkish', 'Bilinmeyen')
            confidence = result.get('confidence', 0.0)
            
            response = f"🎯 Kategori: {category_tr} (%{confidence*100:.1f} güven)\n"
            
            # Add category-specific advice
            advice = self.get_category_advice(result.get('category', ''))
            if advice:
                response += f"\n💡 Öneri: {advice}"
            
            return response
            
        except Exception as e:
            return f"❌ Bir hata oluştu: {str(e)}"
    
    def get_category_advice(self, category: str) -> str:
        """Get category-specific advice"""
        advice_map = {
            "plant_disease": "Hastalık teşhisi için bitki türü, belirtiler ve çevre koşullarını belirtin. Uzman desteği alınması önerilir.",
            "crop_management": "Ekim zamanlaması bölgesel koşullara göre değişir. Yerel tarım uzmanından bilgi alın.",
            "plant_genetics": "Genetik ıslah uzun vadeli projelerdir. Araştırma kurumlarından destek alabilirsiniz.",
            "environmental_factors": "Çevresel stres erken müdahale ile minimize edilebilir. Monitoring sistemleri kurun.",
            "food_security": "Sürdürülebilir üretim yöntemleri benimseyin. Hasat sonrası kayıpları azaltın.",
            "technology": "Teknoloji yatırımında maliyet-fayda analizi yapın. Yerel koşullara uygun çözümler seçin."
        }
        return advice_map.get(category, "Daha detaylı bilgi için uzman desteği alabilirsiniz.")
    
    def run_interactive(self):
        """Run interactive chat mode"""
        self.welcome_message()
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 Soru: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['çıkış', 'quit', 'q', 'exit']:
                    print("\n👋 Görüşürüz! Tarımda başarılar!")
                    break
                
                # Check for special commands
                if user_input.lower() in ['yardım', 'help']:
                    self.help_message()
                    continue
                
                if user_input.lower() in ['örnekler', 'examples']:
                    self.show_examples()
                    continue
                
                if user_input.lower() in ['kategoriler', 'categories']:
                    self.show_categories()
                    continue
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Process query
                print("\n🤖 İşleniyor...")
                response = self.process_query(user_input)
                print(f"\n{self.name}: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Görüşürüz!")
                break
            except Exception as e:
                print(f"\n❌ Beklenmeyen hata: {e}")
    
    def run_single_query(self, query: str):
        """Run single query mode"""
        print(f"🌱 {self.name} - Tek Soru Modu")
        print("=" * 50)
        print(f"👤 Soru: {query}")
        print("\n🤖 İşleniyor...")
        
        response = self.process_query(query)
        print(f"\n{self.name}: {response}")
        print("=" * 50)

def main():
    """Main function"""
    chat = SimpleAgriculturalChat()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        chat.run_single_query(query)
    else:
        # Interactive mode
        chat.run_interactive()

if __name__ == "__main__":
    main() 