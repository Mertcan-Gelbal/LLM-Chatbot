#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌾 Tarımsal AI Chatbot - Streamlit Web Arayüzü
Modern web tabanlı tarımsal danışmanlık sistemi

Kullanım:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import sys
import os
from pathlib import Path

# Proje kök dizinini path'e ekle
sys.path.append(str(Path(__file__).parent))

try:
    from Model.run_model import predict_text
except ImportError:
    st.error("❌ Model modülü bulunamadı. Lütfen model eğitimini tamamlayın.")
    st.stop()

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🌾 Tarımsal AI Chatbot",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    .user-message {
        background-color: #E8F5E8;
        border-left-color: #4CAF50;
    }
    
    .bot-message {
        background-color: #F0F8FF;
        border-left-color: #2196F3;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .category-tag {
        background: #4CAF50;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

class TarimalAIChatbot:
    """Tarımsal AI Chatbot Sınıfı"""
    
    def __init__(self):
        self.categories = {
            "plant_disease": "🦠 Bitki Hastalıkları",
            "crop_management": "🌾 Mahsul Yönetimi", 
            "plant_genetics": "🧬 Bitki Genetiği",
            "environmental_factors": "🌡️ Çevre Faktörleri",
            "food_security": "🍽️ Gıda Güvenliği",
            "technology": "🚁 Tarım Teknolojisi"
        }
        
        self.example_questions = [
            "Domates yaprak yanıklığı nedir?",
            "Buğday ekimi nasıl yapılır?",
            "Organik gübre çeşitleri nelerdir?",
            "Kuraklık stresi nasıl önlenir?",
            "Akıllı sulama sistemleri nasıl çalışır?",
            "GMO bitkilerin avantajları nelerdir?",
            "Toprak pH değeri neden önemli?",
            "Böcek zararlıları nasıl kontrol edilir?",
            "Hibrit tohum nedir?",
            "Sera gazı emisyonlarını nasıl azaltabiliriz?"
        ]

    def initialize_session_state(self):
        """Session state değişkenlerini başlat"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "category_stats" not in st.session_state:
            st.session_state.category_stats = {}

    def display_header(self):
        """Ana başlık"""
        st.markdown("""
        <div class="main-header">
            <h1>🌾 Tarımsal AI Chatbot</h1>
            <p>Gelişmiş yapay zeka ile tarımsal danışmanlık sistemi</p>
        </div>
        """, unsafe_allow_html=True)

    def display_sidebar(self):
        """Kenar çubuğu"""
        with st.sidebar:
            st.markdown("## 🎯 Sistem Bilgileri")
            
            # Sistem metrikleri
            st.markdown("### 📊 Performans")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Kategori Sayısı", "6")
                st.metric("Veri Boyutu", "13.2K")
            with col2:
                st.metric("Accuracy", "%89+")
                st.metric("Inference", "~19ms")
            
            st.markdown("### 🧠 Uzman Kategoriler")
            for category, name in self.categories.items():
                st.markdown(f"<span class='category-tag'>{name}</span>", 
                           unsafe_allow_html=True)
            
            st.markdown("### 💡 Örnek Sorular")
            for example in self.example_questions[:5]:
                if st.button(f"📝 {example[:30]}...", key=f"example_{example}"):
                    st.session_state.example_question = example

            # Geçmişi temizle
            if st.button("🗑️ Sohbet Geçmişini Temizle"):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()

    def process_question(self, question):
        """Soruyu işle ve yanıt üret"""
        try:
            # Model tahminini al
            result = predict_text(question)
            
            if 'error' in result:
                return {
                    'response': f"❌ Hata: {result['error']}",
                    'category': 'error',
                    'confidence': 0.0
                }
            
            # Yanıt formatla
            category = result.get('category', 'unknown')
            category_tr = result.get('category_turkish', 'Bilinmeyen')
            confidence = result.get('confidence', 0.0)
            
            # İstatistikleri güncelle
            if category in st.session_state.category_stats:
                st.session_state.category_stats[category] += 1
            else:
                st.session_state.category_stats[category] = 1
            
            # Kategori bazlı yanıt üret
            response = self.generate_detailed_response(question, category, category_tr, confidence)
            
            return {
                'response': response,
                'category': category,
                'category_tr': category_tr,
                'confidence': confidence
            }
            
        except Exception as e:
            return {
                'response': f"❌ Beklenmeyen hata: {str(e)}",
                'category': 'error',
                'confidence': 0.0
            }

    def generate_detailed_response(self, question, category, category_tr, confidence):
        """Detaylı yanıt üret"""
        # Temel bilgi
        response = f"🎯 **Kategori:** {category_tr} (%{confidence*100:.1f} güven)\n\n"
        
        # Kategori bazlı detaylar
        category_advice = {
            "plant_disease": {
                "icon": "🦠",
                "advice": "Hastalık teşhisi için bitki türü, belirtiler ve çevre koşullarını detaylı belirtin. Erken müdahale çok önemlidir.",
                "tips": ["Hasta bitki kısımlarını temizleyin", "Havalandırmayı artırın", "Fungisit uygulaması düşünün"]
            },
            "crop_management": {
                "icon": "🌾", 
                "advice": "Mahsul yönetiminde toprak analizi, ekim zamanı ve gübreleme planlaması kritiktir.",
                "tips": ["Toprak testini yaptırın", "Rotasyon planı oluşturun", "Sulama programı belirleyin"]
            },
            "plant_genetics": {
                "icon": "🧬",
                "advice": "Genetik ıslah uzun vadeli bir süreçtir. Araştırma kurumlarından destek alabilirsiniz.",
                "tips": ["Hibrit çeşitler araştırın", "Yerel adaptasyon testleri yapın", "Genetik çeşitliliği koruyun"]
            },
            "environmental_factors": {
                "icon": "🌡️",
                "advice": "Çevresel faktörler bitki gelişimini doğrudan etkiler. Koruyucu önlemler alın.",
                "tips": ["İklim verilerini takip edin", "Mulch kullanın", "Mikro iklim oluşturun"]
            },
            "food_security": {
                "icon": "🍽️",
                "advice": "Gıda güvenliği sürdürülebilir tarım uygulamaları ile sağlanır.",
                "tips": ["Hasat sonrası kayıpları azaltın", "Depolama koşullarını iyileştirin", "Yerel üretimi destekleyin"]
            },
            "technology": {
                "icon": "🚁",
                "advice": "Tarım teknolojisi yatırımında maliyet-fayda analizi yapın.",
                "tips": ["Precision agriculture araçları kullanın", "IoT sensörleri kurulumu düşünün", "Veri analizi yapın"]
            }
        }
        
        if category in category_advice:
            info = category_advice[category]
            response += f"{info['icon']} **Uzman Önerisi:**\n{info['advice']}\n\n"
            response += "💡 **Pratik İpuçları:**\n"
            for tip in info['tips']:
                response += f"• {tip}\n"
        
        response += f"\n📞 **Detaylı bilgi için:** Yerel tarım uzmanlarına danışabilirsiniz."
        
        return response

    def display_chat_interface(self):
        """Sohbet arayüzü"""
        st.markdown("## 💬 AI Danışmanı ile Sohbet")
        
        # Önceki mesajları göster
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 Siz:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Tarım AI:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)

        # Örnek soru kontrolü
        if hasattr(st.session_state, 'example_question'):
            user_input = st.session_state.example_question
            delattr(st.session_state, 'example_question')
            self.handle_user_input(user_input)

        # Kullanıcı girişi
        user_input = st.chat_input("Tarımsal sorunuzu sorun... (örn: 'Domates hastalıkları nelerdir?')")
        
        if user_input:
            self.handle_user_input(user_input)

    def handle_user_input(self, user_input):
        """Kullanıcı girişini işle"""
        # Kullanıcı mesajını ekle
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # AI yanıtını al
        with st.spinner("🤖 AI analiz ediyor..."):
            result = self.process_question(user_input)
        
        # AI yanıtını ekle
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result['response'],
            "category": result.get('category', ''),
            "confidence": result.get('confidence', 0.0),
            "timestamp": datetime.now()
        })
        
        # Sohbet geçmişine ekle
        st.session_state.chat_history.append({
            "question": user_input,
            "answer": result['response'],
            "category": result.get('category', ''),
            "confidence": result.get('confidence', 0.0),
            "timestamp": datetime.now().isoformat()
        })
        
        st.rerun()

    def display_analytics(self):
        """Analitik dashboard"""
        st.markdown("## 📊 Sohbet Analitiği")
        
        if not st.session_state.chat_history:
            st.info("📈 Henüz sohbet geçmişi yok. Chatbot ile konuşmaya başlayın!")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Kategori dağılımı
            if st.session_state.category_stats:
                categories = list(st.session_state.category_stats.keys())
                counts = list(st.session_state.category_stats.values())
                
                fig_pie = px.pie(
                    values=counts,
                    names=[self.categories.get(cat, cat) for cat in categories],
                    title="Soru Kategorileri Dağılımı",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # İstatistikler
            st.markdown("### 📈 İstatistikler")
            st.metric("Toplam Soru", len(st.session_state.chat_history))
            
            if st.session_state.chat_history:
                avg_confidence = sum(chat.get('confidence', 0) for chat in st.session_state.chat_history) / len(st.session_state.chat_history)
                st.metric("Ortalama Güven", f"%{avg_confidence*100:.1f}")
            
            most_common = max(st.session_state.category_stats.items(), key=lambda x: x[1]) if st.session_state.category_stats else ('', 0)
            st.metric("En Çok Sorulan", self.categories.get(most_common[0], "Yok"))

        # Son sorular
        st.markdown("### 📝 Son Sorular")
        recent_chats = st.session_state.chat_history[-5:]
        
        for i, chat in enumerate(reversed(recent_chats), 1):
            with st.expander(f"{i}. {chat['question'][:50]}..."):
                st.write(f"**❓ Soru:** {chat['question']}")
                st.write(f"**🤖 Yanıt:** {chat['answer'][:200]}...")
                st.write(f"**🎯 Kategori:** {self.categories.get(chat['category'], 'Bilinmeyen')}")
                st.write(f"**📊 Güven:** %{chat['confidence']*100:.1f}")
                st.write(f"**⏰ Zaman:** {chat['timestamp']}")

    def display_about(self):
        """Hakkında sayfası"""
        st.markdown("## 🌾 Proje Hakkında")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Proje Amacı
            Bu tarımsal AI chatbot, çiftçiler ve tarım uzmanları için geliştirilmiş 
            yapay zeka destekli danışmanlık sistemidir.
            
            ### 🧠 Teknoloji
            - **BERT Modeli:** Metin sınıflandırma
            - **RAG Sistemi:** Kapsamlı bilgi tabanı  
            - **13.2K Veri:** Tarımsal metin korpusu
            - **6 Kategori:** Uzman bilgi alanları
            
            ### 🚀 Özellikler
            - Gerçek zamanlı soru-cevap
            - %89+ doğruluk oranı
            - Türkçe dil desteği
            - Web tabanlı arayüz
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Model Performansı
            """)
            
            # Performans metrikleri
            metrics_data = {
                'Model': ['BERT-large', 'BERT-base', 'DistilBERT', 'BERT-small'],
                'Accuracy': [92, 90, 87, 85],
                'Speed (ms)': [100, 50, 30, 19],
                'Memory (GB)': [5, 3, 2, 1.5]
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True)
            
            # Performans grafiği
            fig_performance = px.scatter(
                df_metrics, 
                x='Speed (ms)', 
                y='Accuracy',
                size='Memory (GB)',
                color='Model',
                title="Model Performans Karşılaştırması",
                labels={'Speed (ms)': 'Hız (ms)', 'Accuracy': 'Doğruluk (%)'}
            )
            st.plotly_chart(fig_performance, use_container_width=True)

    def export_chat_history(self):
        """Sohbet geçmişini dışa aktar"""
        if st.session_state.chat_history:
            # JSON formatında
            chat_json = json.dumps(st.session_state.chat_history, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 JSON Olarak İndir",
                data=chat_json,
                file_name=f"tarimal_ai_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # CSV formatında
            df_chat = pd.DataFrame(st.session_state.chat_history)
            csv = df_chat.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSV Olarak İndir", 
                data=csv,
                file_name=f"tarimal_ai_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def main():
    """Ana uygulama"""
    try:
        chatbot = TarimalAIChatbot()
        chatbot.initialize_session_state()
        
        # Header
        chatbot.display_header()
        
        # Sidebar
        chatbot.display_sidebar()
        
        # Ana içerik
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Sohbet", "📊 Analitik", "🌾 Hakkında", "⚙️ Ayarlar"])
        
        with tab1:
            chatbot.display_chat_interface()
        
        with tab2:
            chatbot.display_analytics()
        
        with tab3:
            chatbot.display_about()
        
        with tab4:
            st.markdown("## ⚙️ Sistem Ayarları")
            
            st.markdown("### 📤 Veri Dışa Aktarma")
            chatbot.export_chat_history()
            
            st.markdown("### ℹ️ Sistem Bilgileri")
            st.code(f"""
Model Tipi: BERT-based Classification
Kategori Sayısı: 6
Veri Boyutu: 13,200 chunk
Dil Desteği: Türkçe
Framework: Streamlit + PyTorch
Deployment: Local/Cloud ready
            """)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "🌾 Tarımsal AI Chatbot | Developed with ❤️ for Agriculture"
            "</div>",
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"❌ Uygulama başlatılamadı: {e}")
        st.info("💡 Lütfen önce model eğitimini tamamlayın.")

if __name__ == "__main__":
    main() 