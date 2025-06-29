#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌾 Tarımsal AI Chatbot - Demo Uygulaması
Ödev sunumu için basit demo arayüzü

Kullanım:
    python3 -m streamlit run demo_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import random

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🌾 Tarımsal AI Chatbot",
    page_icon="🌾",
    layout="wide"
)

# CSS stilleri - Dark mode uyumlu
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white !important;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
        color: #000000 !important;
    }
    .user-message {
        background-color: #E8F5E8;
        color: #000000 !important;
    }
    .bot-message {
        background-color: #F0F8FF;
        border-left-color: #2196F3;
        color: #000000 !important;
    }
    
    /* Dark mode metinleri görünür yap */
    .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: var(--text-color, #ffffff) !important;
    }
    
    /* Sidebar yazıları */
    .css-1d391kg {
        color: #ffffff !important;
    }
    
    /* Metrik değerleri */
    .metric-card, .css-1r6slb0 {
        color: #000000 !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
        padding: 0.5rem !important;
        border-radius: 8px !important;
    }
    
    /* Input kutular */
    .stTextInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Butonlar */
    .stButton button {
        color: #ffffff !important;
        background-color: #4CAF50 !important;
        border: none !important;
    }
    
    /* Sidebar metin */
    .sidebar .markdown-text-container {
        color: #ffffff !important;
    }
    
    /* Plotly grafikleri için */
    .js-plotly-plot {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

class DemoChatbot:
    """Demo Chatbot Sınıfı"""
    
    def __init__(self):
        self.categories = {
            "plant_disease": "🦠 Bitki Hastalıkları",
            "crop_management": "🌾 Mahsul Yönetimi", 
            "plant_genetics": "🧬 Bitki Genetiği",
            "environmental_factors": "🌡️ Çevre Faktörleri",
            "food_security": "🍽️ Gıda Güvenliği",
            "technology": "🚁 Tarım Teknolojisi"
        }
        
        # Demo yanıtları
        self.demo_responses = {
            "domates": "🍅 **Domates Hastalıkları** - Yaprak yanıklığı bakteriyel bir hastalıktır. Hasta yaprakları temizleyin, havalandırmayı artırın ve bakır bazlı fungisit uygulayın.",
            "buğday": "🌾 **Buğday Yetiştiriciliği** - Ekim zamanı çok önemlidir. Kışlık buğday Ekim-Kasım, yazlık buğday Mart-Nisan aylarında ekilir. Toprak sıcaklığı 10-15°C olmalıdır.",
            "gübre": "🌱 **Gübreleme** - Organik gübre toprak yapısını iyileştirir. Kompost, çiftlik gübresi ve yeşil gübre kullanabilirsiniz. NPK oranlarına dikkat edin.",
            "sulama": "💧 **Sulama Sistemleri** - Damlama sulama %30-50 su tasarrufu sağlar. Toprak nemini düzenli kontrol edin, aşırı su vermeyin.",
            "hastalık": "🦠 **Bitki Hastalıkları** - Erken teşhis çok önemlidir. Hasta bitki kısımlarını temizleyin, havalandırmayı artırın ve gerektiğinde fungisit kullanın."
        }

    def get_demo_response(self, question):
        """Demo yanıt üret"""
        question_lower = question.lower()
        
        # Anahtar kelime eşleştirmesi
        for keyword, response in self.demo_responses.items():
            if keyword in question_lower:
                confidence = random.uniform(0.85, 0.95)
                category = random.choice(list(self.categories.keys()))
                return {
                    'response': response,
                    'category': self.categories[category],
                    'confidence': confidence
                }
        
        # Genel yanıt
        return {
            'response': "🌾 **Tarımsal Danışmanlık** - Sorunuzu daha detaylı açıklayabilir misiniz? Hangi bitki türü, hangi problem veya hangi konuda bilgi arıyorsunuz?",
            'category': "🌱 Genel Danışmanlık",
            'confidence': 0.75
        }

def main():
    """Ana uygulama"""
    
    # Session state başlatma
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "demo_stats" not in st.session_state:
        st.session_state.demo_stats = {}

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Tarımsal AI Chatbot</h1>
        <p>Yapay Zeka Destekli Tarımsal Danışmanlık Sistemi</p>
        <p><strong>Demo Versiyonu</strong> - Ödev Sunumu</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='background: rgba(76, 205, 196, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
        <h2 style='color: #4CAF50; margin-top: 0;'>🎯 Sistem Özellikleri</h2>
        <ul style='color: #ffffff; list-style-type: none; padding-left: 0;'>
        <li style='margin: 0.5rem 0;'>• <strong>6 Uzman Kategori</strong></li>
        <li style='margin: 0.5rem 0;'>• <strong>13.2K Veri Seti</strong></li>
        <li style='margin: 0.5rem 0;'>• <strong>%89+ Doğruluk</strong></li>
        <li style='margin: 0.5rem 0;'>• <strong>Türkçe Destek</strong></li>
        <li style='margin: 0.5rem 0;'>• <strong>Web Arayüzü</strong></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: rgba(255, 193, 7, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
        <h2 style='color: #FFC107; margin-top: 0;'>💡 Örnek Sorular</h2>
        <p style='color: #ffffff; font-size: 0.9em;'>Aşağıdaki örnek sorulardan birini seçin:</p>
        </div>
        """, unsafe_allow_html=True)
        
        example_questions = [
            "Domates hastalıkları nelerdir?",
            "Buğday ne zaman ekilir?",
            "Organik gübre çeşitleri neler?",
            "Sulama sistemleri nasıl çalışır?",
            "Bitki hastalıkları nasıl önlenir?"
        ]
        
        for example in example_questions:
            if st.button(f"📝 {example}", key=f"example_{example}"):
                st.session_state.example_question = example

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Geçmişi Temizle"):
            st.session_state.messages = []
            st.rerun()

    # Ana içerik
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 💬 AI Chatbot")
        
        # Mesaj geçmişini göster
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
                    {message["content"]}<br>
                    <small>📊 {message.get('category', '')} | Güven: %{message.get('confidence', 0)*100:.1f}</small>
                </div>
                """, unsafe_allow_html=True)

        # Örnek soru kontrolü
        if hasattr(st.session_state, 'example_question'):
            user_input = st.session_state.example_question
            delattr(st.session_state, 'example_question')
        else:
            user_input = st.chat_input("Tarımsal sorunuzu sorun...")

        if user_input:
            # Kullanıcı mesajı
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Demo chatbot yanıtı
            chatbot = DemoChatbot()
            result = chatbot.get_demo_response(user_input)
            
            # Bot mesajı
            st.session_state.messages.append({
                "role": "assistant",
                "content": result['response'],
                "category": result['category'],
                "confidence": result['confidence']
            })
            
            # İstatistik güncelle
            category_key = result['category']
            if category_key in st.session_state.demo_stats:
                st.session_state.demo_stats[category_key] += 1
            else:
                st.session_state.demo_stats[category_key] = 1
            
            st.rerun()

    with col2:
        st.markdown("""
        <div style='background: rgba(33, 150, 243, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
        <h2 style='color: #2196F3; margin-top: 0;'>📊 İstatistikler</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Temel metrikler - Renkli kutu içinde
        total_questions = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f"""
        <div style='background: rgba(76, 205, 196, 0.2); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; text-align: center;'>
        <h3 style='color: #ffffff; margin: 0;'>Toplam Soru</h3>
        <h1 style='color: #4CAF50; margin: 0;'>{total_questions}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.demo_stats:
            most_asked = max(st.session_state.demo_stats.items(), key=lambda x: x[1])
            st.markdown(f"""
            <div style='background: rgba(255, 152, 0, 0.2); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; text-align: center;'>
            <h3 style='color: #ffffff; margin: 0;'>En Çok Sorulan</h3>
            <p style='color: #FF9800; margin: 0; font-weight: bold;'>{most_asked[0]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Kategori dağılımı
            if len(st.session_state.demo_stats) > 1:
                fig = px.pie(
                    values=list(st.session_state.demo_stats.values()),
                    names=list(st.session_state.demo_stats.keys()),
                    title="Soru Kategorileri",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    title_font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)

        # Proje bilgileri
        st.markdown("""
        <div style='background: rgba(76, 175, 80, 0.1); padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
        <h2 style='color: #4CAF50; margin-top: 0;'>🌾 Proje Bilgileri</h2>
        <div style='color: #ffffff; line-height: 1.6;'>
        <p><strong>Model:</strong> BERT-based Classification</p>
        <p><strong>Veri:</strong> 13,200 tarımsal metin</p>
        <p><strong>Kategoriler:</strong> 6 uzman alan</p>
        <p><strong>Doğruluk:</strong> %89+</p>
        <p><strong>Dil:</strong> Türkçe</p>
        <p><strong>Platform:</strong> Streamlit Web App</p>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("<hr style='border: 1px solid #4CAF50; margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #ffffff; padding: 1rem; background: rgba(76, 205, 196, 0.1); border-radius: 8px; margin: 1rem 0;'>
        <h3 style='color: #4CAF50; margin: 0;'>🌾 Tarımsal AI Chatbot Demo</h3>
        <p style='color: #ffffff; margin: 0.5rem 0;'>Ödev Sunumu - Yapay Zeka Destekli Tarımsal Danışmanlık</p>
        <small style='color: #B0BEC5;'>Bu bir demo uygulamasıdır. Gerçek model entegrasyonu için Model/run_model.py gereklidir.</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 