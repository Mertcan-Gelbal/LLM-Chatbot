#!/usr/bin/env python3
"""
Gelişmiş RAG Tabanlı Tarımsal Chatbot
Embedding-based retrieval ve contextual generation ile gerçek sohbet
"""

import os
import json
import torch
import numpy as np
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from transformers import (
    BertTokenizer, BertForSequenceClassification, BertConfig,
    DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig,
    AutoTokenizer, AutoModel
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.text import Text
from rich import print as rprint

console = Console()

class EmbeddingBasedRAG:
    """Embedding tabanlı RAG sistemi"""
    
    def __init__(self, data_path: str = "../Data/agricultural_bert_dataset.json"):
        self.data_path = data_path
        self.knowledge_base = []
        self.embeddings = None
        self.vectorizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Sentence transformer model (CPU optimized)
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_sentence_transformer = True
            console.print("✅ SentenceTransformer yüklendi", style="green")
        except ImportError:
            # Fallback to TF-IDF if sentence-transformers not available
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
            self.use_sentence_transformer = False
            console.print("⚠️ TF-IDF kullanılacak (SentenceTransformer bulunamadı)", style="yellow")
        
        self.load_knowledge_base()
        self.create_embeddings()
    
    def load_knowledge_base(self):
        """Bilgi tabanını yükle"""
        console.print("📚 Tarımsal bilgi tabanı yükleniyor...", style="cyan")
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Türkçe ve tarımsal içerik filtrele
            for item in data:
                text = item.get('text', '').strip()
                if len(text) > 20 and self._is_agricultural_content(text):
                    self.knowledge_base.append({
                        'text': text,
                        'label': item.get('label', 'general'),
                        'source': item.get('source', 'unknown')
                    })
            
            console.print(f"✅ {len(self.knowledge_base)} tarımsal bilgi yüklendi", style="green")
            
        except Exception as e:
            console.print(f"❌ Veri yükleme hatası: {e}", style="red")
            self.knowledge_base = self._create_fallback_knowledge()
    
    def _is_agricultural_content(self, text: str) -> bool:
        """Tarımsal içerik kontrolü"""
        agricultural_keywords = [
            'tarım', 'ziraat', 'bitki', 'plant', 'crop', 'hastalık', 'disease',
            'tohum', 'seed', 'sulama', 'irrigation', 'gübre', 'fertilizer',
            'hasat', 'harvest', 'mahsul', 'yield', 'toprak', 'soil',
            'elma', 'buğday', 'mısır', 'domates', 'patates', 'çilek',
            'yanıklık', 'çürüklük', 'virüs', 'mantar', 'zararlı', 'pest',
            'ekim', 'planting', 'yetiştirici', 'cultivation', 'agricultural'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in agricultural_keywords)
    
    def _create_fallback_knowledge(self) -> List[Dict]:
        """Yedek bilgi tabanı"""
        return [
            # Elmada erken yanıklığı
            {
                'text': 'Elmada erken yanıklığı (Erwinia amylovora) meyve ağaçlarının en ciddi bakteriyel hastalığıdır. Belirtiler: yapraklarda siyah lekeler, çiçek ve sürgün kurumaları, dal uçlarının yanık görünümü.',
                'label': 'plant_disease',
                'source': 'fallback'
            },
            {
                'text': 'Elmada erken yanıklığı tedavisi: 1) Hasta dalları %10 çamaşır suyu ile sterilize edilmiş makasla kesin 2) Kesim yerlerini bahçe macunu ile kapatın 3) Streptomisin içerikli antibiyotik sprey uygulayın 4) Bakır sülfat spreyi yapın',
                'label': 'plant_disease',
                'source': 'fallback'
            },
            {
                'text': 'Elmada erken yanıklığı korunma: Aşırı azotlu gübreden kaçının, budama aletlerini sterilize edin, çiçeklenme döneminde koruyucu ilaçlama yapın, dayanıklı çeşitler tercih edin.',
                'label': 'plant_disease',
                'source': 'fallback'
            },
            
            # Armut erken yanıklığı
            {
                'text': 'Armutta erken yanıklığı elmadakiyle aynı bakteri (Erwinia amylovora) nedeniyle oluşur. Belirtiler: çiçek kümeleri siyahlaşır, yapraklar yanık görünümü alır, dal uçları kucar şeklinde bükümlür.',
                'label': 'plant_disease',
                'source': 'fallback'
            },
            {
                'text': 'Armutta erken yanıklığı tedavisi: Hasta dalları 30 cm sağlam dokuden kesin, kesim aletlerini %70 alkol ile sterilize edin, streptomisin spreyi uygulayın, bakır bileşikli fungisitler kullanın.',
                'label': 'plant_disease',
                'source': 'fallback'
            },
            {
                'text': 'Armutta erken yanıklığı önleme: Dayanıklı armut çeşitleri seçin, çiçeklenme öncesi koruyucu spreyleme yapın, aşırı nem ve azottan kaçının, düzenli bahçe hijyeni sağlayın.',
                'label': 'plant_disease',
                'source': 'fallback'
            },
            
            # Elma yetiştirme genel
            {
                'text': 'Elma yetiştirmede sulama çok önemlidir. Haftada 2-3 kez derinlemesine sulama yapılmalı. Toprak nemi %60-70 arasında tutulmalı. Aşırı sulama kök çürüklüğüne neden olur.',
                'label': 'crop_management', 
                'source': 'fallback'
            },
            {
                'text': 'Elma ağaçlarında budama çok önemlidir. Kış aylarında (Ocak-Şubat) yapılan budama ile ağaç şekillendirilir ve hasta dallar temizlenir. Yaz budaması ise sürgün kontrolü için yapılır.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            {
                'text': 'Elma yetiştirmede toprak pH 6.0-7.0 arasında olmalı. Drenajı iyi, organik maddece zengin topraklar tercih edilir. Yılda 2-3 kez organik gübre uygulaması yapılır.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            
            # Havuç yetiştirme
            {
                'text': 'Havuç yetiştirme için derin, gevşek, taşsız toprak gerekir. Ekim derinliği 1-2 cm, sıra arası 25-30 cm olmalı. Tohum çıkışı için toprak nemli tutulmalı.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            {
                'text': 'Havuç ekim zamanı ilkbaharda Mart-Nisan, sonbaharda Ağustos-Eylül aylarıdır. Hasat 90-120 günde yapılır. Soğuk havaya dayanıklıdır.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            {
                'text': 'Havuç yetiştirmede sulama düzenli ama aşırı olmamalı. Kuraklık köklerin çatlamasına, aşırı nem ise çürümeye neden olur. Drip sulama ideal.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            {
                'text': 'Havuçta yaygın hastalıklar: Alternaria yaprak yanıklığı, bakteriyel yumuşak çürüklük, sklerotinia beyaz çürüklük. Ekim nöbeti ve iyi drenaj önemli.',
                'label': 'plant_disease',
                'source': 'fallback'
            },
            
            # Armut yetiştirme
            {
                'text': 'Armut yetiştirme elmaya benzer ama daha sıcak iklim sever. Toprak pH 6.5-7.5 arasında olmalı. Derin, drenajı iyi toprakları tercih eder.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            {
                'text': 'Armut ağaçlarında dikim mesafesi 4x5 metre olmalı. Çiçeklenme döneminde don riski var. Tozlanma için farklı çeşitler gerekli.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            
            # Buğday ekim
            {
                'text': 'Buğday ekim zamanı bölgeye göre değişir. Kışlık buğday Ekim-Kasım, yazlık buğday Mart-Nisan aylarında ekilir. Toprak sıcaklığı 8-12°C olmalı. Ekim derinliği 3-4 cm ideal.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            {
                'text': 'Buğday ekim öncesi toprak hazırlığı çok önemli. Derin sürüm, diskaro ve merdane ile toprak hazırlanır. pH 6.0-7.5 arasında olmalı. Fosfor ve potasyum gübresi ekim öncesi verilir.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            
            # Domates sarı yaprak
            {
                'text': 'Domates bitkilerinde sarı yaprak sebepleri: 1) Aşırı veya az sulama 2) Azot eksikliği 3) Magnezyum eksikliği 4) Kök hastalıkları 5) Yaşlanma süreci. En yaygın neden beslenme bozukluğudur.',
                'label': 'plant_disease',
                'source': 'fallback'
            },
            {
                'text': 'Domates sarı yaprak tedavisi: Sulama düzenini kontrol edin, azotlu gübre uygulayın, magnezyum sülfat spreyi yapın, hasta yaprakları temizleyin. Drip sulama sistemi kurun.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            {
                'text': 'Domates yetiştirmede toprak pH 6.0-6.8 arasında olmalı. Sıcaklık 18-25°C ideal. Düzenli sulama ve gübreleme gerekir. Destek sistemi kurulmalı.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            
            # Genel tarım bilgileri
            {
                'text': 'Organik tarım yöntemleri: Kompost kullanımı, crop rotation (ekim nöbeti), beneficial insects (yararlı böcekler), green manure (yeşil gübre), natural pesticides (doğal pestisitler).',
                'label': 'crop_management',
                'source': 'fallback'
            },
            {
                'text': 'Toprak pH değeri bitkiler için kritiktir. Asidik topraklar (pH < 6) kireçleme ile, alkalin topraklar (pH > 7.5) sülfür ile düzeltilir. Çoğu bitki pH 6.0-7.0 arasını sever.',
                'label': 'environmental_factors',
                'source': 'fallback'
            },
            {
                'text': 'Bitki hastalıklarından korunma: Temiz tohum, ekim nöbeti, uygun bitki aralığı, iyi drenaj, dayanıklı çeşit seçimi, biyolojik mücadele yöntemleri.',
                'label': 'plant_disease',
                'source': 'fallback'
            },
            {
                'text': 'Modern tarım teknolojileri: Precision agriculture (hassas tarım), drone teknolojisi, IoT sensörleri, GPS guided tractors, automated irrigation systems.',
                'label': 'technology',
                'source': 'fallback'
            },
            
            # Gübrelik ve beslenme
            {
                'text': 'Bitki beslenmesinde makro elementler: Azot (N) yaprak gelişimi, Fosfor (P) kök ve çiçek gelişimi, Potasyum (K) meyve kalitesi ve hastalık direnci için gereklidir.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            {
                'text': 'Organik gübre çeşitleri: Ahır gübresi, kompost, solucan gübresi, yeşil gübre. Kimyasal gübreler: Üre, DAP, Potasyum sülfat. Dengeli beslenme önemlidir.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            
            # Zararlı ve hastalık
            {
                'text': 'Entegre zararlı yönetimi (IPM): Biyolojik kontrol, kültürel yöntemler, mekanik kontrol, gerektiğinde kimyasal müdahale. Doğal dengeyi korumak önemlidir.',
                'label': 'plant_disease',
                'source': 'fallback'
            },
            {
                'text': 'Yaygın bitki hastalıkları: Külleme, pas hastalıkları, antraknoz, septoria yaprak lekesi, fusarium solgunluğu. Erken teşhis ve müdahale kritiktir.',
                'label': 'plant_disease',
                'source': 'fallback'
            },
            
            # Sebze yetiştirme genel
            {
                'text': 'Sebze yetiştirmede ekim nöbeti önemlidir: Patates-Fasulye-Lahana-Havuç döngüsü ideal. Bu toprak verimliliğini korur ve hastalıkları azaltır.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            {
                'text': 'Sera yetiştirmede havalandırma kritiktir. Nem %60-70, sıcaklık 18-25°C arasında tutulmalı. Hastalık kontrolü için hijyen önemli.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            
            # Meyve ağaçları genel
            {
                'text': 'Meyve ağaçlarında dikim zamanı sonbahar (Ekim-Kasım) veya ilkbahar (Şubat-Mart) aylarıdır. Çıplak köklü fidanlar kış, konteynerli fidanlar her mevsim dikilir.',
                'label': 'crop_management',
                'source': 'fallback'
            },
            {
                'text': 'Meyve ağaçlarında gübreleme: İlkbaharda azotlu, yaz sonunda fosforlu, sonbaharda potasyumlu gübre verilir. Organik gübre kış aylarında uygulanır.',
                'label': 'crop_management',
                'source': 'fallback'
            }
        ]
    
    def create_embeddings(self):
        """Embedding vektörleri oluştur"""
        console.print("🧠 Embedding vektörleri oluşturuluyor...", style="cyan")
        
        texts = [item['text'] for item in self.knowledge_base]
        
        if self.use_sentence_transformer:
            # Sentence transformer embeddings
            self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        else:
            # TF-IDF embeddings
            self.embeddings = self.vectorizer.fit_transform(texts).toarray()
        
        console.print(f"✅ {len(self.embeddings)} embedding oluşturuldu", style="green")
    
    def search_similar_content(self, query: str, top_k: int = 3) -> List[Dict]:
        """Benzer içerik ara"""
        if self.use_sentence_transformer:
            query_embedding = self.embedding_model.encode([query])
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        else:
            query_vec = self.vectorizer.transform([query]).toarray()
            similarities = cosine_similarity(query_vec, self.embeddings)[0]
        
        # En benzer sonuçları bul
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Minimum similarity threshold - düşürüldü
                results.append({
                    'content': self.knowledge_base[idx],
                    'similarity': float(similarities[idx])
                })
        
        # Eğer similarity çok düşükse keyword matching dene
        if not results or max([r['similarity'] for r in results]) < 0.1:
            keyword_results = self._keyword_search(query, top_k)
            results.extend(keyword_results)
        
        return results[:top_k]
    
    def _keyword_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Anahtar kelime bazlı arama"""
        query_lower = query.lower()
        keyword_matches = []
        
        # Önce plant/crop name extraction yap
        extracted_plants = self._extract_plant_names(query_lower)
        
        for idx, item in enumerate(self.knowledge_base):
            text_lower = item['text'].lower()
            match_score = 0
            
            # Plant name matching (yüksek puan)
            for plant in extracted_plants:
                if plant in text_lower:
                    match_score += 3  # Bitki adı eşleşmesi yüksek puan
            
            # Regular keyword matching
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 2 and word in text_lower:
                    match_score += 1
            
            # Context-based matching
            context_score = self._calculate_context_score(query_lower, text_lower)
            match_score += context_score
            
            if match_score > 0:
                # Normalize score
                similarity = min(match_score / max(len(query_words), 3), 1.0) * 0.9
                keyword_matches.append({
                    'content': item,
                    'similarity': similarity
                })
        
        # Sort by similarity
        keyword_matches.sort(key=lambda x: x['similarity'], reverse=True)
        return keyword_matches[:top_k]
    
    def _extract_plant_names(self, query: str) -> List[str]:
        """Sorgudan bitki/ürün adlarını çıkar"""
        plants = []
        plant_keywords = {
            'elma': ['elma', 'apple'],
            'armut': ['armut', 'pear'], 
            'havuç': ['havuç', 'carrot'],
            'domates': ['domates', 'tomato'],
            'buğday': ['buğday', 'wheat'],
            'patates': ['patates', 'potato'],
            'fasulye': ['fasulye', 'bean'],
            'lahana': ['lahana', 'cabbage'],
            'mısır': ['mısır', 'corn'],
            'çilek': ['çilek', 'strawberry']
        }
        
        for plant_name, keywords in plant_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    plants.append(plant_name)
                    break
        
        return plants
    
    def _calculate_context_score(self, query: str, text: str) -> float:
        """Bağlamsal benzerlik puanı hesapla"""
        score = 0
        
        # Hastalık kelimeleri birlikte mi?
        disease_words = ['hastalık', 'yanıklık', 'çürüklük', 'sarı', 'leke', 'tedavi']
        if any(word in query for word in disease_words) and any(word in text for word in disease_words):
            score += 1
        
        # Yetiştirme kelimeleri birlikte mi?
        growing_words = ['yetiştir', 'ekim', 'sulama', 'gübre', 'toprak', 'bakım']
        if any(word in query for word in growing_words) and any(word in text for word in growing_words):
            score += 1
        
        # Zaman kelimeleri
        time_words = ['zaman', 'ne zaman', 'when', 'dönem']
        if any(word in query for word in time_words) and any(word in text for word in time_words):
            score += 1
        
        return score

class AdvancedAgriculturalRAGChatbot:
    """Gelişmiş RAG Tabanlı Tarımsal Chatbot"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model paths
        self.bert_small_path = Path("bert_small_agricultural")
        self.distilbert_path = Path("distilbert_agricultural")
        
        # Classification model
        self.classification_model = None
        self.classification_tokenizer = None
        self.label_mapping = None
        
        # RAG System
        self.rag_system = None
        
        # Konuşma hafızası
        self.conversation_history = []
        self.current_context = {}
        self.user_profile = {}
        self.session_start = datetime.now()
        
        # Bot kişiliği
        self.bot_personality = {
            'name': 'Tarım RAG AI',
            'expertise': 'Detaylı tarımsal bilgi ve pratik öneriler',
            'style': 'Bilimsel ama anlaşılır, samimi ve yardımsever'
        }
        
        console.print("🤖 Gelişmiş RAG Tarımsal AI yükleniyor...", style="bold green")
        self.initialize_models()
        self._welcome_user()
    
    def initialize_models(self):
        """Modelleri başlat"""
        try:
            # 1. RAG sistemi
            console.print("📚 RAG sistemi yükleniyor...")
            self.rag_system = EmbeddingBasedRAG()
            
            # 2. Classification model
            console.print("🧠 Sınıflandırma modeli yükleniyor...")
            self._load_classification_model()
            
            console.print("✅ Tüm sistemler hazır!", style="bold green")
            
        except Exception as e:
            console.print(f"❌ Model yükleme hatası: {e}", style="bold red")
            raise
    
    def _load_classification_model(self):
        """Sınıflandırma modelini yükle"""
        try:
            # Label mapping
            with open(self.bert_small_path / 'label_mapping.json', 'r') as f:
                self.label_mapping = json.load(f)
            
            # DistilBERT model
            with open(self.distilbert_path / 'config.json', 'r') as f:
                config_dict = json.load(f)
            config = DistilBertConfig.from_dict(config_dict)
            
            self.classification_model = DistilBertForSequenceClassification(config)
            state_dict = torch.load(self.distilbert_path / 'pytorch_model.bin', 
                                  map_location=self.device, weights_only=False)
            self.classification_model.load_state_dict(state_dict)
            self.classification_model.to(self.device)
            self.classification_model.eval()
            
            self.classification_tokenizer = DistilBertTokenizer.from_pretrained(str(self.distilbert_path))
            
        except Exception as e:
            console.print(f"⚠️ Sınıflandırma modeli yüklenemedi: {e}", style="yellow")
    
    def _welcome_user(self):
        """Kullanıcıyı karşıla"""
        welcome_panel = Panel.fit(
            "🌾 **Gelişmiş RAG Tabanlı Tarımsal AI'ya Hoş Geldin!**\n\n"
            "✨ **Yeni Özellikler:**\n"
            "🔍 **Akıllı Arama**: 1000+ tarımsal makaleyi anında tarar\n"
            "🧠 **RAG Mimarisi**: Soruna en uygun bilgileri bulur\n"
            "💬 **Gerçek Sohbet**: Bağlamsal ve akıcı konuşma\n"
            "📚 **Detaylı Bilgi**: Bilimsel kaynaklı cevaplar\n"
            "🎯 **Spesifik Öneriler**: Pratik tarımsal çözümler\n\n"
            "💡 **Örnek Sorular:**\n"
            "• 'Elmada erken yanıklığı nedir ve nasıl tedavi edilir?'\n"
            "• 'Buğday ekim zamanı ne zaman?'\n"
            "• 'Domates bitkilerinde sarı yaprak sorunu'\n\n"
            "🗨️ Sorunuzu sorun, detaylı cevap vereyim! 🌱",
            title="🚀 Gelişmiş Tarım AI",
            style="bold green"
        )
        console.print(welcome_panel)
    
    def classify_intent(self, text: str) -> Dict:
        """Kullanıcı niyetini sınıflandır"""
        # Önce keyword bazlı kategori belirleme dene
        keyword_category = self._keyword_based_classification(text)
        
        if not self.classification_model:
            return {'category': keyword_category or 'general_agriculture', 'confidence': 0.5}
        
        try:
            inputs = self.classification_tokenizer(
                text, return_tensors="pt", truncation=True, 
                padding=True, max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.classification_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                predicted_id = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_id].item()
            
            predicted_category = self.label_mapping['id_to_label'][str(predicted_id)]
            
            # Eğer model confidence düşükse keyword bazlı kategoriyi tercih et
            if confidence < 0.6 and keyword_category:
                return {'category': keyword_category, 'confidence': 0.7}
            
            return {
                'category': predicted_category,
                'confidence': confidence
            }
            
        except Exception as e:
            return {'category': keyword_category or 'general_agriculture', 'confidence': 0.5}
    
    def _keyword_based_classification(self, text: str) -> Optional[str]:
        """Anahtar kelime bazlı kategori belirleme"""
        text_lower = text.lower()
        
        # Kategori anahtar kelimeleri
        category_keywords = {
            'plant_disease': [
                'hastalık', 'yanıklık', 'çürüklük', 'sarı yaprak', 'leke', 'mantar', 
                'virüs', 'bakteri', 'pest', 'zararlı', 'böcek', 'kuruma', 'solgunluk',
                'tedavi', 'ilaç', 'sprey', 'disease', 'pathogen', 'infection'
            ],
            'crop_management': [
                'ekim', 'tohum', 'sulama', 'gübre', 'budama', 'hasat', 'yetiştirme',
                'toprak', 'beslenme', 'bakım', 'planting', 'irrigation', 'fertilizer',
                'cultivation', 'management', 'growing', 'farming'
            ],
            'technology': [
                'teknoloji', 'drone', 'sensör', 'gps', 'otomasyon', 'robot', 'makine',
                'technology', 'automation', 'sensor', 'precision', 'digital'
            ],
            'environmental_factors': [
                'iklim', 'hava', 'sıcaklık', 'nem', 'ph', 'çevre', 'toprak', 'su',
                'climate', 'weather', 'temperature', 'humidity', 'soil', 'environmental'
            ],
            'food_security': [
                'gıda', 'güvenlik', 'kalite', 'saklama', 'muhafaza', 'food', 'security',
                'quality', 'storage', 'preservation'
            ]
        }
        
        # Her kategori için puan hesapla
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            if score > 0:
                category_scores[category] = score
        
        # En yüksek puanlı kategoriyi döndür
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def retrieve_relevant_knowledge(self, query: str, category: str) -> List[Dict]:
        """İlgili bilgileri getir"""
        # RAG ile benzer içerik ara
        search_results = self.rag_system.search_similar_content(query, top_k=8)
        
        if not search_results:
            return []
        
        # Önce aynı kategorideki sonuçları getir
        category_results = []
        other_results = []
        
        for result in search_results:
            if result['content']['label'] == category:
                category_results.append(result)
            elif result['similarity'] > 0.3:  # Yüksek similarity'li diğer kategoriler
                other_results.append(result)
        
        # Sonuçları birleştir - önce kategori eşleşmeleri, sonra yüksek similarity'liler
        final_results = category_results + other_results
        
        # En az 1 sonuç garanti et
        if not final_results and search_results:
            final_results = search_results[:1]
        
        return final_results[:3]  # En iyi 3 sonuç
    
    def generate_contextual_response(self, user_query: str) -> str:
        """Bağlamsal cevap üret"""
        # 1. Intent sınıflandırması
        intent = self.classify_intent(user_query)
        category = intent['category']
        confidence = intent['confidence']
        
        # 2. İlgili bilgileri getir
        knowledge_results = self.retrieve_relevant_knowledge(user_query, category)
        
        # 3. Cevap oluştur
        if not knowledge_results:
            return self._generate_fallback_response(user_query, category)
        
        # 4. Bağlamsal cevap üret
        response = self._create_rich_response(user_query, knowledge_results, category, confidence)
        
        # 5. Konuşma geçmişine ekle
        self.conversation_history.append({
            'user': user_query,
            'bot': response,
            'category': category,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'knowledge_used': len(knowledge_results)
        })
        
        return response
    
    def _create_rich_response(self, query: str, knowledge_results: List[Dict], category: str, confidence: float) -> str:
        """Zengin cevap oluştur"""
        response_parts = []
        
        # 1. Giriş ve bağlam
        intro = self._create_intro(query, category, confidence)
        if intro:
            response_parts.append(intro)
        
        # 2. Ana bilgi - en yüksek similarity'li içerik
        main_content = self._extract_main_content(knowledge_results[0])
        response_parts.append(main_content)
        
        # 3. Ek bilgiler varsa
        if len(knowledge_results) > 1:
            additional_info = self._create_additional_info(knowledge_results[1:])
            if additional_info:
                response_parts.append(additional_info)
        
        # 4. Pratik öneriler
        practical_tips = self._generate_practical_tips(query, category)
        if practical_tips:
            response_parts.append(practical_tips)
        
        # 5. Takip sorusu
        followup = self._create_followup_question(query, category)
        if followup:
            response_parts.append(followup)
        
        return '\n\n'.join(response_parts)
    
    def _create_intro(self, query: str, category: str, confidence: float) -> str:
        """Giriş cümlesi oluştur"""
        query_lower = query.lower()
        
        # Bitki adına göre spesifik intro
        if 'elma' in query_lower and 'yanıklık' in query_lower:
            return "Elmada erken yanıklığı konusunda size detaylı bilgi vereyim:"
        elif 'armut' in query_lower and 'yanıklık' in query_lower:
            return "Armutta erken yanıklığı problemi için şu bilgileri paylaşabilirim:"
        elif 'havuç' in query_lower and 'yetiştir' in query_lower:
            return "Havuç yetiştirme hakkında size kapsamlı bilgi verebilirim:"
        elif 'domates' in query_lower and 'sarı' in query_lower:
            return "Domates bitkilerinde sarı yaprak sorunu için şu açıklamaları yapabilirim:"
        elif 'buğday' in query_lower and 'ekim' in query_lower:
            return "Buğday ekim zamanı ve yöntemleri hakkında bilgiler:"
        
        # Confidence'a göre genel intro
        if confidence > 0.8:
            intros = [
                "Bu konuda size detaylı bilgi verebilirim!",
                "Harika bir soru! Araştırmalardan şunu öğreniyoruz:",
                "Bu konu hakkında elimde güzel bilgiler var:",
                "Tam bu konuda yardımcı olabilirim!"
            ]
        else:
            intros = [
                "Bu konuyla ilgili bulduğum bilgiler şöyle:",
                "Araştırmalar bu konuda şunları gösteriyor:",
                "Bu alanda yapılan çalışmalara göre:",
                "Elimdeki tarımsal kaynaklarda şu bilgiler var:"
            ]
        
        return random.choice(intros)
    
    def _extract_main_content(self, result: Dict) -> str:
        """Ana içeriği çıkar ve işle"""
        content = result['content']['text']
        similarity = result['similarity']
        
        # Similarity'e göre güven ifadesi ekle
        if similarity > 0.8:
            confidence_phrase = "🎯 **Tam aradığınız bilgi:**"
        elif similarity > 0.6:
            confidence_phrase = "📚 **Doğrudan ilgili bilgi:**"
        elif similarity > 0.4:
            confidence_phrase = "💡 **İlgili bilgi:**"
        else:
            confidence_phrase = "ℹ️ **Genel bilgi:**"
        
        # İçeriği temizle ve düzenle
        cleaned_content = self._clean_and_format_content(content)
        
        return f"{confidence_phrase}\n{cleaned_content}"
    
    def _clean_and_format_content(self, content: str) -> str:
        """İçeriği temizle ve formatla"""
        # Teknik terimleri Türkçe'ye çevir
        translations = {
            'disease': 'hastalık',
            'pathogen': 'hastalık etmeni',
            'cultivation': 'yetiştiricilik',
            'management': 'yönetim',
            'treatment': 'tedavi',
            'symptoms': 'belirtiler',
            'control': 'kontrol'
        }
        
        for en, tr in translations.items():
            content = content.replace(en, tr)
        
        # Çok uzun cümleleri kısalt
        if len(content) > 300:
            content = content[:300] + "..."
        
        return content
    
    def _create_additional_info(self, additional_results: List[Dict]) -> str:
        """Ek bilgi oluştur"""
        if not additional_results:
            return None
        
        additional_parts = ["📋 **Ek Bilgiler:**"]
        
        for i, result in enumerate(additional_results[:2], 1):
            content = result['content']['text']
            cleaned = self._clean_and_format_content(content)
            additional_parts.append(f"{i}. {cleaned}")
        
        return '\n'.join(additional_parts)
    
    def _generate_practical_tips(self, query: str, category: str) -> str:
        """Praktik öneriler üret"""
        query_lower = query.lower()
        
        # Spesifik bitki + sorun kombinasyonları
        if 'elma' in query_lower and 'yanıklık' in query_lower:
            return '\n'.join([
                "🚨 **Acil Müdahale - Elmada Erken Yanıklığı:**",
                "• Hasta dalları hemen kesin (30 cm sağlam kısımdan)",
                "• Kesim aletlerini %10 çamaşır suyu ile sterilize edin",
                "• Streptomisin içerikli sprey uygulayın",
                "• Etrafındaki sağlam ağaçları da koruma amaçlı ilaçlayın"
            ])
        elif 'armut' in query_lower and 'yanıklık' in query_lower:
            return '\n'.join([
                "🚨 **Acil Müdahale - Armutta Erken Yanıklığı:**",
                "• Hasta dalları 30 cm sağlam dokuden kesin",
                "• Kesim aletlerini %70 alkol ile sterilize edin",
                "• Bakır bileşikli fungisit uygulayın",
                "• Çiçeklenme öncesi koruyucu spreyleme yapın"
            ])
        elif 'havuç' in query_lower:
            return '\n'.join([
                "🥕 **Havuç Yetiştirme İpuçları:**",
                "• Derin, gevşek, taşsız toprak hazırlayın",
                "• Ekim derinliği 1-2 cm, sıra arası 25-30 cm",
                "• Tohum çıkışı için toprağı nemli tutun",
                "• Düzenli ama aşırı olmayan sulama yapın"
            ])
        elif 'domates' in query_lower and 'sarı' in query_lower:
            return '\n'.join([
                "🍅 **Domates Sarı Yaprak Çözümü:**",
                "• Sulama düzenini kontrol edin",
                "• Azotlu gübre uygulayın",
                "• Magnezyum sülfat spreyi yapın",
                "• Hasta yaprakları temizleyin"
            ])
        elif 'buğday' in query_lower and 'ekim' in query_lower:
            return '\n'.join([
                "🌾 **Buğday Ekim Başarı İpuçları:**",
                "• Toprak sıcaklığını kontrol edin (8-12°C)",
                "• Ekim derinliği 3-4 cm olmalı",
                "• pH 6.0-7.5 arasında tutun",
                "• Fosfor ve potasyum gübresi ekim öncesi verin"
            ])
        
        # Kategori bazlı genel ipuçları
        tips_by_category = {
            'plant_disease': [
                "🔍 **Hastalık Yönetimi İpuçları:**",
                "• Erken teşhis çok önemli",
                "• Hasta kısımları hemen temizleyin",
                "• Bahçe hijyenine dikkat edin",
                "• Koruyucu ilaçlama yapın"
            ],
            'crop_management': [
                "🌱 **Yetiştirme İpuçları:**",
                "• Toprak analizini yaptırın",
                "• Uygun ekim zamanını seçin",
                "• Düzenli gözlem yapın",
                "• Dengeli gübreleme uygulayın"
            ]
        }
        
        if category in tips_by_category:
            return '\n'.join(tips_by_category[category])
        
        # Çok genel ipuçları (son çare)
        return '\n'.join([
            "💡 **Genel Öneriler:**",
            "• Düzenli gözlem yapın",
            "• Koruyucu önlemler alın",
            "• Uzman desteği alın",
            "• Organik yöntemleri tercih edin"
        ])
    
    def _create_followup_question(self, query: str, category: str) -> str:
        """Takip sorusu oluştur"""
        followup_questions = {
            'plant_disease': [
                "🤔 Hangi bitkide bu sorunu yaşıyorsunuz?",
                "📸 Fotoğraf paylaşabilir misiniz? Daha net teşhis yapabilirim.",
                "⏰ Bu belirtiler ne kadar süredir görülüyor?",
                "🌱 Daha önce hangi tedavileri denediniz?"
            ],
            'crop_management': [
                "🗺️ Hangi bölgede tarım yapıyorsunuz?",
                "📏 Ne kadar alan için bilgi istiyorsunuz?",
                "🎯 En çok hangi konuda zorlanıyorsunuz?",
                "⚡ Acil bir durum mu, yoksa planlama aşamasında mı?"
            ],
            'technology': [
                "💰 Bütçeniz ne kadar?",
                "⚙️ Hangi teknolojiler daha çok ilginizi çekiyor?",
                "🔧 Teknik konularda deneyiminiz nasıl?",
                "🎯 Hangi sorunları çözmek istiyorsunuz?"
            ]
        }
        
        questions = followup_questions.get(category, [
            "🤝 Bu bilgiler yardımcı oldu mu?",
            "❓ Başka hangi konularda yardım istiyorsunuz?",
            "💬 Daha detayına inmek ister misiniz?"
        ])
        
        return random.choice(questions)
    
    def _generate_fallback_response(self, query: str, category: str) -> str:
        """Bilgi bulunamadığında yedek cevap"""
        return (
            f"🔍 '{query}' hakkında veri tabanımda spesifik bilgi bulamadım.\n\n"
            f"💡 **Yapabileceğim:**\n"
            f"• Size genel {category} bilgisi verebilirim\n"
            f"• Sorunuzu biraz daha spesifik hale getirebilirsiniz\n"
            f"• Hangi bitki/ürün hakkında olduğunu belirtebilirsiniz\n\n"
            f"🤝 Sorunuzu yeniden formüle ederek tekrar deneyebilir misiniz?"
        )
    
    def show_conversation_stats(self):
        """Konuşma istatistikleri"""
        if not self.conversation_history:
            console.print("Henüz konuşma başlamadı! 😊", style="yellow")
            return
        
        # İstatistik tablosu
        table = Table(title="📊 Sohbet İstatistikleri")
        table.add_column("Metrik", style="cyan")
        table.add_column("Değer", style="green")
        
        duration = datetime.now() - self.session_start
        duration_min = duration.seconds // 60
        
        category_counts = {}
        total_confidence = 0
        total_knowledge = 0
        
        for conv in self.conversation_history:
            cat = conv.get('category', 'unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
            total_confidence += conv.get('confidence', 0)
            total_knowledge += conv.get('knowledge_used', 0)
        
        avg_confidence = total_confidence / len(self.conversation_history) if self.conversation_history else 0
        avg_knowledge = total_knowledge / len(self.conversation_history) if self.conversation_history else 0
        
        table.add_row("Sohbet Süresi", f"{duration_min} dakika")
        table.add_row("Toplam Mesaj", str(len(self.conversation_history)))
        table.add_row("Ortalama Güven", f"{avg_confidence:.2f}")
        table.add_row("Ortalama Bilgi Kullanımı", f"{avg_knowledge:.1f}")
        table.add_row("En Çok Sorulan", max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else "N/A")
        
        console.print(table)
    
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
                elif user_input.lower() in ['stats', 'istatistik']:
                    self.show_conversation_stats()
                    continue
                elif user_input.lower() in ['geçmiş', 'history']:
                    self._show_history()
                    continue
                
                # Ana cevap üretimi
                console.print("\n🔍 Bilgi tabanında aranıyor ve analiz ediliyor...", style="italic yellow")
                
                response = self.generate_contextual_response(user_input)
                
                # Cevabı göster
                console.print(f"\n🤖 [bold cyan]Tarım RAG AI[/bold cyan]:\n{response}")
                
            except KeyboardInterrupt:
                self._farewell()
                break
            except Exception as e:
                console.print(f"\n❌ Bir hata oluştu: {e}", style="bold red")
                console.print("Tekrar dener misiniz? 😊", style="yellow")
    
    def _show_help(self):
        """Yardım göster"""
        help_panel = Panel.fit(
            "🆘 **RAG Chatbot Kullanım Kılavuzu**\n\n"
            "💬 **Normal Sohbet:**\n"
            "   Tarımsal sorularınızı doğal dilde sorun\n\n"
            "🎯 **Spesifik Sorular:**\n"
            "   • 'Elmada erken yanıklığı nasıl tedavi edilir?'\n"
            "   • 'Buğday ekimi için ideal toprak koşulları neler?'\n"
            "   • 'Domates bitkilerinde sarı yaprak sorunu'\n\n"
            "⚡ **Özel Komutlar:**\n"
            "   • 'stats' - Sohbet istatistikleri\n"
            "   • 'geçmiş' - Konuşma geçmişi\n"
            "   • 'yardım' - Bu yardım menüsü\n"
            "   • 'çıkış' - Programdan çık\n\n"
            "💡 **İpucu:** Ne kadar detaylı soru sorarsanız,\n"
            "   o kadar spesifik cevap alırsınız!",
            title="🆘 Yardım",
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
            category = conv.get('category', 'unknown')
            confidence = conv.get('confidence', 0)
            
            console.print(f"\n{i}. [{time_str}] Kategori: {category} (Güven: {confidence:.2f})")
            console.print(f"   Soru: {conv['user'][:80]}...")
            console.print(f"   Cevap: {conv['bot'][:100]}...")
    
    def _farewell(self):
        """Vedalaşma"""
        duration = datetime.now() - self.session_start
        duration_min = duration.seconds // 60
        conversation_count = len(self.conversation_history)
        
        farewell_panel = Panel.fit(
            f"👋 **Hoşçakalın!**\n\n"
            f"📊 **Sohbet Özeti:**\n"
            f"⏰ Süre: {duration_min} dakika\n"
            f"💬 Mesaj: {conversation_count} adet\n"
            f"🧠 RAG aramaları başarıyla tamamlandı\n\n"
            f"🌾 **İyi tarımlar dilerim!**\n"
            f"🤝 İhtiyacınız olduğunda tekrar gelin!",
            title="👋 Görüşmek Üzere",
            style="bold green"
        )
        console.print(farewell_panel)

def main():
    """Ana fonksiyon"""
    try:
        bot = AdvancedAgriculturalRAGChatbot()
        bot.chat_loop()
    except Exception as e:
        console.print(f"❌ Program başlatılamadı: {e}", style="bold red")

if __name__ == "__main__":
    main() 