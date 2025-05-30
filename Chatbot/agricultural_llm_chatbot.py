#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌱 Agricultural LLM Chatbot System
AI-powered agricultural expert chatbot with RAG capabilities

Features:
- BERT-based text classification
- Knowledge retrieval from agricultural database
- Generative responses with context
- Multi-language support (Turkish/English)
- Interactive web interface

Author: Botanical BERT Team
Version: 1.0.0
"""

import os
import sys
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, set_seed
)
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
set_seed(42)
torch.manual_seed(42)
np.random.seed(42)

class AgriculturalKnowledgeBase:
    """Agricultural knowledge base with semantic search capabilities"""
    
    def __init__(self, data_path: str = "../Data"):
        self.data_path = Path(data_path)
        self.embeddings_model = None
        self.knowledge_index = None
        self.documents = []
        self.metadata = []
        
    def load_knowledge_base(self) -> bool:
        """Load and index agricultural knowledge"""
        try:
            # Load embedding model
            logger.info("Loading sentence transformer model...")
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load agricultural documents
            logger.info("Loading agricultural knowledge base...")
            self._load_agricultural_documents()
            
            # Create vector index
            logger.info("Creating vector index...")
            self._create_vector_index()
            
            logger.info(f"Knowledge base loaded: {len(self.documents)} documents indexed")
            return True
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return False
    
    def _load_agricultural_documents(self):
        """Load documents from various sources"""
        documents = []
        metadata = []
        
        # Load from detailed CSV
        csv_path = self.data_path / "agricultural_bert_detailed.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                documents.append(row['text'])
                metadata.append({
                    'source': 'agricultural_dataset',
                    'category': row.get('label', 'unknown'),
                    'confidence': 1.0
                })
        
        # Load from JSON dataset
        json_path = self.data_path / "agricultural_bert_dataset.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'text' in item:
                            documents.append(item['text'])
                            metadata.append({
                                'source': 'json_dataset',
                                'category': item.get('label', 'unknown'),
                                'confidence': 1.0
                            })
        
        # Add expert knowledge (hardcoded examples)
        expert_knowledge = self._get_expert_knowledge()
        documents.extend(expert_knowledge['texts'])
        metadata.extend(expert_knowledge['metadata'])
        
        self.documents = documents
        self.metadata = metadata
        
    def _get_expert_knowledge(self) -> Dict:
        """Get expert agricultural knowledge"""
        expert_texts = [
            # Bitki Hastalıkları
            "Domates yaprak yanıklığı fungal bir hastalıktır. Alternaria solani mantarı tarafından neden olur. Yapraklarda kahverengi lekeler görülür. Nemli ve sıcak havalarda hızla yayılır. Bakır içerikli fungisitlerle tedavi edilebilir.",
            
            "Buğday pası hastalığı Puccinia tritici mantarı tarafından neden olur. Yapraklarda kırmızımsı kahverengi pustüller oluşur. Rüzgar ile yayılır. Dayanıklı çeşit kullanımı ve fungisit uygulaması etkilidir.",
            
            "Patates yanıklığı Phytophthora infestans mantarı tarafından neden olur. Yapraklarda koyu lekeler, gövdede çürümeler görülür. Nemli havalarda hızla yayılır. Önleyici fungisit uygulaması şarttır.",
            
            # Mahsul Yönetimi
            "Buğday ekimi için ideal toprak pH'ı 6.0-7.5 arasındadır. Ekim derinliği 2-3 cm olmalıdır. Hektara 120-150 kg tohum kullanılır. Azotlu gübre 3 dönemde verilmelidir.",
            
            "Domates fidesi dikimi için toprak sıcaklığı en az 15°C olmalıdır. Fide dikim mesafesi 40x60 cm'dir. Su gereksinimi günde 4-6 litredir. Azot, fosfor, potasyum gübrelemesi önemlidir.",
            
            "Mısır ekimi için toprak sıcaklığı 12°C üzerinde olmalıdır. Ekim derinliği 3-5 cm'dir. Sıra arası 70 cm, sıra üzeri 15-20 cm olmalıdır. Su gereksinimi yüksektir.",
            
            # Çevre Faktörleri
            "Kuraklık stresi bitkilerde stomatal kapanmaya neden olur. Yaprak su potansiyeli düşer. Fotosentez hızı azalır. Erken uyarı sistemleri kullanılmalıdır. Damla sulama sistemi etkilidir.",
            
            "Tuz stresi toprak EC değerinin 4 dS/m üzerine çıkmasıyla başlar. Bitki gelişimi durur. Yaprak kenarlarında yanıklar görülür. Drenaj ve yıkama sulaması gereklidir.",
            
            # Tarım Teknolojisi
            "Akıllı sulama sistemleri toprak nem sensörleri kullanır. Real-time veri toplar. Su tasarrufu %30-50 arası sağlar. IoT teknolojisi ile uzaktan kontrol edilebilir.",
            
            "Drone teknolojisi tarımda harita çıkarma, ilaçlama, monitöring için kullanılır. Multispektral kameralar ile bitki sağlığı analiz edilir. GPS destekli otomatik uçuş sağlar."
        ]
        
        expert_metadata = []
        categories = [
            "plant_disease", "plant_disease", "plant_disease",
            "crop_management", "crop_management", "crop_management", 
            "environmental_factors", "environmental_factors",
            "technology", "technology"
        ]
        
        for i, category in enumerate(categories):
            expert_metadata.append({
                'source': 'expert_knowledge',
                'category': category,
                'confidence': 0.95
            })
        
        return {
            'texts': expert_texts,
            'metadata': expert_metadata
        }
    
    def _create_vector_index(self):
        """Create FAISS vector index"""
        if not self.documents:
            raise ValueError("No documents to index")
        
        # Generate embeddings
        embeddings = self.embeddings_model.encode(self.documents, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.knowledge_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.knowledge_index.add(embeddings.astype('float32'))
        
    def search_knowledge(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search knowledge base for relevant information"""
        if not self.knowledge_index or not self.embeddings_model:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.knowledge_index.search(query_embedding.astype('float32'), top_k)
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append({
                        'text': self.documents[idx],
                        'score': float(score),
                        'metadata': self.metadata[idx]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []

class AgriculturalClassifier:
    """BERT-based agricultural text classifier"""
    
    def __init__(self, model_path: str = "../Model/botanical_bert_small"):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Category mappings
        self.id2label = {
            0: "plant_disease",
            1: "crop_management", 
            2: "plant_genetics",
            3: "environmental_factors",
            4: "food_security",
            5: "technology"
        }
        
        self.label2turkish = {
            "plant_disease": "Bitki Hastalıkları",
            "crop_management": "Mahsul Yönetimi",
            "plant_genetics": "Bitki Genetiği", 
            "environmental_factors": "Çevre Faktörleri",
            "food_security": "Gıda Güvenliği",
            "technology": "Tarım Teknolojisi"
        }
        
    def load_model(self) -> bool:
        """Load BERT classification model"""
        try:
            logger.info(f"Loading BERT model from {self.model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("BERT model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            return False
    
    def classify_text(self, text: str) -> Dict:
        """Classify agricultural text"""
        if not self.pipeline:
            return {"error": "Model not loaded"}
        
        try:
            # Get prediction
            result = self.pipeline(text)[0]
            
            # Parse result
            label = result['label'].replace('LABEL_', '')
            category = self.id2label.get(int(label), "unknown")
            confidence = result['score']
            
            return {
                'category': category,
                'category_turkish': self.label2turkish.get(category, category),
                'confidence': confidence,
                'raw_prediction': result
            }
            
        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            return {"error": str(e)}

class AgriculturalLLMChatbot:
    """Main chatbot class combining classification and knowledge retrieval"""
    
    def __init__(self):
        self.classifier = AgriculturalClassifier()
        self.knowledge_base = AgriculturalKnowledgeBase()
        self.conversation_history = []
        
    def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("Initializing Agricultural LLM Chatbot...")
        
        # Load classifier
        if not self.classifier.load_model():
            logger.error("Failed to load classifier")
            return False
        
        # Load knowledge base  
        if not self.knowledge_base.load_knowledge_base():
            logger.error("Failed to load knowledge base")
            return False
            
        logger.info("Chatbot initialized successfully!")
        return True
    
    def process_query(self, user_input: str, use_history: bool = True) -> Dict:
        """Process user query and generate response"""
        try:
            # Classify query
            classification = self.classifier.classify_text(user_input)
            
            # Search knowledge base
            knowledge_results = self.knowledge_base.search_knowledge(user_input, top_k=3)
            
            # Generate contextual response
            response = self._generate_response(user_input, classification, knowledge_results)
            
            # Update conversation history
            if use_history:
                self.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'user_input': user_input,
                    'classification': classification,
                    'response': response
                })
            
            return {
                'response': response,
                'classification': classification,
                'knowledge_sources': knowledge_results,
                'conversation_id': len(self.conversation_history)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'response': f"Üzgünüm, sorgunuzu işlerken bir hata oluştu: {str(e)}",
                'error': str(e)
            }
    
    def _generate_response(self, query: str, classification: Dict, knowledge: List[Dict]) -> str:
        """Generate contextual response based on classification and knowledge"""
        
        # Get category info
        category = classification.get('category', 'unknown')
        category_tr = classification.get('category_turkish', 'Bilinmeyen')
        confidence = classification.get('confidence', 0.0)
        
        # Start building response
        response_parts = []
        
        # Add classification info if confidence is high
        if confidence > 0.7:
            response_parts.append(f"🔍 **Konu Kategorisi:** {category_tr} (%{confidence*100:.1f} güven)")
        
        # Add relevant knowledge
        if knowledge:
            response_parts.append("\n📚 **İlgili Bilgiler:**")
            
            for i, item in enumerate(knowledge[:2], 1):  # Top 2 results
                score = item['score']
                if score > 0.3:  # Only include relevant results
                    text = item['text'][:300] + "..." if len(item['text']) > 300 else item['text']
                    response_parts.append(f"\n{i}. {text}")
        
        # Add category-specific guidance
        guidance = self._get_category_guidance(category, query)
        if guidance:
            response_parts.append(f"\n💡 **Öneri:** {guidance}")
        
        # Add follow-up questions
        follow_up = self._get_follow_up_questions(category)
        if follow_up:
            response_parts.append(f"\n❓ **Devam eden sorular:** {follow_up}")
        
        # Combine response
        if response_parts:
            return "\n".join(response_parts)
        else:
            return self._get_fallback_response(query)
    
    def _get_category_guidance(self, category: str, query: str) -> str:
        """Get category-specific guidance"""
        guidance_map = {
            "plant_disease": "Hastalık teşhisi için bitki türü, belirtiler ve çevre koşulları önemlidir. Uzman desteği alınması önerilir.",
            "crop_management": "Ekim, gübreleme ve sulama zamanlaması bölgesel koşullara göre değişebilir. Toprak analizi yaptırın.",
            "plant_genetics": "Genetik çeşitlilik ve ıslah çalışmaları uzun vadeli projelerdir. Uzman görüşü alın.",
            "environmental_factors": "Çevresel stres faktörleri erken müdahale ile minimize edilebilir. Monitöring sistemleri kurun.",
            "food_security": "Gıda güvenliği küresel bir konudur. Sürdürülebilir üretim yöntemleri benimseyin.",
            "technology": "Tarım teknolojileri yatırım gerektirir. Maliyet-fayda analizi yapın."
        }
        return guidance_map.get(category, "Detaylı bilgi için uzman desteği alabilirsiniz.")
    
    def _get_follow_up_questions(self, category: str) -> str:
        """Get follow-up questions for each category"""
        questions_map = {
            "plant_disease": "Hangi bitki türü? Belirtiler ne kadar süredir mevcut? İklim koşulları nasıl?",
            "crop_management": "Toprak tipi nedir? Sulama imkanları nasıl? Hangi bölgedesiniz?",
            "plant_genetics": "Hangi özellikler hedefleniyor? Mevcut çeşitler nelerdir?",
            "environmental_factors": "Hangi stres faktörleri gözlemlendi? Ölçüm değerleri nedir?",
            "food_security": "Hangi ürünler için planlama yapılıyor? Hedef pazar nedir?",
            "technology": "Bütçe ne kadar? Mevcut altyapı durumu nasıl?"
        }
        return questions_map.get(category, "Daha spesifik sorularla devam edebiliriz.")
    
    def _get_fallback_response(self, query: str) -> str:
        """Generate fallback response when no good match found"""
        return f"""
🤖 **Tarımsal AI Asistanı**

Sorununuz için spesifik bir bilgi bulamadım, ancak size yardımcı olmaya çalışacağım.

📋 **Ana kategorilerimiz:**
- 🦠 Bitki Hastalıkları
- 🌾 Mahsul Yönetimi  
- 🧬 Bitki Genetiği
- 🌡️ Çevre Faktörleri
- 🍽️ Gıda Güvenliği
- 🚁 Tarım Teknolojisi

💬 **Örnek sorular:**
- "Domates yaprak yanıklığı nedir?"
- "Buğday ekimi nasıl yapılır?"
- "Akıllı sulama sistemi nedir?"

Lütfen daha spesifik bir soru sormayı deneyin!
        """
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

def create_gradio_interface(chatbot: AgriculturalLLMChatbot):
    """Create Gradio web interface"""
    
    def chat_fn(message, history):
        """Gradio chat function"""
        try:
            result = chatbot.process_query(message)
            response = result['response']
            
            # Add classification info to response if available
            classification = result.get('classification', {})
            if 'category_turkish' in classification:
                category = classification['category_turkish']
                confidence = classification.get('confidence', 0)
                response = f"**[{category} - %{confidence*100:.1f}]**\n\n{response}"
            
            return response
            
        except Exception as e:
            return f"Hata: {str(e)}"
    
    # Create interface
    demo = gr.ChatInterface(
        chat_fn,
        title="🌱 Tarımsal AI Chatbot",
        description="""
        **Gelişmiş Tarımsal AI Asistanı**
        
        Bu chatbot aşağıdaki konularda uzmanlaşmıştır:
        - 🦠 Bitki Hastalıkları ve Tedavileri
        - 🌾 Mahsul Yönetimi ve Ekim Teknikleri
        - 🧬 Bitki Genetiği ve Islah
        - 🌡️ Çevresel Faktörler ve Stres
        - 🍽️ Gıda Güvenliği
        - 🚁 Tarım Teknolojileri
        
        **Örnek sorular:**
        - "Domates yaprak yanıklığı belirtileri nelerdir?"
        - "Buğday ekimi için hangi toprak özellikleri gerekli?"
        - "Akıllı sulama sistemleri nasıl çalışır?"
        """,
        examples=[
            "Domates yaprak yanıklığı nedir?",
            "Buğday ekimi nasıl yapılır?",
            "Organik gübre çeşitleri nelerdir?",
            "Akıllı sulama sistemi avantajları",
            "Kuraklık stresi nasıl önlenir?",
            "GMO bitkilerin faydaları nelerdir?"
        ],
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1000px !important;
        }
        .message {
            font-size: 16px !important;
        }
        """,
        retry_btn="🔄 Tekrar Dene",
        undo_btn="↩️ Geri Al", 
        clear_btn="🗑️ Temizle"
    )
    
    return demo

def main():
    """Main function"""
    print("🌱 Agricultural LLM Chatbot System Starting...")
    
    # Initialize chatbot
    chatbot = AgriculturalLLMChatbot()
    
    if not chatbot.initialize():
        print("❌ Failed to initialize chatbot")
        sys.exit(1)
    
    print("✅ Chatbot initialized successfully!")
    
    # Check if running in interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # CLI mode
        print("\n🤖 CLI Chat Mode - Type 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q', 'çıkış']:
                    print("👋 Görüşürüz!")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 Processing...")
                result = chatbot.process_query(user_input)
                print(f"\n🌱 Bot: {result['response']}")
                
            except KeyboardInterrupt:
                print("\n👋 Görüşürüz!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    else:
        # Web interface mode
        print("🌐 Starting web interface...")
        demo = create_gradio_interface(chatbot)
        
        # Launch
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            show_error=True
        )

if __name__ == "__main__":
    main() 