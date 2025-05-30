#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌱 Agricultural LLM API Server
REST API server for agricultural chatbot

Endpoints:
- POST /chat - Chat with the bot
- POST /classify - Classify text only  
- GET /health - Health check
- GET /categories - Get available categories
- GET /stats - Get usage statistics

Usage:
    python agricultural_api_server.py
    curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d '{"message": "Domates hastalığı"}'
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS
except ImportError:
    print("❌ Flask not installed. Please run: pip install flask flask-cors")
    sys.exit(1)

from Model.run_model import predict_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatRequest:
    """Chat request data structure"""
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    include_knowledge: bool = True

@dataclass
class ChatResponse:
    """Chat response data structure"""
    response: str
    category: str
    category_turkish: str
    confidence: float
    timestamp: str
    processing_time_ms: float
    session_id: Optional[str] = None

class AgriculturalAPIServer:
    """Flask-based API server for agricultural chatbot"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for web requests
        
        self.host = host
        self.port = port
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'start_time': datetime.now().isoformat(),
            'categories_count': {
                'plant_disease': 0,
                'crop_management': 0,
                'plant_genetics': 0,
                'environmental_factors': 0,
                'food_security': 0,
                'technology': 0
            }
        }
        
        self.categories = {
            "plant_disease": "🦠 Bitki Hastalıkları",
            "crop_management": "🌾 Mahsul Yönetimi",
            "plant_genetics": "🧬 Bitki Genetiği",
            "environmental_factors": "🌡️ Çevre Faktörleri",
            "food_security": "🍽️ Gıda Güvenliği",
            "technology": "🚁 Tarım Teknolojisi"
        }
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/', methods=['GET'])
        def home():
            """Home page with API documentation"""
            return render_template_string(HOME_TEMPLATE, 
                                        base_url=f"http://{self.host}:{self.port}")
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            try:
                # Test model prediction
                test_result = predict_text("test")
                model_status = "healthy" if 'error' not in test_result else "error"
                
                return jsonify({
                    'status': 'healthy',
                    'model_status': model_status,
                    'timestamp': datetime.now().isoformat(),
                    'uptime_seconds': (datetime.now() - datetime.fromisoformat(self.stats['start_time'])).total_seconds()
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/categories', methods=['GET'])
        def get_categories():
            """Get available categories"""
            return jsonify({
                'categories': self.categories,
                'count': len(self.categories)
            })
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """Get usage statistics"""
            current_stats = self.stats.copy()
            current_stats['uptime_seconds'] = (datetime.now() - datetime.fromisoformat(self.stats['start_time'])).total_seconds()
            return jsonify(current_stats)
        
        @self.app.route('/classify', methods=['POST'])
        def classify_text():
            """Classify text endpoint"""
            start_time = time.time()
            self.stats['total_requests'] += 1
            
            try:
                # Parse request
                data = request.get_json()
                if not data or 'text' not in data:
                    self.stats['failed_requests'] += 1
                    return jsonify({'error': 'Missing text field'}), 400
                
                text = data['text'].strip()
                if not text:
                    self.stats['failed_requests'] += 1
                    return jsonify({'error': 'Empty text'}), 400
                
                # Classify text
                result = predict_text(text)
                
                if 'error' in result:
                    self.stats['failed_requests'] += 1
                    return jsonify({'error': result['error']}), 500
                
                # Update stats
                category = result.get('category', 'unknown')
                if category in self.stats['categories_count']:
                    self.stats['categories_count'][category] += 1
                
                self.stats['successful_requests'] += 1
                
                # Prepare response
                processing_time = (time.time() - start_time) * 1000
                
                response = {
                    'category': category,
                    'category_turkish': result.get('category_turkish', 'Bilinmeyen'),
                    'confidence': result.get('confidence', 0.0),
                    'timestamp': datetime.now().isoformat(),
                    'processing_time_ms': round(processing_time, 2)
                }
                
                return jsonify(response)
                
            except Exception as e:
                self.stats['failed_requests'] += 1
                logger.error(f"Error in classify endpoint: {e}")
                return jsonify({'error': f'Internal error: {str(e)}'}), 500
        
        @self.app.route('/chat', methods=['POST'])
        def chat():
            """Main chat endpoint"""
            start_time = time.time()
            self.stats['total_requests'] += 1
            
            try:
                # Parse request
                data = request.get_json()
                if not data or 'message' not in data:
                    self.stats['failed_requests'] += 1
                    return jsonify({'error': 'Missing message field'}), 400
                
                message = data['message'].strip()
                if not message:
                    self.stats['failed_requests'] += 1
                    return jsonify({'error': 'Empty message'}), 400
                
                user_id = data.get('user_id')
                session_id = data.get('session_id')
                include_knowledge = data.get('include_knowledge', True)
                
                # Classify and generate response
                result = predict_text(message)
                
                if 'error' in result:
                    self.stats['failed_requests'] += 1
                    return jsonify({'error': result['error']}), 500
                
                # Generate enhanced response
                enhanced_response = self._generate_enhanced_response(message, result)
                
                # Update stats
                category = result.get('category', 'unknown')
                if category in self.stats['categories_count']:
                    self.stats['categories_count'][category] += 1
                
                self.stats['successful_requests'] += 1
                
                # Prepare response
                processing_time = (time.time() - start_time) * 1000
                
                chat_response = ChatResponse(
                    response=enhanced_response,
                    category=category,
                    category_turkish=result.get('category_turkish', 'Bilinmeyen'),
                    confidence=result.get('confidence', 0.0),
                    timestamp=datetime.now().isoformat(),
                    processing_time_ms=round(processing_time, 2),
                    session_id=session_id
                )
                
                return jsonify(asdict(chat_response))
                
            except Exception as e:
                self.stats['failed_requests'] += 1
                logger.error(f"Error in chat endpoint: {e}")
                return jsonify({'error': f'Internal error: {str(e)}'}), 500
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Endpoint not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
    
    def _generate_enhanced_response(self, message: str, prediction: Dict) -> str:
        """Generate enhanced response with context"""
        category = prediction.get('category', 'unknown')
        category_tr = prediction.get('category_turkish', 'Bilinmeyen')
        confidence = prediction.get('confidence', 0.0)
        
        # Start with category info
        response_parts = []
        
        if confidence > 0.7:
            response_parts.append(f"🔍 **Konu Kategorisi:** {category_tr} (%{confidence*100:.1f} güven)")
        
        # Add category-specific guidance
        guidance = self._get_category_guidance(category, message)
        if guidance:
            response_parts.append(f"\n💡 **Öneri:** {guidance}")
        
        # Add practical advice
        practical_advice = self._get_practical_advice(category)
        if practical_advice:
            response_parts.append(f"\n🔧 **Pratik Bilgi:** {practical_advice}")
        
        # Add follow-up questions
        follow_up = self._get_follow_up_questions(category)
        if follow_up:
            response_parts.append(f"\n❓ **İlgili Sorular:** {follow_up}")
        
        if response_parts:
            return "\n".join(response_parts)
        else:
            return f"Bu konu {category_tr} kategorisinde sınıflandırıldı. Daha detaylı bilgi için lütfen spesifik sorularınızı sorun."
    
    def _get_category_guidance(self, category: str, query: str) -> str:
        """Get category-specific guidance"""
        guidance_map = {
            "plant_disease": "Hastalık teşhisi için bitki türü, belirtiler ve çevre koşulları önemlidir. Erken teşhis kritiktir.",
            "crop_management": "Ekim, gübreleme ve sulama zamanlaması bölgesel koşullara göre optimize edilmelidir.",
            "plant_genetics": "Genetik çeşitlilik ve ıslah çalışmaları uzun vadeli yaklaşım gerektirir.",
            "environmental_factors": "Çevresel stres faktörleri proaktif yönetim ile minimize edilebilir.",
            "food_security": "Gıda güvenliği için sürdürülebilir üretim ve hasat sonrası yönetim kritiktir.",
            "technology": "Tarım teknolojileri ROI analizi ile değerlendirilmeli ve kademeli uygulanmalıdır."
        }
        return guidance_map.get(category, "Uzman desteği alınması önerilir.")
    
    def _get_practical_advice(self, category: str) -> str:
        """Get practical advice for each category"""
        advice_map = {
            "plant_disease": "İlaçlama öncesi mutlaka etken tespiti yapın. Önleyici tedbirleri ihmal etmeyin.",
            "crop_management": "Toprak analizi yaptırın ve bölgesel iklim verilerini takip edin.",
            "plant_genetics": "Yerel çeşitlerle başlayıp kademeli olarak gelişmiş çeşitlere geçin.",
            "environmental_factors": "Mikroklima oluşturun ve stres belirtilerini erken tespit edin.",
            "food_security": "Depolama koşullarını optimize edin ve kayıp oranlarını izleyin.",
            "technology": "Pilot uygulama ile başlayın, başarılı olursa ölçeklendirin."
        }
        return advice_map.get(category, "")
    
    def _get_follow_up_questions(self, category: str) -> str:
        """Get follow-up questions"""
        questions_map = {
            "plant_disease": "Hangi bitki? Belirtiler ne kadar süredir var? İklim nasıl?",
            "crop_management": "Toprak tipi? Sulama imkanları? Hangi bölge?",
            "plant_genetics": "Hedef özellikler? Mevcut çeşitler? Bütçe?",
            "environmental_factors": "Hangi stres faktörleri? Ölçüm değerleri?",
            "food_security": "Hangi ürünler? Hedef pazar? Kapasite?",
            "technology": "Mevcut altyapı? Bütçe? Teknik ekip?"
        }
        return questions_map.get(category, "Daha spesifik sorularla devam edebiliriz.")
    
    def run(self, debug: bool = False):
        """Run the Flask server"""
        logger.info(f"🌱 Agricultural API Server starting on {self.host}:{self.port}")
        logger.info(f"📋 Available endpoints:")
        logger.info(f"   • GET  / - API documentation")
        logger.info(f"   • POST /chat - Chat endpoint")
        logger.info(f"   • POST /classify - Classification endpoint")
        logger.info(f"   • GET  /health - Health check")
        logger.info(f"   • GET  /categories - Available categories")
        logger.info(f"   • GET  /stats - Usage statistics")
        
        self.app.run(
            host=self.host,
            port=self.port,
            debug=debug,
            threaded=True
        )

# HTML template for home page
HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🌱 Agricultural LLM API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background: #2d5a27; color: white; padding: 20px; border-radius: 8px; }
        .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .method { background: #007acc; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
        .example { background: #f9f9f9; border-left: 4px solid #007acc; padding: 10px; margin: 10px 0; }
        code { background: #eee; padding: 2px 4px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌱 Agricultural LLM API Server</h1>
        <p>Tarımsal yapay zeka chatbot API'si</p>
    </div>
    
    <h2>📋 API Endpoints</h2>
    
    <div class="endpoint">
        <h3><span class="method">POST</span> /chat</h3>
        <p>Ana chatbot endpoint'i - gelişmiş cevaplar</p>
        <div class="example">
            <strong>Request:</strong><br>
            <code>POST {{ base_url }}/chat</code><br>
            <code>{"message": "Domates yaprak yanıklığı nedir?"}</code>
        </div>
    </div>
    
    <div class="endpoint">
        <h3><span class="method">POST</span> /classify</h3>
        <p>Sadece metin sınıflandırması</p>
        <div class="example">
            <strong>Request:</strong><br>
            <code>POST {{ base_url }}/classify</code><br>
            <code>{"text": "Buğday ekimi"}</code>
        </div>
    </div>
    
    <div class="endpoint">
        <h3><span class="method">GET</span> /health</h3>
        <p>Sistem sağlık kontrolü</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method">GET</span> /categories</h3>
        <p>Mevcut kategoriler listesi</p>
    </div>
    
    <div class="endpoint">
        <h3><span class="method">GET</span> /stats</h3>
        <p>Kullanım istatistikleri</p>
    </div>
    
    <h2>💬 Test Examples</h2>
    
    <div class="example">
        <strong>cURL Examples:</strong><br><br>
        
        <strong>Chat:</strong><br>
        <code>curl -X POST {{ base_url }}/chat -H "Content-Type: application/json" -d '{"message": "Domates hastalığı nasıl tedavi edilir?"}'</code><br><br>
        
        <strong>Classify:</strong><br>
        <code>curl -X POST {{ base_url }}/classify -H "Content-Type: application/json" -d '{"text": "Buğday ekimi için en uygun zaman"}'</code><br><br>
        
        <strong>Health:</strong><br>
        <code>curl {{ base_url }}/health</code>
    </div>
    
    <h2>🔧 Kategoriler</h2>
    <ul>
        <li>🦠 Bitki Hastalıkları (plant_disease)</li>
        <li>🌾 Mahsul Yönetimi (crop_management)</li>
        <li>🧬 Bitki Genetiği (plant_genetics)</li>
        <li>🌡️ Çevre Faktörleri (environmental_factors)</li>
        <li>🍽️ Gıda Güvenliği (food_security)</li>
        <li>🚁 Tarım Teknolojisi (technology)</li>
    </ul>
    
    <p><em>API Server - Botanical BERT Team © 2024</em></p>
</body>
</html>
"""

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Agricultural LLM API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Create and run server
    server = AgriculturalAPIServer(host=args.host, port=args.port)
    server.run(debug=args.debug)

if __name__ == "__main__":
    main() 