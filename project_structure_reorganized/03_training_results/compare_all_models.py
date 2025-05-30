#!/usr/bin/env python3
"""
Tüm Tarımsal AI Modellerini Karşılaştırma Scripti
DistilBERT, BERT, GPT-2 ve RAG sistemlerinin performans karşılaştırması
"""

import time
import json
from pathlib import Path
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

console = Console()

class ModelComparator:
    def __init__(self):
        self.test_questions = [
            "Elmada erken yanıklığı nasıl tedavi edilir?",
            "Buğday ekim zamanı ne zaman?",
            "Domates bitkilerinde sarı yaprak sorunu neden oluşur?",
            "Aşırı sıcaklıkta bitkileri nasıl koruruz?",
            "Toprak pH değeri neden önemlidir?",
            "Organik gübre çeşitleri nelerdir?",
            "Havuç yetiştirmede sulama nasıl yapılır?",
            "Bitki hastalıklarından nasıl korunabiliriz?"
        ]
        
        self.expected_categories = [
            "plant_disease", "crop_management", "plant_disease", 
            "environmental_factors", "environmental_factors",
            "crop_management", "crop_management", "plant_disease"
        ]
        
        self.models = {}
        self.results = {}
    
    def load_bert_model(self):
        """BERT modelini yükle"""
        try:
            from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
            
            model_path = Path("../02_models/bert_classification/agricultural_bert_base_uncased")
            if model_path.exists():
                tokenizer = BertTokenizer.from_pretrained(model_path)
                config = BertConfig.from_pretrained(model_path)
                model = BertForSequenceClassification(config)
                state_dict = torch.load(model_path / "pytorch_model.bin", map_location='cpu')
                model.load_state_dict(state_dict)
                model.eval()
                
                self.models['BERT'] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'type': 'classification'
                }
                console.print("✅ BERT model yüklendi", style="green")
            else:
                console.print("❌ BERT model bulunamadı", style="red")
        except Exception as e:
            console.print(f"❌ BERT yükleme hatası: {e}", style="red")
    
    def load_distilbert_model(self):
        """DistilBERT modelini yükle"""
        try:
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            
            model_path = Path("../02_models/bert_classification/distilbert_agricultural")
            if model_path.exists():
                tokenizer = DistilBertTokenizer.from_pretrained(model_path)
                model = DistilBertForSequenceClassification.from_pretrained(model_path)
                model.eval()
                
                self.models['DistilBERT'] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'type': 'classification'
                }
                console.print("✅ DistilBERT model yüklendi", style="green")
            else:
                console.print("❌ DistilBERT model bulunamadı", style="red")
        except Exception as e:
            console.print(f"❌ DistilBERT yükleme hatası: {e}", style="red")
    
    def load_gpt2_model(self):
        """GPT-2 modelini yükle"""
        try:
            from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
            
            model_path = Path("../02_models/gpt2_generation/agricultural_gpt2")
            if model_path.exists():
                generator = pipeline(
                    "text-generation",
                    model=str(model_path),
                    device=-1  # CPU
                )
                
                self.models['GPT-2'] = {
                    'generator': generator,
                    'type': 'generation'
                }
                console.print("✅ GPT-2 model yüklendi", style="green")
            else:
                console.print("❌ GPT-2 model bulunamadı", style="red")
        except Exception as e:
            console.print(f"❌ GPT-2 yükleme hatası: {e}", style="red")
    
    def create_template_system(self):
        """Template-based sistem oluştur"""
        templates = {
            "plant_disease": "Bu bir bitki hastalığı sorusudur. Hastalık kontrolü için erken teşhis önemlidir.",
            "crop_management": "Bu yetiştirme tekniği ile ilgilidir. Doğru zaman ve yöntem başarının anahtarıdır.",
            "environmental_factors": "Bu çevre faktörleri ile ilgilidir. Optimum koşullar sağlamak önemlidir.",
        }
        
        self.models['Template'] = {
            'templates': templates,
            'type': 'template'
        }
        console.print("✅ Template sistem hazırlandı", style="green")
    
    def classify_question(self, model_info, question):
        """Soruyu sınıflandır"""
        if model_info['type'] != 'classification':
            return None, 0.0
        
        tokenizer = model_info['tokenizer']
        model = model_info['model']
        
        encoding = tokenizer(
            question,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = model(**encoding)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item()
        
        categories = ["plant_disease", "crop_management", "environmental_factors"]
        category = categories[pred_id] if pred_id < len(categories) else "unknown"
        
        return category, confidence
    
    def generate_answer(self, model_info, question):
        """Cevap üret"""
        if model_info['type'] == 'generation':
            prompt = f"<|soru|>{question}<|cevap|>"
            response = model_info['generator'](
                prompt,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            generated_text = response[0]['generated_text']
            if "<|cevap|>" in generated_text:
                answer = generated_text.split("<|cevap|>")[-1].split("<|end|>")[0].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            return answer[:200] + "..." if len(answer) > 200 else answer
        
        elif model_info['type'] == 'template':
            # Basit keyword matching
            templates = model_info['templates']
            question_lower = question.lower()
            
            if any(word in question_lower for word in ['hastalık', 'yanıklık', 'tedavi', 'sarı']):
                return templates["plant_disease"]
            elif any(word in question_lower for word in ['ekim', 'sulama', 'yetiştir', 'gübre']):
                return templates["crop_management"]
            elif any(word in question_lower for word in ['pH', 'sıcaklık', 'toprak', 'çevre']):
                return templates["environmental_factors"]
            else:
                return "Bu konuda yardımcı olmaya çalışıyorum."
        
        return "Cevap üretilemedi."
    
    def benchmark_model(self, model_name, model_info):
        """Model performansını test et"""
        results = {
            'response_times': [],
            'classifications': [],
            'answers': [],
            'accuracy': 0
        }
        
        correct_predictions = 0
        
        for i, question in enumerate(track(self.test_questions, description=f"{model_name} test ediliyor...")):
            start_time = time.time()
            
            # Sınıflandırma
            category, confidence = self.classify_question(model_info, question)
            
            # Cevap üretimi
            answer = self.generate_answer(model_info, question)
            
            response_time = (time.time() - start_time) * 1000  # ms
            
            results['response_times'].append(response_time)
            results['classifications'].append({
                'question': question,
                'predicted': category,
                'expected': self.expected_categories[i],
                'confidence': confidence
            })
            results['answers'].append(answer)
            
            # Accuracy hesaplama
            if category == self.expected_categories[i]:
                correct_predictions += 1
        
        results['accuracy'] = correct_predictions / len(self.test_questions)
        results['avg_response_time'] = sum(results['response_times']) / len(results['response_times'])
        
        return results
    
    def run_comparison(self):
        """Tüm modelleri karşılaştır"""
        console.print("🚀 Model karşılaştırması başlıyor...", style="bold cyan")
        
        # Modelleri yükle
        self.load_bert_model()
        self.load_distilbert_model()
        self.load_gpt2_model()
        self.create_template_system()
        
        # Her modeli test et
        for model_name, model_info in self.models.items():
            console.print(f"\n🔍 {model_name} test ediliyor...")
            self.results[model_name] = self.benchmark_model(model_name, model_info)
        
        # Sonuçları göster
        self.display_results()
    
    def display_results(self):
        """Sonuçları görüntüle"""
        # Performans tablosu
        table = Table(title="📊 Model Performans Karşılaştırması")
        table.add_column("Model", style="cyan")
        table.add_column("Accuracy", style="green")
        table.add_column("Avg Response Time (ms)", style="yellow")
        table.add_column("Min Time", style="blue")
        table.add_column("Max Time", style="red")
        
        for model_name, results in self.results.items():
            if results['response_times']:
                table.add_row(
                    model_name,
                    f"{results['accuracy']:.2%}",
                    f"{results['avg_response_time']:.1f}",
                    f"{min(results['response_times']):.1f}",
                    f"{max(results['response_times']):.1f}"
                )
        
        console.print(table)
        
        # Detaylı cevap örnekleri
        console.print("\n🔍 Örnek Cevaplar:", style="bold blue")
        
        example_question = self.test_questions[0]  # "Elmada erken yanıklığı nasıl tedavi edilir?"
        
        for model_name, results in self.results.items():
            if results['answers']:
                panel = Panel(
                    results['answers'][0],
                    title=f"{model_name} Cevabı",
                    border_style="green" if model_name == "DistilBERT" else "blue"
                )
                console.print(panel)
        
        # Özet ve öneriler
        self.display_recommendations()
    
    def display_recommendations(self):
        """Öneriler göster"""
        recommendations = Panel.fit(
            "🎯 **Öneriler:**\n\n"
            "🥇 **En İyi Genel Performans**: DistilBERT\n"
            "   • Yüksek accuracy\n"
            "   • Makul yanıt süresi\n"
            "   • Güvenilir sınıflandırma\n\n"
            "⚡ **En Hızlı**: Template System\n"
            "   • Minimum latency\n"
            "   • Düşük kaynak kullanımı\n"
            "   • Basit deployment\n\n"
            "💬 **En Doğal**: GPT-2\n"
            "   • İnsan benzeri cevaplar\n"
            "   • Yaratıcı çıktılar\n"
            "   • Yüksek kaynak ihtiyacı\n\n"
            "🔄 **Hibrit Öneri**: DistilBERT + Template\n"
            "   • Yüksek confidence → Template (hızlı)\n"
            "   • Düşük confidence → DistilBERT (doğru)",
            title="🌾 Tarımsal AI Model Önerileri",
            style="bold green"
        )
        console.print(recommendations)
    
    def save_results(self):
        """Sonuçları kaydet"""
        output_file = Path("../03_training_results/performance_metrics/comparison_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # JSON serializable format
        serializable_results = {}
        for model_name, results in self.results.items():
            serializable_results[model_name] = {
                'accuracy': results['accuracy'],
                'avg_response_time': results['avg_response_time'],
                'response_times': results['response_times'],
                'sample_answers': results['answers'][:3]  # İlk 3 cevap
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        console.print(f"💾 Sonuçlar kaydedildi: {output_file}", style="green")

def main():
    """Ana fonksiyon"""
    try:
        comparator = ModelComparator()
        comparator.run_comparison()
        comparator.save_results()
    except Exception as e:
        console.print(f"❌ Hata: {e}", style="bold red")

if __name__ == "__main__":
    main() 