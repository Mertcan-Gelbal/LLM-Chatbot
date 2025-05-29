#!/usr/bin/env python3
"""
Agricultural Test Data Generator
===============================
İndekslenmiş tarımsal chunk verileri kullanarak BERT classification test dataset'i oluşturur
"""

import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
import random

class AgriculturalTestGenerator:
    """Tarımsal test verisi oluşturucu"""
    
    def __init__(self, chunks_file="final_system/complete_index/chunks/all_chunks.json"):
        self.chunks_file = chunks_file
        self.categories = {
            "plant_disease": ["disease", "pathogen", "infection", "symptom", "blight", "spot", "virus", "fungus", "bacteria", "pest", "mold", "rust"],
            "crop_management": ["fertilizer", "irrigation", "planting", "harvest", "yield", "cultivation", "farming", "agriculture", "soil", "nutrient"],
            "plant_genetics": ["variety", "breed", "gene", "trait", "resistance", "chromosome", "mutation", "hybrid", "genetics", "adaptation"],
            "environmental_factors": ["climate", "weather", "temperature", "rainfall", "drought", "flood", "season", "humidity", "wind", "solar"],
            "food_security": ["food", "nutrition", "hunger", "security", "supply", "production", "consumption", "availability", "access", "quality"],
            "technology": ["sensor", "monitoring", "AI", "machine learning", "automation", "robot", "drone", "precision", "digital", "smart"]
        }
        
        print("🌾 Tarımsal Test Verisi Oluşturucu başlatıldı")
        
    def load_chunks(self):
        """Chunk verilerini yükle"""
        print(f"📂 Chunk verileri yükleniyor: {self.chunks_file}")
        
        try:
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            print(f"✅ {len(chunks)} chunk yüklendi")
            return chunks
            
        except Exception as e:
            print(f"❌ Chunks yükleme hatası: {e}")
            return []
    
    def categorize_chunk(self, text):
        """Chunk'ı kategorilere ayır"""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.categories.items():
            score = 0
            for keyword in keywords:
                # Kelime sayısı (keyword matching)
                score += len(re.findall(r'\b' + keyword + r'\b', text_lower))
                # Keyword variants
                score += len(re.findall(keyword, text_lower)) * 0.5
            scores[category] = score
        
        # En yüksek skor alan kategori
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return "general_agriculture"
    
    def clean_text(self, text):
        """Metni temizle"""
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text)
        # Özel karakterleri temizle (bazıları)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        # Çok kısa metinleri filtrele
        if len(text.split()) < 10:
            return None
        # Çok uzun metinleri kısalt
        words = text.split()
        if len(words) > 200:
            text = ' '.join(words[:200]) + "..."
        
        return text.strip()
    
    def extract_agricultural_segments(self, text, max_segments=3):
        """Metinden tarımsal segmentler çıkar"""
        sentences = re.split(r'[.!?]+', text)
        agricultural_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < 5:  # Çok kısa cümleler
                continue
                
            # Tarımsal kelime kontrolü
            agri_words = 0
            for category, keywords in self.categories.items():
                for keyword in keywords:
                    if keyword in sentence.lower():
                        agri_words += 1
            
            if agri_words >= 1:  # En az 1 tarımsal kelime
                agricultural_sentences.append(sentence)
        
        # En fazla max_segments segment al
        if len(agricultural_sentences) > max_segments:
            agricultural_sentences = random.sample(agricultural_sentences, max_segments)
        
        return agricultural_sentences
    
    def generate_test_dataset(self, target_samples=2000, min_per_category=200):
        """Test dataset'i oluştur"""
        print("🔄 Test dataset'i oluşturuluyor...")
        
        chunks = self.load_chunks()
        if not chunks:
            return None
        
        # Kategorilere göre veri toplama
        categorized_data = defaultdict(list)
        
        print("📊 Chunk'ları kategorize ediliyor...")
        for i, chunk in enumerate(chunks):
            if i % 1000 == 0:
                print(f"  İşlenen: {i}/{len(chunks)}")
            
            text = chunk.get('text', '')
            if len(text.split()) < 10:  # Çok kısa chunks
                continue
            
            # Tarımsal segmentler çıkar
            segments = self.extract_agricultural_segments(text)
            
            for segment in segments:
                cleaned = self.clean_text(segment)
                if cleaned:
                    category = self.categorize_chunk(cleaned)
                    categorized_data[category].append({
                        'text': cleaned,
                        'label': category,
                        'source': chunk.get('source', 'unknown'),
                        'filename': chunk.get('filename', 'unknown')
                    })
        
        print(f"\n📈 Kategori dağılımı:")
        for category, items in categorized_data.items():
            print(f"  {category}: {len(items)} sample")
        
        # Balanced dataset oluştur
        balanced_data = []
        
        for category, items in categorized_data.items():
            if len(items) >= min_per_category:
                # Random sample al
                selected = random.sample(items, min(len(items), min_per_category))
                balanced_data.extend(selected)
                print(f"✅ {category}: {len(selected)} sample alındı")
            else:
                # Tüm veriyi al
                balanced_data.extend(items)
                print(f"⚠️ {category}: Sadece {len(items)} sample (minimum {min_per_category})")
        
        # Shuffle
        random.shuffle(balanced_data)
        
        # Target sample size'a göre ayarla
        if len(balanced_data) > target_samples:
            balanced_data = balanced_data[:target_samples]
        
        print(f"\n🎯 Final dataset: {len(balanced_data)} samples")
        
        return balanced_data
    
    def save_datasets(self, data, output_dir="agricultural_datasets"):
        """Dataset'leri kaydet"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # DataFrame oluştur
        df = pd.DataFrame(data)
        
        # Ana dataset
        main_file = output_path / "agricultural_bert_dataset.csv"
        df[['text', 'label']].to_csv(main_file, index=False, encoding='utf-8')
        print(f"📁 Ana dataset kaydedildi: {main_file}")
        
        # Metadata ile birlikte
        detailed_file = output_path / "agricultural_bert_detailed.csv"
        df.to_csv(detailed_file, index=False, encoding='utf-8')
        print(f"📁 Detaylı dataset kaydedildi: {detailed_file}")
        
        # JSON format
        json_file = output_path / "agricultural_bert_dataset.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"📁 JSON dataset kaydedildi: {json_file}")
        
        # Category statistics
        stats = df['label'].value_counts().to_dict()
        stats_file = output_path / "category_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"📁 İstatistikler kaydedildi: {stats_file}")
        
        # Train/Val/Test split
        self.create_splits(df, output_path)
        
        return main_file
    
    def create_splits(self, df, output_path):
        """Train/Validation/Test split'leri oluştur"""
        print("🔄 Train/Val/Test split'leri oluşturuluyor...")
        
        # Stratified split için kategorilere göre ayır
        train_data, val_data, test_data = [], [], []
        
        for label in df['label'].unique():
            label_data = df[df['label'] == label].copy()
            n = len(label_data)
            
            # 70% train, 15% val, 15% test
            train_size = int(n * 0.7)
            val_size = int(n * 0.15)
            
            # Shuffle
            label_data = label_data.sample(frac=1).reset_index(drop=True)
            
            train_data.append(label_data[:train_size])
            val_data.append(label_data[train_size:train_size + val_size])
            test_data.append(label_data[train_size + val_size:])
        
        # Concat and shuffle
        train_df = pd.concat(train_data).sample(frac=1).reset_index(drop=True)
        val_df = pd.concat(val_data).sample(frac=1).reset_index(drop=True)
        test_df = pd.concat(test_data).sample(frac=1).reset_index(drop=True)
        
        # Save splits
        train_df[['text', 'label']].to_csv(output_path / "train.csv", index=False, encoding='utf-8')
        val_df[['text', 'label']].to_csv(output_path / "val.csv", index=False, encoding='utf-8')
        test_df[['text', 'label']].to_csv(output_path / "test.csv", index=False, encoding='utf-8')
        
        print(f"📊 Split sizes:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
    
    def create_simple_sentiment_dataset(self, output_dir="agricultural_datasets"):
        """Basit sentiment analizi için dataset oluştur"""
        print("🔄 Basit sentiment dataset'i oluşturuluyor...")
        
        chunks = self.load_chunks()
        if not chunks:
            return None
        
        sentiment_data = []
        
        # Pozitif ve negatif belirten kelimeler
        positive_words = ["good", "excellent", "effective", "improved", "successful", "high yield", "healthy", "resistant", "quality", "beneficial", "optimize", "enhance"]
        negative_words = ["disease", "damage", "poor", "failed", "infected", "pest", "problem", "loss", "decline", "risk", "stress", "deficiency"]
        
        for chunk in chunks[:3000]:  # İlk 3000 chunk
            text = chunk.get('text', '')
            segments = self.extract_agricultural_segments(text, max_segments=2)
            
            for segment in segments:
                cleaned = self.clean_text(segment)
                if not cleaned:
                    continue
                
                # Sentiment skorlama
                text_lower = cleaned.lower()
                pos_score = sum(1 for word in positive_words if word in text_lower)
                neg_score = sum(1 for word in negative_words if word in text_lower)
                
                # Label belirleme
                if pos_score > neg_score and pos_score > 0:
                    label = "positive"
                elif neg_score > pos_score and neg_score > 0:
                    label = "negative"
                else:
                    label = "neutral"
                
                sentiment_data.append({
                    'text': cleaned,
                    'label': label
                })
        
        # Balance the dataset
        label_counts = {}
        for item in sentiment_data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"📊 Sentiment dağılımı: {label_counts}")
        
        # Balanced subset
        min_count = min(label_counts.values())
        balanced_sentiment = []
        counts = {label: 0 for label in label_counts.keys()}
        
        random.shuffle(sentiment_data)
        for item in sentiment_data:
            label = item['label']
            if counts[label] < min_count:
                balanced_sentiment.append(item)
                counts[label] += 1
        
        # Save sentiment dataset
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        sentiment_df = pd.DataFrame(balanced_sentiment)
        sentiment_file = output_path / "agricultural_sentiment.csv"
        sentiment_df.to_csv(sentiment_file, index=False, encoding='utf-8')
        
        print(f"📁 Sentiment dataset kaydedildi: {sentiment_file}")
        print(f"🎯 Final sentiment dataset: {len(balanced_sentiment)} samples")
        
        return sentiment_file

def main():
    """Ana fonksiyon"""
    print("🌾 Agricultural BERT Test Data Generator")
    print("=" * 50)
    
    generator = AgriculturalTestGenerator()
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    try:
        # 1. Ana kategorik dataset
        print("\n1️⃣ Ana Kategorik Dataset Oluşturuluyor...")
        data = generator.generate_test_dataset(target_samples=3000, min_per_category=300)
        
        if data:
            main_file = generator.save_datasets(data)
            print(f"✅ Ana dataset hazır: {main_file}")
        
        # 2. Sentiment dataset
        print("\n2️⃣ Sentiment Dataset Oluşturuluyor...")
        sentiment_file = generator.create_simple_sentiment_dataset()
        
        if sentiment_file:
            print(f"✅ Sentiment dataset hazır: {sentiment_file}")
        
        print("\n🎉 Tüm dataset'ler başarıyla oluşturuldu!")
        print("\n📁 Oluşturulan dosyalar:")
        print("  - agricultural_datasets/agricultural_bert_dataset.csv")
        print("  - agricultural_datasets/train.csv")
        print("  - agricultural_datasets/val.csv") 
        print("  - agricultural_datasets/test.csv")
        print("  - agricultural_datasets/agricultural_sentiment.csv")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 