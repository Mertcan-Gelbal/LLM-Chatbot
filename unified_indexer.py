#!/usr/bin/env python3
"""
🌾 Unified Agricultural Data Indexer
Tarımsal verileri indeksleyen ve kategorize eden sistem
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

class UnifiedIndexer:
    """
    Tarımsal verileri indeksleyen unified sistem
    """
    
    def __init__(self, data_dir: str = "final_system/complete_index"):
        self.data_dir = Path(data_dir)
        self.setup_logging()
        
        # Agricultural categories
        self.categories = {
            "Plant Disease": ["disease", "pathogen", "infection", "blight", "virus", "fungus", "bacteria"],
            "Crop Management": ["fertilizer", "irrigation", "planting", "harvest", "cultivation", "farming"],
            "Plant Genetics": ["variety", "gene", "resistance", "hybrid", "breeding", "genetics"],
            "Environmental Factors": ["climate", "weather", "drought", "temperature", "rainfall", "soil"],
            "Food Security": ["nutrition", "supply", "production", "access", "food", "security"],
            "Technology": ["AI", "sensor", "automation", "precision", "technology", "digital"]
        }
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def categorize_text(self, text: str) -> str:
        """Metni kategorize et"""
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
            
        # En yüksek skora sahip kategoriyi döndür
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        else:
            return "Technology"  # Default category
            
    def load_indexed_data(self) -> List[Dict[str, Any]]:
        """İndekslenmiş verileri yükle"""
        indexed_data = []
        
        if not self.data_dir.exists():
            self.logger.warning(f"Data directory {self.data_dir} not found")
            return indexed_data
            
        # JSON dosyalarını yükle
        for json_file in self.data_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        indexed_data.extend(data)
                    else:
                        indexed_data.append(data)
                        
                self.logger.info(f"Loaded {json_file.name}")
                
            except Exception as e:
                self.logger.error(f"Error loading {json_file}: {e}")
                
        self.logger.info(f"Total indexed chunks: {len(indexed_data)}")
        return indexed_data
        
    def process_and_categorize(self) -> pd.DataFrame:
        """Verileri işle ve kategorize et"""
        indexed_data = self.load_indexed_data()
        
        processed_data = []
        
        for item in indexed_data:
            # Text field'ı bul
            text = ""
            if isinstance(item, dict):
                text = item.get('content', item.get('text', str(item)))
            else:
                text = str(item)
                
            if text and len(text.strip()) > 10:  # Minimum text length
                category = self.categorize_text(text)
                
                processed_data.append({
                    'text': text.strip(),
                    'category': category,
                    'length': len(text),
                    'timestamp': datetime.now().isoformat()
                })
                
        df = pd.DataFrame(processed_data)
        
        # Category distribution
        if not df.empty:
            self.logger.info("Category distribution:")
            for category, count in df['category'].value_counts().items():
                self.logger.info(f"  {category}: {count}")
                
        return df
        
    def save_processed_data(self, df: pd.DataFrame, output_file: str = "processed_agricultural_data.csv"):
        """İşlenmiş verileri kaydet"""
        output_path = Path(output_file)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Processed data saved to {output_path}")
        
        return output_path

def main():
    """Main function"""
    print("🌾 Unified Agricultural Data Indexer")
    print("=" * 40)
    
    indexer = UnifiedIndexer()
    
    # Process and categorize data
    df = indexer.process_and_categorize()
    
    if not df.empty:
        # Save processed data
        indexer.save_processed_data(df)
        
        print(f"\n📊 Processing Summary:")
        print(f"Total chunks: {len(df)}")
        print(f"Categories: {df['category'].nunique()}")
        print(f"Average text length: {df['length'].mean():.1f}")
        
    else:
        print("⚠️  No data found to process")

if __name__ == "__main__":
    main() 