#!/usr/bin/env python3
"""
🌱 Botanical BERT Model Eğitimi
Basit ve temiz model eğitimi scripti
"""

import os
import sys
import time
import warnings
import pandas as pd
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer, BertConfig
)
from transformers import DataCollatorWithPadding
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class BotanicalBERTTrainer:
    """Botanical BERT Model Eğitici"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Kategori mappings
        self.label2id = {
            'plant_disease': 0,
            'crop_management': 1, 
            'plant_genetics': 2,
            'environmental_factors': 3,
            'food_security': 4,
            'technology': 5
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        print(f"🎮 Device: {self.device}")
        if torch.cuda.is_available():
            print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    def load_data(self, data_path="../Data"):
        """Dataset'leri yükle"""
        print("📊 Dataset yükleniyor...")
        
        try:
            train_df = pd.read_csv(f"{data_path}/train.csv")
            val_df = pd.read_csv(f"{data_path}/val.csv") 
            test_df = pd.read_csv(f"{data_path}/test.csv")
            
            print(f"✅ Eğitim: {len(train_df)} örnekler")
            print(f"✅ Validation: {len(val_df)} örnekler")
            print(f"✅ Test: {len(test_df)} örnekler")
            
            # Kategori dağılımını göster
            print("\n📈 Kategori dağılımı:")
            for label, count in train_df['label'].value_counts().items():
                print(f"   {label}: {count}")
            
            return train_df, val_df, test_df
            
        except FileNotFoundError as e:
            print(f"❌ Dataset dosyaları bulunamadı: {e}")
            print("💡 Lütfen Data/ klasöründe train.csv, val.csv, test.csv dosyalarının olduğundan emin olun.")
            return None, None, None
    
    def prepare_model(self, model_name="bert-base-uncased"):
        """BERT modelini hazırla"""
        print(f"🤖 BERT model hazırlanıyor: {model_name}")
        
        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Config - küçük model için optimize
        config = BertConfig.from_pretrained(
            model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            hidden_size=384,  # Küçük model
            num_hidden_layers=6,  # Daha az layer
            num_attention_heads=6,  # Daha az head
            intermediate_size=1536  # Daha küçük FFN
        )
        
        # Model
        self.model = BertForSequenceClassification(config)
        self.model.to(self.device)
        
        # Model boyutunu hesapla
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Toplam parametreler: {total_params:,}")
        print(f"🎯 Eğitilebilir parametreler: {trainable_params:,}")
        print(f"💾 Model boyutu: ~{total_params * 4 / 1024**2:.1f}MB")
    
    def tokenize_data(self, df):
        """Text'leri tokenize et"""
        encodings = self.tokenizer(
            df['text'].tolist(),
            truncation=True,
            padding=True,
            max_length=128,  # Kısa text'ler için optimize
            return_tensors='pt'
        )
        
        # Labels'ları ID'ye çevir
        labels = [self.label2id[label] for label in df['label'].tolist()]
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels)
        }
    
    def train_model(self, train_df, val_df, epochs=3, batch_size=8):
        """Model eğitimi"""
        print(f"\n🚀 Model eğitimi başlıyor...")
        print(f"   📖 Epochs: {epochs}")
        print(f"   📦 Batch size: {batch_size}")
        
        # Data tokenize
        train_dataset = self.tokenize_data(train_df)
        val_dataset = self.tokenize_data(val_df)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='../Model/checkpoints',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='../Model/logs',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=torch.cuda.is_available(),  # Mixed precision if GPU
            dataloader_num_workers=0,  # Jetson için
            remove_unused_columns=False
        )
        
        # Custom Dataset class
        class BotanicalDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.encodings['input_ids'][idx],
                    'attention_mask': self.encodings['attention_mask'][idx],
                    'labels': self.encodings['labels'][idx]
                }
            
            def __len__(self):
                return len(self.encodings['labels'])
        
        train_dataset = BotanicalDataset(train_dataset)
        val_dataset = BotanicalDataset(val_dataset)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorWithPadding(self.tokenizer)
        )
        
        # Eğitimi başlat
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        print(f"✅ Eğitim tamamlandı! Süre: {training_time:.1f} saniye")
        
        return trainer
    
    def evaluate_model(self, trainer, test_df):
        """Model değerlendirmesi"""
        print("\n📊 Model değerlendirmesi...")
        
        # Test verisi tokenize
        test_dataset = self.tokenize_data(test_df)
        
        class BotanicalDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.encodings['input_ids'][idx],
                    'attention_mask': self.encodings['attention_mask'][idx],
                    'labels': self.encodings['labels'][idx]
                }
            
            def __len__(self):
                return len(self.encodings['labels'])
        
        test_dataset = BotanicalDataset(test_dataset)
        
        # Predictions
        predictions = trainer.predict(test_dataset)
        y_pred = predictions.predictions.argmax(axis=1)
        y_true = test_dataset.encodings['labels'].numpy()
        
        # Metrikleri hesapla
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"🎯 Test Accuracy: {accuracy:.4f}")
        print(f"📊 F1 Score: {f1:.4f}")
        
        # Kategori bazında rapor
        print("\n📋 Kategori bazında performans:")
        report = classification_report(
            y_true, y_pred,
            target_names=list(self.label2id.keys()),
            digits=3
        )
        print(report)
        
        return accuracy, f1
    
    def save_model(self, trainer, accuracy, f1):
        """Model'i kaydet"""
        print("\n💾 Model kaydediliyor...")
        
        model_dir = "../Model/botanical_bert_model"
        os.makedirs(model_dir, exist_ok=True)
        
        # Model ve tokenizer kaydet
        trainer.save_model(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        # Model bilgilerini kaydet
        model_info = {
            "model_type": "BertForSequenceClassification",
            "model_name": "botanical_bert_small",
            "num_labels": len(self.label2id),
            "label2id": self.label2id,
            "id2label": self.id2label,
            "test_accuracy": accuracy,
            "test_f1_score": f1,
            "training_date": datetime.now().isoformat(),
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "model_size_mb": sum(p.numel() for p in self.model.parameters()) * 4 / 1024**2
        }
        
        import json
        with open(f"{model_dir}/model_info.json", 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Model kaydedildi: {model_dir}")
        print(f"📁 Model boyutu: ~{model_info['model_size_mb']:.1f}MB")
        
        return model_dir

def main():
    """Ana eğitim fonksiyonu"""
    print("🌱" + "="*50 + "🌱")
    print("    BOTANICAL BERT MODEL EĞİTİMİ")
    print("🌱" + "="*50 + "🌱")
    print()
    
    # Trainer oluştur
    trainer_obj = BotanicalBERTTrainer()
    
    # Data yükle
    train_df, val_df, test_df = trainer_obj.load_data()
    if train_df is None:
        return
    
    # Model hazırla
    trainer_obj.prepare_model()
    
    # Model eğit
    trainer = trainer_obj.train_model(train_df, val_df, epochs=3, batch_size=8)
    
    # Değerlendir
    accuracy, f1 = trainer_obj.evaluate_model(trainer, test_df)
    
    # Kaydet
    model_path = trainer_obj.save_model(trainer, accuracy, f1)
    
    print(f"\n🎉 EĞİTİM TAMAMLANDI!")
    print(f"🎯 Final Accuracy: {accuracy:.4f}")
    print(f"📊 Final F1 Score: {f1:.4f}")
    print(f"💾 Model: {model_path}")
    print(f"\n🚀 Kullanım için: cd ../Model && python run_model.py")

if __name__ == "__main__":
    main() 