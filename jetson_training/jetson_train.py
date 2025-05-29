#!/usr/bin/env python3
"""
Jetson Orin Nano Optimized RAG Training
========================================
GPU-accelerated training for agricultural disease RAG system
"""

import os
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Jetson optimized imports
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Dataset
from transformers import (
    RagTokenizer, RagSequenceForGeneration, RagConfig,
    TrainingArguments, Trainer, AutoTokenizer
)
from sentence_transformers import SentenceTransformer
import faiss

# Custom imports
from gpu_optimizer import JetsonOptimizer
from export_onnx import export_to_onnx

class AgriculturalRAGDataset(Dataset):
    """Tarımsal RAG veri seti"""
    
    def __init__(self, chunks_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Chunk'ları yükle
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # QA pairs oluştur
        self.qa_pairs = self._create_qa_pairs()
        
    def _create_qa_pairs(self):
        """Chunk'lardan QA çiftleri oluştur"""
        qa_pairs = []
        
        templates = [
            "Bu metinde hangi hastalık açıklanıyor?",
            "Hangi bitki türü hakkında bilgi veriliyor?",
            "Bu hastalığın belirtileri nelerdir?",
            "Tedavi yöntemi nedir?",
            "Bu hastalığa karşı önlem nedir?"
        ]
        
        for chunk in self.chunks[:1000]:  # İlk 1000 chunk ile başla
            text = chunk['text']
            if len(text) > 100:  # Yeterli uzunlukta olsun
                for template in templates:
                    qa_pairs.append({
                        'question': template,
                        'context': text,
                        'answer': text[:200] + "..."  # İlk 200 karakter
                    })
        
        return qa_pairs
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        item = self.qa_pairs[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            item['question'],
            item['context'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Target encoding
        target_encoding = self.tokenizer(
            item['answer'],
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

class JetsonRAGTrainer:
    """Jetson optimized RAG trainer"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Jetson optimizer
        self.optimizer = JetsonOptimizer()
        
        print(f"🚀 Jetson RAG Trainer başlatılıyor...")
        print(f"📱 Device: {self.device}")
        print(f"🧠 CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    def setup_model(self):
        """Model ve tokenizer kurulumu"""
        print("🔧 Model kurulumu...")
        
        # Lightweight model for Jetson
        model_name = "facebook/rag-token-base"
        
        try:
            # Tokenizer
            self.tokenizer = RagTokenizer.from_pretrained(model_name)
            
            # Config - Jetson için optimize
            config = RagConfig.from_pretrained(model_name)
            config.use_cache = False  # Memory optimization
            config.gradient_checkpointing = True  # Memory trade-off
            
            # Model
            self.model = RagSequenceForGeneration.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float16 if self.args.mixed_precision else torch.float32
            )
            
            # GPU'ya taşı
            self.model.to(self.device)
            
            # Jetson optimizasyonları uygula
            if self.args.gpu:
                self.model = self.optimizer.optimize_model(self.model)
            
            print("✅ Model kurulumu tamamlandı")
            
        except Exception as e:
            print(f"❌ Model kurulum hatası: {e}")
            # Fallback to smaller model
            self.setup_fallback_model()
    
    def setup_fallback_model(self):
        """Daha küçük model fallback"""
        print("🔄 Fallback model yükleniyor...")
        
        model_name = "t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        from transformers import T5ForConditionalGeneration
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.args.mixed_precision else torch.float32
        )
        self.model.to(self.device)
        
        print("✅ Fallback model hazır")
    
    def load_data(self):
        """Veri yükleme"""
        print("📂 Veri yükleniyor...")
        
        chunks_file = Path("../final_system/complete_index/chunks/all_chunks.json")
        
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunk dosyası bulunamadı: {chunks_file}")
        
        # Dataset oluştur
        self.train_dataset = AgriculturalRAGDataset(
            chunks_file, self.tokenizer, max_length=512
        )
        
        # DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2,  # Jetson için optimize
            pin_memory=True
        )
        
        print(f"✅ {len(self.train_dataset)} sample yüklendi")
    
    def train(self):
        """Ana eğitim fonksiyonu"""
        print("🎯 Eğitim başlıyor...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"../models/rag_jetson_{datetime.now().strftime('%Y%m%d_%H%M')}",
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=4,  # Memory efficiency
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            fp16=self.args.mixed_precision,  # Jetson FP16 support
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb for Jetson
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Memory optimization
        if self.args.gpu:
            torch.cuda.empty_cache()
        
        # Start training with error handling
        try:
            print("🚀 Eğitim başlatılıyor...")
            trainer.train()
            
            # Save model
            model_path = training_args.output_dir
            trainer.save_model(model_path)
            self.tokenizer.save_pretrained(model_path)
            
            print(f"✅ Model kaydedildi: {model_path}")
            
            # Export to ONNX if requested
            if self.args.export_onnx:
                print("📦 ONNX export...")
                export_to_onnx(self.model, self.tokenizer, model_path)
            
        except Exception as e:
            print(f"❌ Eğitim hatası: {e}")
            print("💡 Batch size'ı küçültmeyi deneyin: --batch_size 1")
    
    def evaluate(self):
        """Model değerlendirme"""
        print("📊 Model değerlendiriliyor...")
        
        # Test queries
        test_queries = [
            "Domates yaprak leke hastalığı nedir?",
            "Buğday pas hastalığı nasıl tedavi edilir?",
            "Mısır kurt zararı belirtileri nelerdir?"
        ]
        
        self.model.eval()
        
        with torch.no_grad():
            for query in test_queries:
                # Tokenize
                inputs = self.tokenizer(
                    query,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # Generate
                with amp.autocast(enabled=self.args.mixed_precision):
                    outputs = self.model.generate(
                        **inputs,
                        max_length=256,
                        num_beams=2,
                        early_stopping=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Decode
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                print(f"\n🔍 Soru: {query}")
                print(f"💬 Cevap: {response}")
        
        print("✅ Değerlendirme tamamlandı")

def main():
    parser = argparse.ArgumentParser(description="Jetson RAG Training")
    
    # Model parameters
    parser.add_argument("--epochs", type=int, default=3, help="Eğitim epoch sayısı")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    
    # Jetson optimizations
    parser.add_argument("--gpu", action="store_true", help="GPU kullan")
    parser.add_argument("--mixed_precision", action="store_true", help="FP16 kullan")
    parser.add_argument("--tensorrt", action="store_true", help="TensorRT optimizasyonu")
    
    # Export options
    parser.add_argument("--export_onnx", action="store_true", help="ONNX export yap")
    
    args = parser.parse_args()
    
    print("🌾 Jetson Agricultural RAG Training")
    print("="*50)
    print(f"📅 Başlangıç: {datetime.now()}")
    
    # Trainer oluştur
    trainer = JetsonRAGTrainer(args)
    
    try:
        # Setup
        trainer.setup_model()
        trainer.load_data()
        
        # Train
        trainer.train()
        
        # Evaluate
        trainer.evaluate()
        
        print("🎉 Eğitim başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"❌ Ana hata: {e}")
        print("💡 GPU memory yetersizse --batch_size 1 deneyin")

if __name__ == "__main__":
    main() 