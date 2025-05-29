#!/usr/bin/env python3
"""
Full Performance Jetson Trainer
===============================
JetPack 6.0 ile maksimum performans için optimize edilmiş eğitim sistemi
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import logging
import argparse

# Advanced imports for full performance
from transformers import (
    RagTokenizer, RagSequenceForGeneration, RagConfig,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from sentence_transformers import SentenceTransformer
import faiss

# Custom optimizations
from gpu_optimizer import JetsonOptimizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/full_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedAgriculturalDataset(Dataset):
    """Gelişmiş tarımsal RAG veri seti - tam performans için"""
    
    def __init__(self, chunks_file, tokenizer, max_input_length=512, max_target_length=256):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        logger.info(f"Veri seti yükleniyor: {chunks_file}")
        
        # Load all chunks
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        logger.info(f"Toplam {len(self.chunks)} chunk yüklendi")
        
        # Create comprehensive QA pairs
        self.qa_pairs = self._create_comprehensive_qa_pairs()
        
        logger.info(f"Toplam {len(self.qa_pairs)} QA çifti oluşturuldu")
    
    def _create_comprehensive_qa_pairs(self):
        """Kapsamlı QA çiftleri oluştur"""
        qa_pairs = []
        
        # Gelişmiş soru şablonları
        question_templates = {
            'disease_identification': [
                "Bu metinde hangi bitki hastalığı tanımlanıyor?",
                "Açıklanan hastalığın adı nedir?",
                "Bu hastalık hangi bitkiyi etkiliyor?"
            ],
            'symptoms': [
                "Bu hastalığın belirtileri nelerdir?",
                "Hastalığın görünür semptomları nasıldır?",
                "Bu hastalıkta gözlenen değişiklikler nelerdir?"
            ],
            'treatment': [
                "Bu hastalık nasıl tedavi edilir?",
                "Önerilen tedavi yöntemi nedir?",
                "Bu hastalığa karşı hangi ilaçlar kullanılır?"
            ],
            'prevention': [
                "Bu hastalıktan nasıl korunulur?",
                "Önleyici tedbirler nelerdir?",
                "Bu hastalığı engellemek için ne yapılmalı?"
            ],
            'causes': [
                "Bu hastalığın nedeni nedir?",
                "Hastalığa neden olan etken nedir?",
                "Bu hastalık nasıl oluşur?"
            ]
        }
        
        for chunk in tqdm(self.chunks, desc="QA çiftleri oluşturuluyor"):
            text = chunk['text']
            
            # Yeterli uzunlukta metinler
            if len(text.split()) < 20:
                continue
            
            # Her kategori için QA çiftleri
            for category, templates in question_templates.items():
                for template in templates:
                    # Context olarak chunk text
                    context = text
                    
                    # Answer olarak ilgili kısmı extract et
                    answer = self._extract_relevant_answer(text, category)
                    
                    if answer and len(answer.split()) > 5:
                        qa_pairs.append({
                            'question': template,
                            'context': context,
                            'answer': answer,
                            'category': category,
                            'source': chunk.get('source', 'unknown')
                        })
        
        return qa_pairs
    
    def _extract_relevant_answer(self, text, category):
        """Kategoriye göre ilgili cevap çıkar"""
        sentences = text.split('. ')
        
        keywords = {
            'disease_identification': ['hastalık', 'disease', 'fungus', 'virus', 'bacteria'],
            'symptoms': ['belirtiler', 'semptom', 'görünüm', 'symptoms', 'signs'],
            'treatment': ['tedavi', 'ilaç', 'treatment', 'control', 'fungicide'],
            'prevention': ['önlem', 'korunma', 'prevention', 'avoid'],
            'causes': ['neden', 'sebep', 'cause', 'pathogen']
        }
        
        relevant_sentences = []
        for sentence in sentences:
            for keyword in keywords.get(category, []):
                if keyword.lower() in sentence.lower():
                    relevant_sentences.append(sentence.strip())
                    break
        
        if relevant_sentences:
            return '. '.join(relevant_sentences[:3])  # İlk 3 ilgili cümle
        else:
            return sentences[0] if sentences else ""  # Fallback
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        item = self.qa_pairs[idx]
        
        # Input: question + context
        input_text = f"question: {item['question']} context: {item['context']}"
        target_text = item['answer']
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

class FullPerformanceTrainer:
    """Tam performans Jetson trainer"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        
        # Setup directories
        self.setup_directories()
        
        # Initialize optimizer
        self.jetson_optimizer = JetsonOptimizer()
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=args.mixed_precision)
        
        logger.info("🚀 Full Performance Trainer başlatıldı")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🧠 CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            logger.info(f"📊 GPU: {props.name}")
            logger.info(f"💾 GPU Memory: {props.total_memory / 1e9:.1f}GB")
    
    def setup_directories(self):
        """Dizinleri oluştur"""
        self.output_dir = Path(f"../models/full_performance_{datetime.now().strftime('%Y%m%d_%H%M')}")
        self.logs_dir = Path("../logs")
        self.checkpoints_dir = self.output_dir / "checkpoints"
        
        for dir_path in [self.output_dir, self.logs_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_model_and_tokenizer(self):
        """Model ve tokenizer kurulumu"""
        logger.info("🔧 Model ve tokenizer yükleniyor...")
        
        model_name = "facebook/rag-token-base"
        
        try:
            # Tokenizer
            self.tokenizer = RagTokenizer.from_pretrained(model_name)
            
            # Special tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Config for full performance
            config = RagConfig.from_pretrained(model_name)
            config.use_cache = False
            config.gradient_checkpointing = True
            config.max_length = 512
            config.min_length = 10
            config.num_beams = 4
            config.length_penalty = 1.0
            config.early_stopping = True
            
            # Model with optimizations
            self.model = RagSequenceForGeneration.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float16 if self.args.mixed_precision else torch.float32
            )
            
            # Move to device
            self.model.to(self.device)
            
            # Apply Jetson optimizations
            if self.args.gpu:
                self.model = self.jetson_optimizer.optimize_model(self.model)
            
            logger.info("✅ Model başarıyla yüklendi")
            
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}")
            raise e
    
    def load_dataset(self):
        """Gelişmiş veri seti yükleme"""
        logger.info("📂 Veri seti hazırlanıyor...")
        
        chunks_file = Path("../final_system/complete_index/chunks/all_chunks.json")
        
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunk dosyası bulunamadı: {chunks_file}")
        
        # Full dataset
        self.train_dataset = AdvancedAgriculturalDataset(
            chunks_file, 
            self.tokenizer,
            max_input_length=512,
            max_target_length=256
        )
        
        # Data collator
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            max_length=512
        )
        
        logger.info(f"✅ {len(self.train_dataset)} sample hazır")
    
    def setup_training_args(self):
        """Tam performans eğitim argümanları"""
        
        # Optimal batch size calculation
        optimal_batch_size = self.jetson_optimizer.get_optimal_batch_size(
            self.model, 
            (512,), 
            max_memory_gb=6
        )
        
        if self.args.auto_batch_size:
            batch_size = min(optimal_batch_size, self.args.batch_size)
        else:
            batch_size = self.args.batch_size
        
        logger.info(f"🎯 Kullanılan batch size: {batch_size}")
        
        self.training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            
            # Training parameters
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=max(1, self.args.batch_size // batch_size),
            
            # Optimization
            learning_rate=self.args.learning_rate,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            
            # Scheduler
            warmup_steps=500,
            lr_scheduler_type="linear",
            
            # Mixed precision
            fp16=self.args.mixed_precision,
            fp16_opt_level="O1",
            
            # Memory optimization
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            gradient_checkpointing=True,
            
            # Logging and saving
            logging_steps=50,
            save_steps=1000,
            save_total_limit=3,
            evaluation_strategy="no",  # Memory optimization
            
            # Performance
            ignore_data_skip=True,
            remove_unused_columns=False,
            prediction_loss_only=True,
            
            # Disable external reporting for Jetson
            report_to=None,
            
            # Advanced optimizations
            ddp_find_unused_parameters=False,
            dataloader_drop_last=True,
        )
    
    def train(self):
        """Tam performans eğitim"""
        logger.info("🎯 Tam performans eğitim başlıyor...")
        
        # Setup everything
        self.setup_model_and_tokenizer()
        self.load_dataset()
        self.setup_training_args()
        
        # Trainer with custom optimizations
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        
        # Custom optimizer for better performance
        if self.args.custom_optimizer:
            optimizer = self.setup_custom_optimizer()
            trainer.optimizer = optimizer
        
        # Memory cleanup before training
        self.jetson_optimizer.optimize_memory()
        
        try:
            # Start training
            logger.info("🚀 Eğitim başlatıldı...")
            training_result = trainer.train()
            
            # Save final model
            trainer.save_model(str(self.output_dir / "final_model"))
            self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
            
            # Save training metrics
            with open(self.output_dir / "training_metrics.json", 'w') as f:
                json.dump(training_result.metrics, f, indent=2)
            
            logger.info(f"✅ Eğitim tamamlandı: {self.output_dir}")
            
            return training_result
            
        except Exception as e:
            logger.error(f"❌ Eğitim hatası: {e}")
            raise e
    
    def setup_custom_optimizer(self):
        """Jetson için optimize edilmiş optimizer"""
        
        # Parameter groups
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # AdamW with optimizations
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        return optimizer
    
    def evaluate_model(self):
        """Model değerlendirme"""
        logger.info("📊 Model değerlendiriliyor...")
        
        test_queries = [
            "Domates yaprak leke hastalığının belirtileri nelerdir?",
            "Buğday pas hastalığı nasıl tedavi edilir?",
            "Mısır kurt zararından nasıl korunulur?",
            "Fungal enfeksiyonların nedenleri nelerdir?",
            "Bitki hastalıklarında erken teşhis nasıl yapılır?"
        ]
        
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for query in test_queries:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                # Tokenize
                inputs = self.tokenizer(
                    f"question: {query}",
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                start_time.record()
                
                # Generate with optimizations
                with autocast(enabled=self.args.mixed_precision):
                    outputs = self.model.generate(
                        **inputs,
                        max_length=256,
                        min_length=20,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=2,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=False
                    )
                
                end_time.record()
                torch.cuda.synchronize()
                
                # Decode
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                inference_time = start_time.elapsed_time(end_time)
                
                results.append({
                    'query': query,
                    'response': response,
                    'inference_time_ms': inference_time
                })
                
                logger.info(f"🔍 Soru: {query}")
                logger.info(f"💬 Cevap: {response}")
                logger.info(f"⏱️ Süre: {inference_time:.2f}ms")
                logger.info("-" * 50)
        
        # Save evaluation results
        with open(self.output_dir / "evaluation_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        avg_time = np.mean([r['inference_time_ms'] for r in results])
        logger.info(f"📈 Ortalama inference süresi: {avg_time:.2f}ms")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Full Performance Jetson RAG Training")
    
    # Model parameters
    parser.add_argument("--epochs", type=int, default=5, help="Eğitim epoch sayısı")
    parser.add_argument("--batch_size", type=int, default=8, help="Target batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    
    # Jetson optimizations
    parser.add_argument("--gpu", action="store_true", default=True, help="GPU kullan")
    parser.add_argument("--mixed_precision", action="store_true", default=True, help="FP16 kullan")
    parser.add_argument("--auto_batch_size", action="store_true", default=True, help="Otomatik batch size")
    parser.add_argument("--custom_optimizer", action="store_true", default=True, help="Custom optimizer")
    
    # Advanced features
    parser.add_argument("--tensorrt", action="store_true", help="TensorRT export")
    parser.add_argument("--profile", action="store_true", help="Performance profiling")
    
    args = parser.parse_args()
    
    print("🌾 Full Performance Jetson Agricultural RAG Training")
    print("=" * 60)
    print(f"📅 Başlangıç: {datetime.now()}")
    print(f"🎯 Hedef: Maksimum performans eğitimi")
    
    # Create trainer
    trainer = FullPerformanceTrainer(args)
    
    try:
        # Monitor initial state
        trainer.jetson_optimizer.monitor_gpu()
        
        # Train model
        training_result = trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate_model()
        
        # Final monitoring
        trainer.jetson_optimizer.monitor_gpu()
        
        print("🎉 Tam performans eğitimi başarıyla tamamlandı!")
        print(f"📊 Model kaydedildi: {trainer.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Ana hata: {e}")
        print("💡 Çözüm önerileri:")
        print("   - --batch_size 4 deneyin")
        print("   - GPU memory durumunu kontrol edin")
        print("   - nvpmodel -m 0 ve jetson_clocks çalıştırın")

if __name__ == "__main__":
    main() 