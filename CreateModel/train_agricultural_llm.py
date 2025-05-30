#!/usr/bin/env python3
"""
Tarımsal LLM Eğitimi
GPT-2 small model ile tarımsal chatbot eğitimi
"""

import os
import json
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    Trainer, TrainingArguments, 
    DataCollatorForLanguageModeling,
    pipeline
)
from datasets import Dataset
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

console = Console()

class AgriculturalLLMTrainer:
    """Tarımsal LLM Eğitim Sistemi"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "gpt2"  # Küçük model - Jetson için uygun
        self.output_dir = Path("agricultural_gpt2")
        
        # Model ve tokenizer
        self.tokenizer = None
        self.model = None
        
        console.print("🚀 Tarımsal LLM Eğitim Sistemi başlatılıyor...", style="bold green")
        
    def prepare_training_data(self):
        """Eğitim verilerini hazırla"""
        console.print("📚 Eğitim verileri hazırlanıyor...", style="cyan")
        
        # Tarımsal Q&A verileri oluştur
        training_data = []
        
        # Elmada erken yanıklığı
        training_data.extend([
            {
                "input": "Elmada erken yanıklığı nedir?",
                "output": "Elmada erken yanıklığı (Erwinia amylovora), meyve ağaçlarının en ciddi bakteriyel hastalığıdır. Bu hastalık çiçek kümeleri, yapraklar, sürgünler ve dalları etkiler. Belirtileri arasında yapraklarda siyah lekeler, çiçek kümeleri ve sürgünlerin yanık görünümü, dal uçlarının kucar şeklinde bükümlümesi yer alır."
            },
            {
                "input": "Elmada erken yanıklığını nasıl tedavi edebilirim?",
                "output": "Elmada erken yanıklığı tedavisi için şu adımları izleyin: 1) Hasta dalları sağlam kısımdan 30 cm mesafeden kesin, 2) Kesim aletlerini %10 çamaşır suyu ile sterilize edin, 3) Streptomisin içerikli antibiyotik sprey uygulayın, 4) Bakır sülfat spreyi yapın, 5) Kesim yerlerini bahçe macunu ile kapatın. Koruyucu önlem olarak çiçeklenme döneminde düzenli ilaçlama yapın."
            },
            {
                "input": "Elmada erken yanıklığından nasıl korunabilirim?",
                "output": "Elmada erken yanıklığından korunmak için: Aşırı azotlu gübreden kaçının, budama aletlerini sterilize edin, çiçeklenme döneminde koruyucu ilaçlama yapın, dayanıklı elma çeşitlerini tercih edin, bahçe hijyenine dikkat edin ve hasta bitki artıklarını temizleyin."
            }
        ])
        
        # Buğday ekim
        training_data.extend([
            {
                "input": "Buğday ekim zamanı ne zaman?",
                "output": "Buğday ekim zamanı bölgeye göre değişir. Kışlık buğday Ekim-Kasım aylarında, yazlık buğday ise Mart-Nisan aylarında ekilir. Ekim için toprak sıcaklığının 8-12°C arasında olması gerekir. Ekim derinliği 3-4 cm ideal kabul edilir."
            },
            {
                "input": "Buğday ekimi için toprak nasıl hazırlanır?",
                "output": "Buğday ekim öncesi toprak hazırlığı çok önemlidir. Önce derin sürüm yapın, ardından diskaro ve merdane ile toprağı hazırlayın. Toprak pH'sı 6.0-7.5 arasında olmalıdır. Ekim öncesi fosfor ve potasyum gübresi verilmelidir. Toprak iyi drenaja sahip olmalı ve yabancı otlardan temizlenmelidir."
            }
        ])
        
        # Domates sarı yaprak
        training_data.extend([
            {
                "input": "Domates bitkilerinde sarı yaprak sorunu neden oluşur?",
                "output": "Domates bitkilerinde sarı yaprak sorununun başlıca sebepleri şunlardır: 1) Aşırı veya yetersiz sulama, 2) Azot eksikliği, 3) Magnezyum eksikliği, 4) Kök hastalıkları, 5) Doğal yaşlanma süreci. En yaygın neden beslenme bozukluklarıdır."
            },
            {
                "input": "Domates sarı yaprak sorununu nasıl çözerim?",
                "output": "Domates sarı yaprak tedavisi için: Sulama düzenini kontrol edin ve düzenli sulama yapın, azotlu gübre uygulayın, magnezyum sülfat spreyi yapın, hasta yaprakları temizleyin, drip sulama sistemi kurun. Ayrıca toprak pH'sını kontrol edin ve gerekirse düzeltin."
            }
        ])
        
        # Havuç yetiştirme
        training_data.extend([
            {
                "input": "Havuç nasıl yetiştirilir?",
                "output": "Havuç yetiştirmek için derin, gevşek ve taşsız toprak gerekir. Ekim derinliği 1-2 cm, sıra arası 25-30 cm olmalıdır. Tohum çıkışı için toprağı nemli tutun. Ekim zamanı ilkbaharda Mart-Nisan, sonbaharda Ağustos-Eylül aylarıdır. Hasat 90-120 günde yapılır."
            },
            {
                "input": "Havuç yetiştirmede sulama nasıl yapılır?",
                "output": "Havuç yetiştirmede sulama düzenli ama aşırı olmamalıdır. Kuraklık köklerin çatlamasına, aşırı nem ise çürümeye neden olur. Drip sulama ideal yöntemdir. Toprak nemini kontrol ederek sulama yapın, su kökün derinlerine ulaşmalıdır."
            }
        ])
        
        # Genel tarım konuları
        training_data.extend([
            {
                "input": "Organik tarım nedir?",
                "output": "Organik tarım, kimyasal gübre ve pestisit kullanmadan, doğal yöntemlerle üretim yapan tarım sistemidir. Kompost kullanımı, ekim nöbeti, yararlı böcekler, yeşil gübre ve doğal pestisitler kullanılır. Toprak sağlığını korur ve çevre dostu üretim sağlar."
            },
            {
                "input": "Toprak pH'sı neden önemlidir?",
                "output": "Toprak pH'sı bitki beslenmesi için kritiktir. Asidik topraklar (pH < 6) kireçleme ile, alkalin topraklar (pH > 7.5) sülfür ile düzeltilir. Çoğu bitki pH 6.0-7.0 arasını tercih eder. Doğru pH besin alımını optimize eder ve hastalık direncini artırır."
            }
        ])
        
        # Aşırı sıcaklık ve bitki korunması
        training_data.extend([
            {
                "input": "Aşırı sıcaklıkta bitkileri nasıl koruruz?",
                "output": "Aşırı sıcaklıkta bitkileri korumak için: Gölgeleme ağları kurun, mulch uygulayın, sık ve düzenli sulama yapın, potasyum gübresi verin (sıcaklık stresine karşı direnç artırır), erken sabah veya akşam sulaması yapın, yaprakları nemli tutmak için spreyleme yapın, sera ventilasyonunu artırın."
            },
            {
                "input": "Sıcak havada bitkilere ne tür takviye verilir?",
                "output": "Sıcak havada bitkilere şu takviyeler verilebilir: Potasyum sülfat (stres direnci artırır), magnezyum sülfat (klorofil korunur), kalsiyum nitrat (hücre duvarı güçlenir), seaweed extract (doğal stres direnci), silikon gübresi (yaprak yüzeyi güçlenir), aminoasit karışımları (stres recovery), zeolitik mineraller (su tutma kapasitesi artar)."
            }
        ])
        
        console.print(f"✅ {len(training_data)} eğitim örneği hazırlandı", style="green")
        return training_data
    
    def format_training_data(self, data: List[Dict]) -> List[str]:
        """Eğitim verilerini GPT-2 formatına çevir"""
        formatted_data = []
        
        for item in data:
            # Soru-cevap formatı
            formatted_text = f"<|soru|>{item['input']}<|cevap|>{item['output']}<|end|>"
            formatted_data.append(formatted_text)
        
        return formatted_data
    
    def prepare_model_and_tokenizer(self):
        """Model ve tokenizer'ı hazırla"""
        console.print("🧠 GPT-2 model yükleniyor...", style="cyan")
        
        # Tokenizer yükle
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        
        # Özel tokenlar ekle
        special_tokens = {
            "pad_token": "<|pad|>",
            "eos_token": "<|end|>",
            "additional_special_tokens": ["<|soru|>", "<|cevap|>"]
        }
        
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Model yükle
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        
        console.print("✅ Model ve tokenizer hazır", style="green")
    
    def create_dataset(self, texts: List[str]) -> Dataset:
        """Dataset oluştur"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding=True, 
                max_length=512,
                return_tensors="pt"
            )
        
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train_model(self, dataset: Dataset):
        """Modeli eğit"""
        console.print("🏋️ Model eğitimi başlıyor...", style="cyan")
        
        # Eğitim argümanları
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=3,  # Jetson için makul
            per_device_train_batch_size=2,  # Küçük batch size
            save_steps=100,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            warmup_steps=50,
            learning_rate=5e-5,
            fp16=torch.cuda.is_available(),  # Mixed precision
            dataloader_num_workers=0,  # Jetson için
            report_to=None,  # Wandb kapalı
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # GPT-2 causal LM
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        # Eğitimi başlat
        trainer.train()
        
        # Modeli kaydet
        trainer.save_model()
        self.tokenizer.save_pretrained(str(self.output_dir))
        
        console.print("✅ Model eğitimi tamamlandı ve kaydedildi!", style="bold green")
    
    def test_model(self):
        """Eğitilmiş modeli test et"""
        console.print("🧪 Model test ediliyor...", style="cyan")
        
        # Text generation pipeline
        generator = pipeline(
            "text-generation",
            model=str(self.output_dir),
            tokenizer=str(self.output_dir),
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Test soruları
        test_questions = [
            "Elmada erken yanıklığı nedir?",
            "Buğday ekim zamanı ne zaman?",
            "Domates sarı yaprak sorunu nasıl çözülür?",
            "Aşırı sıcaklıkta bitkileri nasıl koruruz?"
        ]
        
        for question in test_questions:
            prompt = f"<|soru|>{question}<|cevap|>"
            
            response = generator(
                prompt,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            answer = generated_text.split("<|cevap|>")[-1].split("<|end|>")[0].strip()
            
            console.print(f"\n[bold blue]Soru:[/bold blue] {question}")
            console.print(f"[bold green]Cevap:[/bold green] {answer}")
            console.print("-" * 50)
    
    def run_training(self):
        """Tam eğitim sürecini çalıştır"""
        try:
            # 1. Verileri hazırla
            training_data = self.prepare_training_data()
            formatted_texts = self.format_training_data(training_data)
            
            # 2. Model ve tokenizer hazırla
            self.prepare_model_and_tokenizer()
            
            # 3. Dataset oluştur
            dataset = self.create_dataset(formatted_texts)
            
            # 4. Modeli eğit
            self.train_model(dataset)
            
            # 5. Test et
            self.test_model()
            
            console.print("\n🎉 Tarımsal LLM başarıyla eğitildi!", style="bold green")
            
        except Exception as e:
            console.print(f"❌ Eğitim sırasında hata: {e}", style="bold red")

def main():
    """Ana fonksiyon"""
    trainer = AgriculturalLLMTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main() 