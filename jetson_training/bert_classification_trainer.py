#!/usr/bin/env python3
"""
BERT Classification Trainer - JetPack 6.2 Optimized
==================================================
Custom dataset ve AG News için optimize edilmiş BERT eğitimi
"""

import torch
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
import matplotlib.pyplot as plt
from tqdm import tqdm

# JetPack 6.2 optimizations
from gpu_optimizer_jp62 import JetsonOptimizerJP62, JetsonProfilerJP62

class BERTDataset(Dataset):
    """JetPack 6.2 optimize edilmiş BERT dataset"""
    
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class JetsonBERTTrainer:
    """JetPack 6.2 optimized BERT trainer"""
    
    def __init__(self, mixed_precision=True, max_length=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mixed_precision = mixed_precision
        self.max_length = max_length
        
        # JetPack 6.2 optimizer
        self.jetson_opt = JetsonOptimizerJP62()
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=mixed_precision)
        
        # Results storage
        self.results = []
        
        print(f"🚀 JetPack 6.2 BERT Trainer başlatıldı")
        print(f"📱 Device: {self.device}")
        print(f"🔧 Mixed Precision: {mixed_precision}")
        
        # Monitor initial GPU state
        self.jetson_opt.monitor_jetson_gpu()
    
    def load_custom_data(self, path="veri.csv"):
        """Kendi veri setini yükle"""
        print(f"📂 Kendi veri seti yükleniyor: {path}")
        
        try:
            df = pd.read_csv(path)
            
            # Kategorik label'ları sayısala çevir
            label2id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
            id2label = {idx: label for label, idx in label2id.items()}
            df['label'] = df['label'].map(label2id)
            
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)
            
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)
            
            print(f"✅ Custom dataset: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
            
            return (train_texts.tolist(), val_texts.tolist(), test_texts.tolist(),
                   train_labels.tolist(), val_labels.tolist(), test_labels.tolist(),
                   id2label)
                   
        except FileNotFoundError:
            print(f"❌ Dosya bulunamadı: {path}")
            return None
    
    def load_ag_news_data(self):
        """AG News veri setini yükle"""
        print("📂 AG News veri seti yükleniyor...")
        
        try:
            url_train = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
            url_test = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
            
            train_df = pd.read_csv(url_train, header=None, names=["label", "title", "description"])
            test_df = pd.read_csv(url_test, header=None, names=["label", "title", "description"])
            
            # Label'ları 0-3 aralığına çevir
            train_df["label"] = train_df["label"] - 1
            test_df["label"] = test_df["label"] - 1
            
            # Text'leri birleştir
            texts = (train_df["title"] + ". " + train_df["description"]).tolist()
            labels = train_df["label"].tolist()
            
            test_texts = (test_df["title"] + ". " + test_df["description"]).tolist()
            test_labels = test_df["label"].tolist()
            
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, stratify=labels, random_state=42)
            
            id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
            
            print(f"✅ AG News: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
            
            return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, id2label
            
        except Exception as e:
            print(f"❌ AG News yükleme hatası: {e}")
            return None
    
    def create_data_loaders(self, train_texts, val_texts, test_texts, 
                           train_labels, val_labels, test_labels, tokenizer, batch_size=8):
        """Data loader'ları oluştur"""
        
        train_dataset = BERTDataset(train_texts, train_labels, tokenizer, self.max_length)
        val_dataset = BERTDataset(val_texts, val_labels, tokenizer, self.max_length)
        test_dataset = BERTDataset(test_texts, test_labels, tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, train_loader, optimizer, scheduler):
        """Tek epoch eğitimi"""
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=self.mixed_precision):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def evaluate_model(self, model, data_loader):
        """Model değerlendirme"""
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with autocast(enabled=self.mixed_precision):
                    outputs = model(input_ids, attention_mask=attention_mask)
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return acc, precision, f1, all_labels, all_preds
    
    def train_model(self, model_name, train_loader, val_loader, test_loader, 
                   num_labels, epochs=3, lr=2e-5, batch_size=8):
        """Model eğitimi"""
        print(f"\n🎯 {model_name} eğitimi başlıyor...")
        
        # Profiler başlat
        profiler = JetsonProfilerJP62()
        profiler.start()
        
        # Model ve tokenizer yükle
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
        # JetPack 6.2 optimizasyonları uygula
        model = self.jetson_opt.optimize_model_jp62(model)
        model.to(self.device)
        
        # Optimizer ve scheduler
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * epochs
        )
        
        # Training history
        history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
        
        best_f1 = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, scheduler)
            
            # Validate
            val_acc, val_prec, val_f1, _, _ = self.evaluate_model(model, val_loader)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")
            print(f"  Val Precision: {val_prec:.4f}")
            print(f"  Val F1: {val_f1:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            # Memory monitoring
            self.jetson_opt.optimize_memory_jp62()
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_path = Path(f"../models/best_{model_name.replace('/', '_')}")
                best_model_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
        
        total_time = time.time() - start_time
        
        # Test evaluation
        test_acc, test_prec, test_f1, test_labels, test_preds = self.evaluate_model(model, test_loader)
        
        # Profile end
        prof_time, prof_memory = profiler.end()
        
        print(f"\n📊 {model_name} Final Results:")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test Precision: {test_prec:.4f}")
        print(f"  Test F1: {test_f1:.4f}")
        print(f"  Total Time: {total_time:.2f}s")
        
        # Classification report
        report = classification_report(test_labels, test_preds, output_dict=True)
        
        # Save results
        result = {
            'model': model_name,
            'test_accuracy': test_acc,
            'test_precision': test_prec,
            'test_f1': test_f1,
            'training_time': total_time,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'best_val_f1': best_f1,
            'history': history,
            'classification_report': report
        }
        
        self.results.append(result)
        
        # Memory cleanup
        del model, tokenizer, optimizer, scheduler
        torch.cuda.empty_cache()
        
        return result
    
    def run_custom_experiment(self, data_path="veri.csv"):
        """Kendi veri seti deneyi"""
        print("🌾 Custom Dataset Experiment Başlıyor...")
        
        data = self.load_custom_data(data_path)
        if data is None:
            print("❌ Custom dataset yüklenemedi")
            return
        
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, id2label = data
        num_labels = len(id2label)
        
        print(f"📊 Label mapping: {id2label}")
        
        # BERT base model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, 
            tokenizer, batch_size=8
        )
        
        result = self.train_model(
            'bert-base-uncased', train_loader, val_loader, test_loader, 
            num_labels, epochs=3, batch_size=8
        )
        
        self.save_results("custom_dataset_results")
        return result
    
    def run_ag_news_experiments(self):
        """AG News dataset deneyleri"""
        print("📰 AG News Experiments Başlıyor...")
        
        data = self.load_ag_news_data()
        if data is None:
            print("❌ AG News dataset yüklenemedi")
            return
        
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, id2label = data
        
        models = ['bert-base-uncased', 'bert-large-uncased']
        batch_sizes = {'bert-base-uncased': 8, 'bert-large-uncased': 4}  # Large model için küçük batch
        
        for model_name in models:
            print(f"\n{'='*50}")
            print(f"🎯 {model_name} Training")
            print(f"{'='*50}")
            
            try:
                tokenizer = BertTokenizer.from_pretrained(model_name)
                batch_size = batch_sizes[model_name]
                
                train_loader, val_loader, test_loader = self.create_data_loaders(
                    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels,
                    tokenizer, batch_size=batch_size
                )
                
                result = self.train_model(
                    model_name, train_loader, val_loader, test_loader,
                    4, epochs=3, batch_size=batch_size
                )
                
            except Exception as e:
                print(f"❌ {model_name} eğitim hatası: {e}")
                continue
        
        self.save_results("ag_news_results")
        self.plot_results()
    
    def save_results(self, filename):
        """Sonuçları kaydet"""
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        
        # JSON format
        with open(results_dir / f"{filename}.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # CSV format
        df_results = []
        for result in self.results:
            df_results.append({
                'Model': result['model'],
                'Test_Accuracy': result['test_accuracy'],
                'Test_Precision': result['test_precision'],
                'Test_F1': result['test_f1'],
                'Training_Time': result['training_time'],
                'Best_Val_F1': result['best_val_f1']
            })
        
        df = pd.DataFrame(df_results)
        df.to_csv(results_dir / f"{filename}.csv", index=False)
        
        print(f"📁 Sonuçlar kaydedildi: {results_dir}")
        print("\n📊 Final Results:")
        print(df)
    
    def plot_results(self):
        """Sonuçları görselleştir"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        models = [r['model'] for r in self.results]
        accuracies = [r['test_accuracy'] for r in self.results]
        f1_scores = [r['test_f1'] for r in self.results]
        times = [r['training_time'] for r in self.results]
        
        # Accuracy comparison
        axes[0, 0].bar(models, accuracies, color='skyblue')
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        axes[0, 1].bar(models, f1_scores, color='lightgreen')
        axes[0, 1].set_title('Test F1 Score Comparison')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training time comparison
        axes[1, 0].bar(models, times, color='salmon')
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Combined metrics
        x = np.arange(len(models))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        axes[1, 1].bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
        axes[1, 1].set_title('Accuracy vs F1 Score')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        results_dir = Path("../results")
        plt.savefig(results_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Grafik kaydedildi: {results_dir}/model_comparison.png")

    def load_agricultural_data(self, dataset_type="categorized"):
        """Tarımsal veri setini yükle"""
        print(f"🌾 Tarımsal veri seti yükleniyor: {dataset_type}")
        
        try:
            if dataset_type == "categorized":
                # Ana kategorik dataset - yol kontrolü
                base_paths = ["../agricultural_datasets/", "agricultural_datasets/"]
                train_df, val_df, test_df = None, None, None
                
                for base_path in base_paths:
                    try:
                        train_path = base_path + "train.csv"
                        val_path = base_path + "val.csv"
                        test_path = base_path + "test.csv"
                        
                        if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
                            train_df = pd.read_csv(train_path)
                            val_df = pd.read_csv(val_path)
                            test_df = pd.read_csv(test_path)
                            print(f"📁 Dataset dosyaları bulundu: {base_path}")
                            break
                    except:
                        continue
                
                if train_df is None:
                    raise FileNotFoundError("Kategorik dataset dosyaları bulunamadı")
                
                # Label mapping oluştur
                all_labels = sorted(pd.concat([train_df['label'], val_df['label'], test_df['label']]).unique())
                label2id = {label: idx for idx, label in enumerate(all_labels)}
                id2label = {idx: label for label, idx in label2id.items()}
                
                # Label'ları sayıya çevir
                train_labels = train_df['label'].map(label2id).tolist()
                val_labels = val_df['label'].map(label2id).tolist()
                test_labels = test_df['label'].map(label2id).tolist()
                
                train_texts = train_df['text'].tolist()
                val_texts = val_df['text'].tolist()
                test_texts = test_df['text'].tolist()
                
                print(f"✅ Agricultural Categorized: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
                print(f"📊 Categories: {list(id2label.values())}")
                
            elif dataset_type == "sentiment":
                # Sentiment dataset - yol kontrolü
                sentiment_paths = ["../agricultural_datasets/agricultural_sentiment.csv", 
                                 "agricultural_datasets/agricultural_sentiment.csv"]
                df = None
                
                for path in sentiment_paths:
                    if os.path.exists(path):
                        df = pd.read_csv(path)
                        print(f"📁 Sentiment dataset bulundu: {path}")
                        break
                
                if df is None:
                    raise FileNotFoundError("Sentiment dataset dosyası bulunamadı")
                
                # Label mapping
                label2id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
                id2label = {idx: label for label, idx in label2id.items()}
                df['label'] = df['label'].map(label2id)
                
                # Train/val/test split
                train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                    df['text'], df['label'], test_size=0.3, stratify=df['label'], random_state=42)
                
                val_texts, test_texts, val_labels, test_labels = train_test_split(
                    temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)
                
                train_texts, val_texts, test_texts = train_texts.tolist(), val_texts.tolist(), test_texts.tolist()
                train_labels, val_labels, test_labels = train_labels.tolist(), val_labels.tolist(), test_labels.tolist()
                
                print(f"✅ Agricultural Sentiment: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
                print(f"📊 Sentiments: {list(id2label.values())}")
            
            return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, id2label
            
        except Exception as e:
            print(f"❌ Tarımsal veri yükleme hatası: {e}")
            print(f"🔍 Mevcut dizin: {os.getcwd()}")
            print(f"🔍 Dosya listesi:")
            for item in ["../agricultural_datasets/", "agricultural_datasets/"]:
                if os.path.exists(item):
                    print(f"  {item}: {os.listdir(item)[:5]}")  # İlk 5 dosya
            return None
    
    def run_agricultural_experiments(self):
        """Tarımsal veri seti deneyleri"""
        print("🌾 Agricultural Dataset Experiments Başlıyor...")
        
        experiments = [
            ("categorized", "bert-base-uncased", 8),
            ("sentiment", "bert-base-uncased", 8),
            ("categorized", "bert-large-uncased", 4)
        ]
        
        for dataset_type, model_name, batch_size in experiments:
            print(f"\n{'='*60}")
            print(f"🎯 {dataset_type.title()} Dataset + {model_name}")
            print(f"{'='*60}")
            
            try:
                # Veri yükle
                data = self.load_agricultural_data(dataset_type)
                if data is None:
                    continue
                    
                train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, id2label = data
                num_labels = len(id2label)
                
                # Tokenizer ve data loaders
                tokenizer = BertTokenizer.from_pretrained(model_name)
                train_loader, val_loader, test_loader = self.create_data_loaders(
                    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels,
                    tokenizer, batch_size=batch_size
                )
                
                # Model eğitimi
                result = self.train_model(
                    f"{model_name}_{dataset_type}", train_loader, val_loader, test_loader,
                    num_labels, epochs=3, batch_size=batch_size
                )
                
                print(f"✅ {dataset_type} + {model_name} tamamlandı")
                
            except Exception as e:
                print(f"❌ {dataset_type} + {model_name} hatası: {e}")
                continue
        
        self.save_results("agricultural_experiments_results")
        self.plot_results()

def main():
    """Ana fonksiyon"""
    print("🚀 JetPack 6.2 BERT Classification Training")
    print("=" * 50)
    
    trainer = JetsonBERTTrainer(mixed_precision=True, max_length=128)
    
    # Custom dataset deneyi (eğer dosya varsa)
    print("\n1️⃣ Custom Dataset Experiment")
    custom_result = trainer.run_custom_experiment("veri.csv")
    
    # AG News deneyleri
    print("\n2️⃣ AG News Dataset Experiments")
    trainer.run_ag_news_experiments()
    
    # Tarımsal veri seti deneyleri
    print("\n3️⃣ Agricultural Dataset Experiments")
    trainer.run_agricultural_experiments()
    
    print("\n🎉 Tüm deneyler tamamlandı!")
    
    # Final GPU monitoring
    trainer.jetson_opt.monitor_jetson_gpu()

if __name__ == '__main__':
    main() 