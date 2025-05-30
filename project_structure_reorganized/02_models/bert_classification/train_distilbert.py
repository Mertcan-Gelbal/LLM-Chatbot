#!/usr/bin/env python3
"""
DistilBERT Model Eğitimi
Tarımsal classification için optimize edilmiş DistilBERT modeli
"""

import os
import json
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()

class DistilBERTTrainer:
    """DistilBERT model eğitim sınıfı"""
    
    def __init__(self, 
                 data_dir: str = "../Data",
                 output_dir: str = "distilbert_agricultural",
                 max_length: int = 512):
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.label_encoder = LabelEncoder()
        
        # Performance tracking
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'eval_f1': []
        }
        
        console.print(f"🎮 Device: {self.device}")
        console.print(f"📁 Output: {self.output_dir}")
    
    def load_data(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Veri setlerini yükle ve hazırla"""
        console.print("📂 Veri setleri yükleniyor...", style="bold blue")
        
        # CSV dosyalarını yükle
        train_df = pd.read_csv(self.data_dir / 'train.csv')
        val_df = pd.read_csv(self.data_dir / 'val.csv')
        test_df = pd.read_csv(self.data_dir / 'test.csv')
        
        console.print(f"📊 Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Label encoding
        all_labels = pd.concat([train_df['label'], val_df['label'], test_df['label']]).unique()
        self.label_encoder.fit(all_labels)
        
        train_df['label_encoded'] = self.label_encoder.transform(train_df['label'])
        val_df['label_encoded'] = self.label_encoder.transform(val_df['label'])
        test_df['label_encoded'] = self.label_encoder.transform(test_df['label'])
        
        self.num_labels = len(self.label_encoder.classes_)
        console.print(f"🏷️  {self.num_labels} kategori: {list(self.label_encoder.classes_)}")
        
        # Label mapping kaydet
        label_mapping = {
            'label_to_id': dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_)))),
            'id_to_label': dict(zip(range(len(self.label_encoder.classes_)), self.label_encoder.classes_))
        }
        
        with open(self.output_dir / 'label_mapping.json', 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        # Tokenizer yükle
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Datasets oluştur
        train_dataset = self._prepare_dataset(train_df)
        val_dataset = self._prepare_dataset(val_df)
        test_dataset = self._prepare_dataset(test_df)
        
        return train_dataset, val_dataset, test_dataset
    
    def _prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        """DataFrame'i Dataset'e dönüştür"""
        def tokenize_function(examples):
            # Text listesini düzgün şekilde handle et
            texts = examples['text'] if isinstance(examples['text'], list) else [examples['text']]
            
            return self.tokenizer(
                texts,
                truncation=True,
                padding=True,  # Padding açık
                max_length=self.max_length,
                return_tensors=None
            )
        
        # Label sütununu düzelt
        dataset = Dataset.from_pandas(df[['text', 'label_encoded']].copy())
        dataset = dataset.rename_column('label_encoded', 'labels')
        
        # Map fonksiyonunu çalıştır
        dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=['text']  # Text sütununu kaldır
        )
        
        return dataset
    
    def create_distilbert_model(self):
        """DistilBERT modeli oluştur"""
        console.print("🤖 DistilBERT modeli oluşturuluyor...", style="bold cyan")
        
        # DistilBERT modelini yükle
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=self.num_labels
        )
        self.model.to(self.device)
        
        # Model bilgileri
        param_count = sum(p.numel() for p in self.model.parameters())
        console.print(f"📊 Model parametreleri: {param_count:,}")
        console.print(f"🔧 Hidden size: 768")
        console.print(f"📚 Layers: 6")
        console.print(f"👀 Attention heads: 12")
        console.print(f"⚡ Model type: DistilBERT (66M parameters)")
    
    def setup_trainer(self, train_dataset, val_dataset):
        """Trainer konfigürasyonu"""
        console.print("⚙️  Trainer konfigürasyonu...", style="bold yellow")
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=6,  # DistilBERT daha hızlı converge olur
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=100,
            
            eval_strategy="epoch",  # Save ile eşleştiriyorum
            eval_steps=100,
            save_strategy="no",  # DTensor hatasını çözmek için kayıt kapatıldı
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=False,  # Save strategy "no" olduğu için false
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            
            logging_steps=50,
            logging_dir=str(self.output_dir / "logs"),
            report_to=None,
            
            dataloader_num_workers=2,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=False,  # DTensor ile çakışabileceği için kapatıldı
            
            seed=42,
            data_seed=42,
            remove_unused_columns=False,
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    
    def _compute_metrics(self, eval_pred):
        """Evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1
        }
    
    def train_model(self):
        """Model eğitimi"""
        console.print("🚀 DistilBERT eğitimi başlıyor...", style="bold green")
        
        start_time = time.time()
        
        # Eğitim
        train_result = self.trainer.train()
        
        training_time = time.time() - start_time
        
        console.print(f"✅ Eğitim tamamlandı! Süre: {training_time/60:.1f} dakika")
        
        # Training history
        log_history = self.trainer.state.log_history
        self._extract_training_history(log_history)
        
        return {
            'training_time_minutes': training_time / 60,
            'final_train_loss': train_result.training_loss
        }
    
    def _extract_training_history(self, log_history):
        """Training history extract"""
        for log_entry in log_history:
            if 'train_loss' in log_entry:
                self.training_history['train_loss'].append(log_entry['train_loss'])
            if 'eval_loss' in log_entry:
                self.training_history['eval_loss'].append(log_entry['eval_loss'])
                self.training_history['eval_accuracy'].append(log_entry.get('eval_accuracy', 0))
                self.training_history['eval_f1'].append(log_entry.get('eval_f1', 0))
    
    def evaluate_model(self, test_dataset):
        """Test değerlendirmesi"""
        console.print("📊 Test değerlendirmesi...", style="bold magenta")
        
        # Test predictions
        test_results = self.trainer.evaluate(test_dataset)
        predictions = self.trainer.predict(test_dataset)
        
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=list(self.label_encoder.classes_),
            output_dict=True
        )
        
        # Results table
        table = Table(title="DistilBERT Test Sonuçları")
        table.add_column("Metrik", style="cyan")
        table.add_column("Değer", style="magenta")
        
        table.add_row("Test Accuracy", f"{test_results['eval_accuracy']:.4f}")
        table.add_row("Test F1 Score", f"{test_results['eval_f1']:.4f}")
        table.add_row("Test Loss", f"{test_results['eval_loss']:.4f}")
        
        console.print(table)
        
        return {
            'test_accuracy': test_results['eval_accuracy'],
            'test_f1': test_results['eval_f1'],
            'test_loss': test_results['eval_loss'],
            'classification_report': report
        }
    
    def save_model(self):
        """Modeli kaydet"""
        console.print("💾 Model kaydediliyor...", style="bold blue")
        
        # DTensor uyumluluğu için manuel kayıt
        try:
            # Model ve tokenizer kaydet
            self.model.save_pretrained(self.output_dir, safe_serialization=True)
            self.tokenizer.save_pretrained(self.output_dir)
        except Exception as e:
            console.print(f"⚠️  Normal kayıt hatası: {e}", style="yellow")
            console.print("🔄 Alternatif kayıt yöntemi deneniyor...", style="yellow")
            
            # Alternatif kayıt yöntemi
            import torch
            torch.save(self.model.state_dict(), self.output_dir / "pytorch_model.bin")
            
            # Config dosyası manuel kaydet
            config = self.model.config.to_dict()
            with open(self.output_dir / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            # Tokenizer kaydet
            self.tokenizer.save_pretrained(self.output_dir)
        
        # Training history kaydet
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        console.print(f"✅ Model kaydedildi: {self.output_dir}")
    
    def create_visualizations(self):
        """Eğitim görselleştirmeleri"""
        console.print("📊 Görselleştirmeler oluşturuluyor...", style="bold yellow")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DistilBERT Training Analysis', fontsize=16, fontweight='bold')
        
        # Loss curves
        if self.training_history['train_loss'] and self.training_history['eval_loss']:
            axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss', alpha=0.7)
            axes[0, 0].plot(self.training_history['eval_loss'], label='Validation Loss', alpha=0.7)
            axes[0, 0].set_title('Training & Validation Loss')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy and F1
        if self.training_history['eval_accuracy'] and self.training_history['eval_f1']:
            axes[0, 1].plot(self.training_history['eval_accuracy'], label='Accuracy', alpha=0.7)
            axes[0, 1].plot(self.training_history['eval_f1'], label='F1 Score', alpha=0.7)
            axes[0, 1].set_title('Validation Metrics')
            axes[0, 1].set_xlabel('Evaluation Steps')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Model architecture info
        axes[1, 0].text(0.1, 0.8, 'DistilBERT Architecture', fontsize=14, fontweight='bold')
        axes[1, 0].text(0.1, 0.6, f'Hidden Size: 768', fontsize=12)
        axes[1, 0].text(0.1, 0.5, f'Layers: 6', fontsize=12)
        axes[1, 0].text(0.1, 0.4, f'Attention Heads: 12', fontsize=12)
        axes[1, 0].text(0.1, 0.3, f'Parameters: ~66M', fontsize=12)
        axes[1, 0].text(0.1, 0.2, f'Speed: 40% faster than BERT', fontsize=12)
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')
        
        # Performance summary
        if self.training_history['eval_accuracy']:
            final_acc = self.training_history['eval_accuracy'][-1]
            final_f1 = self.training_history['eval_f1'][-1]
            
            axes[1, 1].text(0.1, 0.8, 'Final Performance', fontsize=14, fontweight='bold')
            axes[1, 1].text(0.1, 0.6, f'Validation Accuracy: {final_acc:.4f}', fontsize=12)
            axes[1, 1].text(0.1, 0.5, f'Validation F1: {final_f1:.4f}', fontsize=12)
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"📊 Görselleştirme kaydedildi: {self.output_dir / 'training_analysis.png'}")

    def save_training_metrics_table(self):
        """Eğitim metriklerini güzel tablo formatında kaydet"""
        console.print("📊 Eğitim metrikleri kaydediliyor...", style="bold cyan")
        
        # Training metrics table
        if self.training_history['train_loss']:
            metrics_table = Table(title="DistilBERT Eğitim Metrikleri")
            metrics_table.add_column("Step", style="cyan")
            metrics_table.add_column("Train Loss", style="red")
            metrics_table.add_column("Eval Loss", style="yellow")
            metrics_table.add_column("Eval Accuracy", style="green")
            metrics_table.add_column("Eval F1", style="magenta")
            
            # Training history'yi step bazlı organize et
            max_len = max(
                len(self.training_history['train_loss']),
                len(self.training_history['eval_loss'])
            )
            
            for i in range(min(20, max_len)):  # Son 20 entry
                step = (i + 1) * 50  # logging_steps = 50
                
                train_loss = f"{self.training_history['train_loss'][i]:.4f}" if i < len(self.training_history['train_loss']) else "-"
                eval_loss = f"{self.training_history['eval_loss'][i]:.4f}" if i < len(self.training_history['eval_loss']) else "-"
                eval_acc = f"{self.training_history['eval_accuracy'][i]:.4f}" if i < len(self.training_history['eval_accuracy']) else "-"
                eval_f1 = f"{self.training_history['eval_f1'][i]:.4f}" if i < len(self.training_history['eval_f1']) else "-"
                
                metrics_table.add_row(str(step), train_loss, eval_loss, eval_acc, eval_f1)
            
            console.print(metrics_table)
            
            # CSV olarak da kaydet
            metrics_df = pd.DataFrame({
                'step': [(i + 1) * 50 for i in range(len(self.training_history['train_loss']))],
                'train_loss': self.training_history['train_loss'],
                'eval_loss': self.training_history['eval_loss'][:len(self.training_history['train_loss'])],
                'eval_accuracy': self.training_history['eval_accuracy'][:len(self.training_history['train_loss'])],
                'eval_f1': self.training_history['eval_f1'][:len(self.training_history['train_loss'])]
            })
            
            csv_path = self.output_dir / 'training_metrics.csv'
            metrics_df.to_csv(csv_path, index=False)
            console.print(f"📄 Eğitim metrikleri CSV olarak kaydedildi: {csv_path}")

def main():
    """Ana eğitim fonksiyonu"""
    console.print("🤖 DistilBERT Tarımsal Model Eğitimi", style="bold green")
    console.print("=" * 60)
    
    trainer = DistilBERTTrainer()
    
    try:
        # 1. Veri yükleme
        train_dataset, val_dataset, test_dataset = trainer.load_data()
        
        # 2. Model oluşturma
        trainer.create_distilbert_model()
        
        # 3. Trainer kurulumu
        trainer.setup_trainer(train_dataset, val_dataset)
        
        # 4. Model eğitimi
        training_metrics = trainer.train_model()
        
        # 5. Test değerlendirmesi
        test_metrics = trainer.evaluate_model(test_dataset)
        
        # 6. Model kaydetme
        trainer.save_model()
        
        # 7. Görselleştirmeler
        trainer.create_visualizations()
        
        # 8. Eğitim metrikleri kaydet
        trainer.save_training_metrics_table()
        
        # Final özet
        console.print("\n✅ DistilBERT eğitimi başarıyla tamamlandı!", style="bold green")
        console.print(f"🎯 Test Accuracy: {test_metrics['test_accuracy']:.4f}")
        console.print(f"📊 Test F1 Score: {test_metrics['test_f1']:.4f}")
        console.print(f"⏱️  Eğitim Süresi: {training_metrics['training_time_minutes']:.1f} dakika")
        
    except Exception as e:
        console.print(f"❌ Eğitim sırasında hata: {e}", style="bold red")
        raise

if __name__ == "__main__":
    main() 