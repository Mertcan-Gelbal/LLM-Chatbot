#!/usr/bin/env python3
"""
Botanical Expert BERT Trainer - Small Model Optimized
====================================================
Bitki bilimi uzmanı chatbot için optimize edilmiş küçük BERT modeli
"""

import torch
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup, BertConfig, TFBertForSequenceClassification
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# JetPack 6.2 optimizations
from gpu_optimizer_jp62 import JetsonOptimizerJP62

class BotanicalBERTDataset(Dataset):
    """Bitki bilimi için optimize edilmiş BERT dataset"""
    
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Botanical terms için daha detaylı tokenization
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

class BotanicalBERTTrainer:
    """Bitki bilimi uzmanı için küçük BERT eğitim sistemi"""
    
    def __init__(self, mixed_precision=True, max_length=256, small_model=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mixed_precision = mixed_precision
        self.max_length = max_length
        self.small_model = small_model
        
        # JetPack 6.2 optimizer
        self.jetson_opt = JetsonOptimizerJP62()
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=mixed_precision)
        
        # Results storage
        self.results = []
        
        # Bitki bilimi kategorileri
        self.botanical_categories = {
            'plant_disease': 'Bitki Hastalıkları',
            'crop_management': 'Mahsul Yönetimi', 
            'plant_genetics': 'Bitki Genetiği',
            'environmental_factors': 'Çevre Faktörleri',
            'food_security': 'Gıda Güvenliği',
            'technology': 'Tarım Teknolojisi'
        }
        
        print(f"🌱 Botanical BERT Trainer başlatıldı")
        print(f"📱 Device: {self.device}")
        print(f"🔧 Mixed Precision: {mixed_precision}")
        print(f"📏 Max Length: {max_length}")
        print(f"🤖 Small Model: {small_model}")
        
        # Monitor initial GPU state
        self.jetson_opt.monitor_jetson_gpu()
    
    def create_small_bert_config(self, num_labels):
        """Küçük BERT modeli konfigürasyonu"""
        if self.small_model:
            # Küçük ve hızlı model konfigürasyonu
            config = BertConfig(
                vocab_size=30522,
                hidden_size=384,          # Orijinal: 768
                num_hidden_layers=6,      # Orijinal: 12  
                num_attention_heads=6,    # Orijinal: 12
                intermediate_size=1536,   # Orijinal: 3072
                max_position_embeddings=512,
                num_labels=num_labels,
                attention_probs_dropout_prob=0.1,
                hidden_dropout_prob=0.1
            )
            print(f"🤖 Küçük BERT model konfigürasyonu oluşturuldu")
            print(f"   - Hidden Size: {config.hidden_size}")
            print(f"   - Layers: {config.num_hidden_layers}")
            print(f"   - Attention Heads: {config.num_attention_heads}")
            return config
        else:
            # Standart BERT-base config
            return BertConfig.from_pretrained('bert-base-uncased', num_labels=num_labels)
    
    def load_botanical_data(self):
        """Bitki bilimi veri setini yükle"""
        print(f"🌿 Bitki bilimi veri seti yükleniyor...")
        
        try:
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
                raise FileNotFoundError("Bitki bilimi dataset dosyaları bulunamadı")
            
            # 'general_agriculture' kategorisini filtrele - çok az veri var
            train_df = train_df[train_df['label'] != 'general_agriculture']
            val_df = val_df[val_df['label'] != 'general_agriculture']
            test_df = test_df[test_df['label'] != 'general_agriculture']
            
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
            
            print(f"✅ Botanical Dataset: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")
            print(f"🌱 Categories: {list(id2label.values())}")
            
            # Kategori dağılımını göster
            self.show_category_distribution(train_df, val_df, test_df)
            
            return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, id2label
            
        except Exception as e:
            print(f"❌ Bitki bilimi veri yükleme hatası: {e}")
            return None
    
    def show_category_distribution(self, train_df, val_df, test_df):
        """Kategori dağılımını göster"""
        print(f"\n📊 Kategori Dağılımı:")
        
        # Train set dağılımı
        train_dist = train_df['label'].value_counts()
        print(f"🏋️ Training Set:")
        for category, count in train_dist.items():
            print(f"   {self.botanical_categories.get(category, category)}: {count}")
        
        # Total statistics
        total_samples = len(train_df) + len(val_df) + len(test_df)
        print(f"\n📈 Toplam Veri: {total_samples} sample")
        print(f"   Training: {len(train_df)} ({len(train_df)/total_samples*100:.1f}%)")
        print(f"   Validation: {len(val_df)} ({len(val_df)/total_samples*100:.1f}%)")
        print(f"   Test: {len(test_df)} ({len(test_df)/total_samples*100:.1f}%)")
    
    def create_data_loaders(self, train_texts, val_texts, test_texts, 
                           train_labels, val_labels, test_labels, tokenizer, batch_size=8):
        """Data loader'ları oluştur"""
        
        train_dataset = BotanicalBERTDataset(train_texts, train_labels, tokenizer, self.max_length)
        val_dataset = BotanicalBERTDataset(val_texts, val_labels, tokenizer, self.max_length)
        test_dataset = BotanicalBERTDataset(test_texts, test_labels, tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, train_loader, optimizer, scheduler, epoch):
        """Tek epoch eğitimi"""
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=self.mixed_precision):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            total_loss += loss.item()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Progress bar update
            accuracy = correct_predictions / total_predictions
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.3f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = correct_predictions / total_predictions
        
        return avg_loss, avg_accuracy
    
    def evaluate_model(self, model, data_loader, dataset_name=""):
        """Model değerlendirme"""
        model.eval()
        all_preds, all_labels = [], []
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc=f"Evaluating {dataset_name}")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with autocast(enabled=self.mixed_precision):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'f1_score': f1,
            'predictions': all_preds,
            'true_labels': all_labels
        }
    
    def train_botanical_bert(self, epochs=4, lr=2e-5, batch_size=12):
        """Ana bitki bilimi BERT eğitimi"""
        print(f"🌱 Botanical BERT Eğitimi Başlıyor...")
        print(f"📊 Epochs: {epochs}, LR: {lr}, Batch Size: {batch_size}")
        
        start_time = time.time()
        
        try:
            # Veri yükle
            data = self.load_botanical_data()
            if data is None:
                return None
                
            train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, id2label = data
            num_labels = len(id2label)
            
            # Tokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            # Data loaders
            train_loader, val_loader, test_loader = self.create_data_loaders(
                train_texts, val_texts, test_texts, train_labels, val_labels, test_labels,
                tokenizer, batch_size=batch_size
            )
            
            # Model oluştur
            if self.small_model:
                config = self.create_small_bert_config(num_labels)
                model = BertForSequenceClassification(config)
                print(f"🤖 Küçük BERT modeli oluşturuldu")
            else:
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
                print(f"🤖 Standart BERT modeli yüklendi")
            
            model.to(self.device)
            
            # Model parametrelerini göster
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"📊 Model Parametreleri:")
            print(f"   Total: {total_params:,}")
            print(f"   Trainable: {trainable_params:,}")
            
            # Optimizer ve scheduler
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps
            )
            
            # Training history
            train_losses, train_accuracies = [], []
            val_losses, val_accuracies = [], []
            
            best_val_accuracy = 0
            best_model_state = None
            
            print(f"\n🚀 Eğitim başlıyor...")
            
            for epoch in range(epochs):
                print(f"\n📚 Epoch {epoch+1}/{epochs}")
                print("-" * 50)
                
                # Training
                train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, scheduler, epoch)
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                
                # Validation
                val_metrics = self.evaluate_model(model, val_loader, "Validation")
                val_losses.append(val_metrics['loss'])
                val_accuracies.append(val_metrics['accuracy'])
                
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
                print(f"Val F1: {val_metrics['f1_score']:.4f}, Val Precision: {val_metrics['precision']:.4f}")
                
                # Best model kaydet
                if val_metrics['accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['accuracy']
                    best_model_state = model.state_dict().copy()
                    print(f"🎯 En iyi model güncellendi! Val Acc: {best_val_accuracy:.4f}")
                
                # GPU monitoring
                if epoch % 1 == 0:
                    self.jetson_opt.monitor_jetson_gpu()
            
            # En iyi modeli yükle
            model.load_state_dict(best_model_state)
            
            # Final test evaluation
            print(f"\n🎯 Final Test Evaluation")
            print("-" * 30)
            test_metrics = self.evaluate_model(model, test_loader, "Test")
            
            print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
            print(f"Test Precision: {test_metrics['precision']:.4f}")
            
            # Classification report
            print(f"\n📊 Detailed Classification Report:")
            class_names = [id2label[i] for i in range(num_labels)]
            report = classification_report(test_metrics['true_labels'], test_metrics['predictions'], 
                                         target_names=class_names, digits=4)
            print(report)
            
            # Modeli kaydet
            model_dir = Path("../models/botanical_bert_small" if self.small_model else "../models/botanical_bert_base")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Manual save to avoid DTensor issue
            try:
                torch.save(model.state_dict(), model_dir / "pytorch_model.bin")
                tokenizer.save_pretrained(model_dir)
                
                # Save config manually
                if self.small_model:
                    config = self.create_small_bert_config(num_labels)
                    config.save_pretrained(model_dir)
                else:
                    model.config.save_pretrained(model_dir)
                
                print(f"✅ Model manuel olarak kaydedildi")
            except Exception as save_error:
                print(f"⚠️ Model kaydetme hatası (devam ediliyor): {save_error}")
                # Continue without saving the model files
            
            # Config kaydet
            config_path = model_dir / "training_config.json"
            training_config = {
                'model_type': 'small_bert' if self.small_model else 'bert_base',
                'num_labels': num_labels,
                'id2label': id2label,
                'label2id': {v: k for k, v in id2label.items()},
                'max_length': self.max_length,
                'batch_size': batch_size,
                'learning_rate': lr,
                'epochs': epochs,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'best_val_accuracy': best_val_accuracy,
                'test_accuracy': test_metrics['accuracy'],
                'test_f1_score': test_metrics['f1_score'],
                'training_time': time.time() - start_time
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(training_config, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Model kaydedildi: {model_dir}")
            print(f"⏱️ Eğitim süresi: {time.time() - start_time:.2f} saniye")
            
            # Grafikler oluştur
            self.plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
            self.plot_confusion_matrix(test_metrics['true_labels'], test_metrics['predictions'], class_names)
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'config': training_config,
                'test_metrics': test_metrics
            }
            
        except Exception as e:
            print(f"❌ Eğitim hatası: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_training_history(self, train_losses, train_accuracies, val_losses, val_accuracies):
        """Eğitim geçmişi grafikleri"""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Combined metrics
        plt.subplot(1, 3, 3)
        plt.plot(epochs, train_accuracies, 'b-', label='Train Acc', alpha=0.7)
        plt.plot(epochs, val_accuracies, 'r-', label='Val Acc', alpha=0.7)
        plt.fill_between(epochs, train_accuracies, alpha=0.3, color='blue')
        plt.fill_between(epochs, val_accuracies, alpha=0.3, color='red')
        plt.title('Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / "botanical_bert_training_history.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Eğitim grafikleri kaydedildi: {results_dir}/botanical_bert_training_history.png")
    
    def plot_confusion_matrix(self, true_labels, predictions, class_names):
        """Confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Botanical BERT - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        results_dir = Path("../results")
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / "botanical_bert_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Confusion matrix kaydedildi: {results_dir}/botanical_bert_confusion_matrix.png")

    def save_model_h5_format(self, model, tokenizer, save_path: str, metrics: Dict = None):
        """
        Model'i H5 formatında kaydet (TensorFlow/Keras uyumluluğu için)
        """
        print("\n💾 H5 FORMATINDA KAYDETME")
        print("="*40)
        
        try:
            import tensorflow as tf
            from transformers import TFBertForSequenceClassification
            
            # PyTorch'dan TensorFlow'a dönüştürme
            print("🔄 PyTorch → TensorFlow dönüşümü...")
            
            # Model konfigürasyonunu al
            config = model.config
            
            # TensorFlow modeli oluştur
            tf_model = TFBertForSequenceClassification.from_pretrained(
                save_path,
                from_tf=False,
                config=config
            )
            
            # H5 formatında kaydet
            h5_path = os.path.join(save_path, "model.h5")
            tf_model.save_weights(h5_path)
            
            # Model bilgilerini kaydet
            model_info = {
                'model_format': 'h5',
                'framework': 'tensorflow',
                'model_type': 'bert_for_sequence_classification',
                'num_labels': config.num_labels,
                'vocab_size': config.vocab_size,
                'hidden_size': config.hidden_size,
                'num_layers': config.num_hidden_layers,
                'num_attention_heads': config.num_attention_heads,
                'created_date': datetime.now().isoformat()
            }
            
            if metrics:
                model_info.update({
                    'accuracy': float(metrics.get('accuracy', 0)),
                    'f1_score': float(metrics.get('f1_score', 0)),
                    'precision': float(metrics.get('precision', 0))
                })
            
            # JSON olarak kaydet
            info_path = os.path.join(save_path, "model_info_h5.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            print(f"✅ H5 model kaydedildi: {h5_path}")
            print(f"📊 Model bilgileri: {info_path}")
            
            # Dosya boyutlarını göster
            h5_size = os.path.getsize(h5_path) / 1024 / 1024
            print(f"📦 H5 dosya boyutu: {h5_size:.1f}MB")
            
            return True
            
        except ImportError:
            print("⚠️ TensorFlow bulunamadı. H5 formatı için TensorFlow gerekli.")
            print("💡 Kurulum: pip install tensorflow")
            return False
        except Exception as e:
            print(f"❌ H5 dönüştürme hatası: {e}")
            return False

    def export_multiple_formats(self, model, tokenizer, base_path: str, metrics: Dict = None):
        """
        Model'i birden fazla formatta dışa aktar
        """
        print("\n📦 ÇOK FORMATLI MODEL İHRACATI")
        print("="*45)
        
        export_results = {}
        
        # 1. PyTorch format (varsayılan)
        try:
            self.save_model_manual(model, tokenizer, base_path, metrics)
            export_results['pytorch'] = True
            print("✅ PyTorch formatı başarılı")
        except Exception as e:
            print(f"❌ PyTorch formatı hatası: {e}")
            export_results['pytorch'] = False
        
        # 2. H5 format
        try:
            h5_success = self.save_model_h5_format(model, tokenizer, base_path, metrics)
            export_results['h5'] = h5_success
            if h5_success:
                print("✅ H5 formatı başarılı")
        except Exception as e:
            print(f"❌ H5 formatı hatası: {e}")
            export_results['h5'] = False
        
        # 3. ONNX format (isteğe bağlı)
        try:
            onnx_path = os.path.join(base_path, "model.onnx")
            # ONNX export burada implement edilebilir
            print("💡 ONNX formatı: Gelecek sürümlerde eklenecek")
            export_results['onnx'] = False
        except Exception as e:
            export_results['onnx'] = False
        
        # 4. Model özet dosyası
        try:
            summary_path = os.path.join(base_path, "export_summary.json")
            summary_data = {
                'export_date': datetime.now().isoformat(),
                'formats': export_results,
                'model_info': {
                    'type': 'botanical_bert_small',
                    'parameters': '22M',
                    'categories': len(self.id2label) if hasattr(self, 'id2label') else 6
                },
                'performance': metrics if metrics else {}
            }
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            print(f"📊 Export özeti: {summary_path}")
            
        except Exception as e:
            print(f"⚠️ Özet dosyası oluşturulamadı: {e}")
        
        # Sonuçları göster
        print(f"\n📋 EXPORT SONUÇLARI")
        print("-" * 25)
        for format_name, success in export_results.items():
            status = "✅" if success else "❌"
            print(f"{status} {format_name.upper()}: {'Başarılı' if success else 'Başarısız'}")
        
        return export_results

def main():
    """Ana fonksiyon"""
    print("🌱 Botanical Expert BERT Training")
    print("=" * 50)
    
    # Küçük model ile eğitim
    trainer = BotanicalBERTTrainer(mixed_precision=True, max_length=256, small_model=True)
    
    print("\n🚀 Küçük BERT modeli eğitimi başlıyor...")
    result = trainer.train_botanical_bert(epochs=4, lr=2e-5, batch_size=12)
    
    if result:
        print("\n🎉 Botanical BERT eğitimi başarıyla tamamlandı!")
        print(f"🎯 Test Accuracy: {result['test_metrics']['accuracy']:.4f}")
        print(f"🎯 Test F1 Score: {result['test_metrics']['f1_score']:.4f}")
    else:
        print("\n❌ Eğitim başarısız oldu!")
    
    # Final GPU monitoring
    trainer.jetson_opt.monitor_jetson_gpu()

if __name__ == '__main__':
    main() 