#!/usr/bin/env python3
"""
🌾 Agricultural BERT Classification Trainer
Jetson Orin Nano Super için optimize edilmiş BERT eğitim sistemi
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

# ML Libraries
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Jetson optimizations
import psutil
import gc

class JetsonBERTTrainer:
    """
    Jetson Orin Nano Super için optimize edilmiş BERT trainer
    """
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 max_length: int = 512,
                 batch_size: int = 8,
                 learning_rate: float = 2e-5,
                 num_epochs: int = 3):
        
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Jetson optimizations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logging()
        self.setup_directories()
        
        # Agricultural categories
        self.categories = [
            "Plant Disease",
            "Crop Management", 
            "Plant Genetics",
            "Environmental Factors",
            "Food Security",
            "Technology"
        ]
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.categories)
        
        print(f"🚀 JetsonBERTTrainer initialized")
        print(f"📱 Device: {self.device}")
        print(f"🧠 Model: {self.model_name}")
        print(f"📊 Categories: {len(self.categories)}")
        
    def setup_logging(self):
        """Logging setup"""
        log_dir = Path("../logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = ["../results", "../models", "../checkpoints", "../logs"]
        for dir_path in dirs:
            Path(dir_path).mkdir(exist_ok=True)
            
    def load_agricultural_data(self) -> DatasetDict:
        """Load agricultural datasets"""
        try:
            # Load datasets
            train_df = pd.read_csv("../agricultural_datasets/train.csv")
            val_df = pd.read_csv("../agricultural_datasets/val.csv") 
            test_df = pd.read_csv("../agricultural_datasets/test.csv")
            
            self.logger.info(f"📊 Loaded datasets:")
            self.logger.info(f"   Train: {len(train_df)} samples")
            self.logger.info(f"   Val: {len(val_df)} samples")
            self.logger.info(f"   Test: {len(test_df)} samples")
            
            # Encode labels
            train_df['labels'] = self.label_encoder.transform(train_df['category'])
            val_df['labels'] = self.label_encoder.transform(val_df['category'])
            test_df['labels'] = self.label_encoder.transform(test_df['category'])
            
            # Create datasets
            train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
            val_dataset = Dataset.from_pandas(val_df[['text', 'labels']])
            test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])
            
            return DatasetDict({
                'train': train_dataset,
                'validation': val_dataset,
                'test': test_dataset
            })
            
        except Exception as e:
            self.logger.error(f"❌ Error loading data: {e}")
            raise
            
    def tokenize_function(self, examples):
        """Tokenize text data"""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
    def optimize_for_jetson(self):
        """Jetson-specific optimizations"""
        # Memory optimization
        torch.cuda.empty_cache()
        gc.collect()
        
        # Set optimal batch size based on available memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"🔧 GPU Memory: {gpu_memory:.1f}GB")
            
            if gpu_memory < 4:
                self.batch_size = 4
            elif gpu_memory < 6:
                self.batch_size = 6
            else:
                self.batch_size = 8
                
        self.logger.info(f"⚡ Optimized batch size: {self.batch_size}")
        
    def train_model(self, dataset: DatasetDict) -> Tuple[Trainer, Dict]:
        """Train BERT model"""
        self.logger.info(f"🚀 Starting training with {self.model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.categories)
        )
        
        # Tokenize datasets
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Optimize for Jetson
        self.optimize_for_jetson()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"../checkpoints/{self.model_name.replace('/', '_')}",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"../logs/{self.model_name.replace('/', '_')}",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            dataloader_pin_memory=False,  # Jetson optimization
            fp16=True,  # Mixed precision for Jetson
            gradient_checkpointing=True,  # Memory optimization
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train model
        start_time = datetime.now()
        self.logger.info(f"⏰ Training started at {start_time}")
        
        trainer.train()
        
        end_time = datetime.now()
        training_time = end_time - start_time
        self.logger.info(f"✅ Training completed in {training_time}")
        
        # Evaluate on test set
        test_results = trainer.evaluate(tokenized_datasets["test"])
        self.logger.info(f"📊 Test Results: {test_results}")
        
        # Save model
        model_path = f"../models/{self.model_name.replace('/', '_')}_agricultural"
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        return trainer, test_results
        
    def create_confusion_matrix(self, trainer: Trainer, dataset: Dataset):
        """Create and save confusion matrix"""
        predictions = trainer.predict(dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.categories,
                   yticklabels=self.categories)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(f"../results/confusion_matrix_{self.model_name.replace('/', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Confusion matrix saved")
        
    def run_agricultural_experiments(self):
        """Run complete agricultural BERT experiments"""
        self.logger.info("🌾 Starting Agricultural BERT Experiments")
        
        # Load data
        dataset = self.load_agricultural_data()
        
        # Train model
        trainer, results = self.train_model(dataset)
        
        # Create confusion matrix
        self.create_confusion_matrix(trainer, dataset["test"])
        
        # Save results
        results_df = pd.DataFrame([results])
        results_df['model'] = self.model_name
        results_df['timestamp'] = datetime.now()
        results_df.to_csv(f"../results/results_{self.model_name.replace('/', '_')}.csv", index=False)
        
        self.logger.info("✅ Agricultural BERT experiments completed!")
        
        return results

def main():
    """Main function"""
    print("🌾 Agricultural BERT Classification Trainer")
    print("=" * 50)
    
    # BERT-base experiment
    print("\n🔬 Training BERT-base-uncased...")
    trainer_base = JetsonBERTTrainer(
        model_name="bert-base-uncased",
        batch_size=8,
        num_epochs=3
    )
    results_base = trainer_base.run_agricultural_experiments()
    
    # Memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # BERT-large experiment (if enough memory)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory >= 6:
            print("\n🔬 Training BERT-large-uncased...")
            trainer_large = JetsonBERTTrainer(
                model_name="bert-large-uncased",
                batch_size=4,
                num_epochs=3
            )
            results_large = trainer_large.run_agricultural_experiments()
        else:
            print("⚠️  Insufficient GPU memory for BERT-large")
    
    print("\n✅ All experiments completed!")

if __name__ == "__main__":
    main() 