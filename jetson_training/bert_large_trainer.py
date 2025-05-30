"""
🌾 Agricultural BERT-Large Classification Trainer
High-performance BERT-Large model for research and server deployment
"""

import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    model_name: str
    batch_size: int
    learning_rate: float
    num_epochs: int
    max_length: int
    min_gpu_memory: float  # GB
    expected_accuracy: str
    use_case: str

class BERTLargeTrainer:
    """Agricultural BERT-Large Specialized Trainer"""
    
    def __init__(self, model_name: str = "bert-large-uncased", batch_size: int = 4, 
                 num_epochs: int = 3, learning_rate: float = 1e-5, max_length: int = 512):
        """Initialize BERT-Large trainer"""
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        
        # Agricultural categories
        self.categories = [
            "plant_disease", "crop_management", "plant_genetics", 
            "environmental_factors", "food_security", "technology"
        ]
        
        # BERT-Large specific configuration
        self.model_config = ModelConfig(
            model_name="bert-large-uncased",
            batch_size=4,
            learning_rate=1e-5,
            num_epochs=3,
            max_length=512,
            min_gpu_memory=6.0,
            expected_accuracy="89-92%",
            use_case="Research/Server"
        )
        
        # Create directories
        os.makedirs("../checkpoints", exist_ok=True)
        os.makedirs("../models", exist_ok=True)
        os.makedirs("../results", exist_ok=True)
        os.makedirs("../logs", exist_ok=True)
        
        self.logger = logging.getLogger(f"BERTLargeTrainer")
        
    def load_agricultural_data(self) -> DatasetDict:
        """Load agricultural dataset"""
        self.logger.info("📊 Loading agricultural dataset...")
        
        # Load data files
        try:
            train_df = pd.read_csv("../agricultural_datasets/train.csv")
            val_df = pd.read_csv("../agricultural_datasets/val.csv") 
            test_df = pd.read_csv("../agricultural_datasets/test.csv")
            
            self.logger.info(f"✅ Loaded datasets - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
        except FileNotFoundError:
            self.logger.error("❌ Dataset files not found! Please run unified_comprehensive_indexer.py first")
            raise
            
        # Create label mapping
        unique_labels = sorted(train_df['category'].unique())
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        # Convert to datasets
        def prepare_dataset(df):
            return Dataset.from_dict({
                'text': df['text'].tolist(),
                'labels': [self.label2id[cat] for cat in df['category'].tolist()]
            })
            
        dataset = DatasetDict({
            'train': prepare_dataset(train_df),
            'validation': prepare_dataset(val_df),
            'test': prepare_dataset(test_df)
        })
        
        return dataset
        
    def check_system_requirements(self) -> bool:
        """Check if system meets BERT-Large requirements"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"🔧 Available GPU Memory: {gpu_memory:.1f}GB")
            self.logger.info(f"🎯 Required GPU Memory: {self.model_config.min_gpu_memory}GB")
            
            if gpu_memory < self.model_config.min_gpu_memory:
                self.logger.error(f"❌ Insufficient GPU memory for BERT-Large")
                self.logger.error(f"   Required: {self.model_config.min_gpu_memory}GB, Available: {gpu_memory:.1f}GB")
                return False
                
            # Adjust batch size based on available memory
            if gpu_memory < self.model_config.min_gpu_memory + 2:
                self.batch_size = max(2, self.model_config.batch_size // 2)
                self.logger.warning(f"⚠️  Reduced batch size to {self.batch_size} due to memory constraints")
            else:
                self.batch_size = self.model_config.batch_size
                
        return True
        
    def tokenize_function(self, examples):
        """Tokenize text examples"""
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
        
    def setup_bert_large_optimizations(self):
        """Setup BERT-Large specific optimizations"""
        self.logger.info("🧠 Applying BERT-Large-specific optimizations")
        
        # Lower learning rate for large models
        if self.learning_rate > 1e-5:
            self.learning_rate = 1e-5
            self.logger.info(f"📉 Reduced learning rate to {self.learning_rate} for BERT-Large")
            
        # Increase warmup steps for large models
        self.warmup_steps = 1000
        self.logger.info(f"📈 Increased warmup steps to {self.warmup_steps} for BERT-Large")
                
    def train_model(self, dataset: DatasetDict) -> Tuple[Trainer, Dict]:
        """Train BERT-Large model"""
        self.logger.info(f"🚀 Starting BERT-Large training")
        
        # Check system requirements
        if not self.check_system_requirements():
            raise RuntimeError("System requirements not met for BERT-Large")
            
        # Setup BERT-Large optimizations
        self.setup_bert_large_optimizations()
        
        # Load tokenizer and model
        self.logger.info("📥 Loading BERT-Large tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.categories)
        )
        
        # Model info
        num_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"🔢 BERT-Large parameters: {num_params:,}")
        self.logger.info(f"📊 Expected parameter count: ~340M")
        
        # Tokenize datasets
        self.logger.info("🔧 Tokenizing datasets...")
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Training arguments optimized for BERT-Large
        training_args = TrainingArguments(
            output_dir=f"../checkpoints/bert_large_uncased",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=0.01,
            logging_dir=f"../logs/bert_large_uncased",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            dataloader_pin_memory=False,
            fp16=True,  # Mixed precision for memory efficiency
            gradient_checkpointing=True,  # Memory optimization
            report_to=None,  # Disable wandb/tensorboard for cleaner output
            # BERT-Large specific settings
            gradient_accumulation_steps=2,  # Effective batch size = batch_size * 2
            max_grad_norm=1.0,  # Gradient clipping
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train model
        start_time = datetime.now()
        self.logger.info(f"⏰ BERT-Large training started at {start_time}")
        
        # Training with memory monitoring
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        trainer.train()
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            self.logger.info(f"🔥 Peak GPU memory usage: {peak_memory:.2f}GB")
        
        end_time = datetime.now()
        training_time = end_time - start_time
        self.logger.info(f"✅ BERT-Large training completed in {training_time}")
        
        # Evaluate on test set
        self.logger.info("📊 Evaluating BERT-Large on test set...")
        test_results = trainer.evaluate(tokenized_datasets["test"])
        self.logger.info(f"📈 BERT-Large Test Results: {test_results}")
        
        # Save model
        model_path = f"../models/bert_large_uncased_agricultural"
        self.logger.info(f"💾 Saving BERT-Large model to {model_path}")
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        return trainer, test_results
        
    def create_detailed_analysis(self, trainer: Trainer, dataset: Dataset, results: Dict):
        """Create detailed performance analysis for BERT-Large"""
        self.logger.info("📊 Creating BERT-Large detailed analysis...")
        
        # Predictions for detailed analysis
        predictions = trainer.predict(dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.categories,
            output_dict=True
        )
        
        # Save classification report
        report_df = pd.DataFrame(report).transpose()
        report_file = f"../results/classification_report_bert_large.csv"
        report_df.to_csv(report_file)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.categories,
                   yticklabels=self.categories,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - BERT-Large\nAccuracy: {results["eval_accuracy"]:.3f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_file = f"../results/confusion_matrix_bert_large.png"
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Per-category performance
        plt.figure(figsize=(12, 8))
        categories_short = [cat.replace('_', ' ').title() for cat in self.categories]
        f1_scores = [report[cat]['f1-score'] for cat in self.categories]
        
        bars = plt.bar(categories_short, f1_scores, color='darkblue', alpha=0.8)
        plt.title(f'Per-Category F1 Scores - BERT-Large')
        plt.ylabel('F1 Score')
        plt.xlabel('Category')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
                    
        plt.tight_layout()
        f1_file = f"../results/f1_scores_bert_large.png"
        plt.savefig(f1_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 BERT-Large analysis saved: {report_file}, {cm_file}, {f1_file}")
        
    def run_bert_large_experiment(self):
        """Run complete BERT-Large experiment"""
        self.logger.info(f"🌾 Starting BERT-Large Agricultural Experiment")
        
        # Load data
        dataset = self.load_agricultural_data()
        
        # Train model
        trainer, results = self.train_model(dataset)
        
        # Create detailed analysis
        self.create_detailed_analysis(trainer, dataset["test"], results)
        
        # Save comprehensive results
        model_info = {
            'model': 'bert-large-uncased',
            'parameters': sum(p.numel() for p in trainer.model.parameters()),
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'warmup_steps': self.warmup_steps,
            'timestamp': datetime.now().isoformat(),
            **results
        }
        
        results_df = pd.DataFrame([model_info])
        results_file = f"../results/results_bert_large.csv"
        results_df.to_csv(results_file, index=False)
        
        # Print summary
        print(f"\n🎯 BERT-Large Training Summary:")
        print(f"   📊 Test Accuracy: {results['eval_accuracy']:.4f}")
        print(f"   📈 Test F1: {results['eval_f1']:.4f}")
        print(f"   🎯 Expected Accuracy: {self.model_config.expected_accuracy}")
        print(f"   🎮 Use Case: {self.model_config.use_case}")
        print(f"   🔢 Parameters: {model_info['parameters']:,}")
        print(f"   💾 Model saved to: ../models/bert_large_uncased_agricultural")
        
        self.logger.info("✅ BERT-Large experiment completed!")
        
        return results

def main():
    """Main function for BERT-Large training"""
    print("🌾 Agricultural BERT-Large Specialized Training")
    print("=" * 60)
    
    try:
        trainer = BERTLargeTrainer()
        results = trainer.run_bert_large_experiment()
        
        print(f"\n🏆 BERT-Large Final Results:")
        print(f"   📊 Accuracy: {results['eval_accuracy']:.4f}")
        print(f"   📈 F1-Score: {results['eval_f1']:.4f}")
        print(f"   ✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error training BERT-Large: {str(e)}")
        raise
    
    print("\n✅ BERT-Large experiment completed!")

if __name__ == "__main__":
    main() 