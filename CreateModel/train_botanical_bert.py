#!/usr/bin/env python3
"""
Botanical BERT Training Launcher
===============================
Bitki bilimi BERT modelini kolayca eğitmek için başlatıcı script
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from botanical_bert_trainer import BotanicalBERTTrainer
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    print("🔍 Gerekli dosyaların mevcut olduğundan emin olun:")
    print("   - botanical_bert_trainer.py")
    print("   - gpu_optimizer_jp62.py")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Botanical BERT Model Eğitimi')
    
    # Model parametreleri
    parser.add_argument('--small-model', action='store_true', default=True,
                      help='Küçük BERT modeli kullan (varsayılan: True)')
    parser.add_argument('--full-model', action='store_true', 
                      help='Tam BERT-base modeli kullan')
    
    # Eğitim parametreleri
    parser.add_argument('--epochs', type=int, default=4,
                      help='Epoch sayısı (varsayılan: 4)')
    parser.add_argument('--batch-size', type=int, default=12,
                      help='Batch size (varsayılan: 12)')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                      help='Learning rate (varsayılan: 2e-5)')
    parser.add_argument('--max-length', type=int, default=256,
                      help='Maksimum token uzunluğu (varsayılan: 256)')
    
    # Sistem parametreleri  
    parser.add_argument('--no-mixed-precision', action='store_true',
                      help='Mixed precision training kapatır')
    
    # Preset konfigürasyonlar
    parser.add_argument('--fast', action='store_true',
                      help='Hızlı eğitim (2 epoch, küçük model)')
    parser.add_argument('--accurate', action='store_true',
                      help='Doğru eğitim (6 epoch, tam model)')
    
    args = parser.parse_args()
    
    # Preset configurations
    if args.fast:
        print("🚀 Hızlı Eğitim Modu")
        args.epochs = 2
        args.batch_size = 16
        args.small_model = True
        args.max_length = 128
    
    elif args.accurate:
        print("🎯 Doğru Eğitim Modu") 
        args.epochs = 6
        args.batch_size = 8
        args.full_model = True
        args.max_length = 512
    
    # Model türü belirle
    if args.full_model:
        small_model = False
        model_type = "Full BERT-base"
    else:
        small_model = True
        model_type = "Small BERT"
    
    mixed_precision = not args.no_mixed_precision
    
    print("🌱 Botanical BERT Training Launcher")
    print("=" * 50)
    print(f"🤖 Model Type: {model_type}")
    print(f"📊 Epochs: {args.epochs}")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"📖 Learning Rate: {args.learning_rate}")
    print(f"📏 Max Length: {args.max_length}")
    print(f"🔧 Mixed Precision: {mixed_precision}")
    print("=" * 50)
    
    # Eğitim başlat
    try:
        trainer = BotanicalBERTTrainer(
            mixed_precision=mixed_precision,
            max_length=args.max_length,
            small_model=small_model
        )
        
        print(f"\n🚀 Eğitim başlıyor...")
        result = trainer.train_botanical_bert(
            epochs=args.epochs,
            lr=args.learning_rate,
            batch_size=args.batch_size
        )
        
        if result:
            print(f"\n🎉 Eğitim başarıyla tamamlandı!")
            print(f"🎯 Test Accuracy: {result['test_metrics']['accuracy']:.4f}")
            print(f"🎯 Test F1 Score: {result['test_metrics']['f1_score']:.4f}")
            print(f"⏱️ Training Time: {result['config']['training_time']:.2f} seconds")
            print(f"💾 Model saved to: models/botanical_bert_{'small' if small_model else 'base'}")
        else:
            print(f"\n❌ Eğitim başarısız!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Eğitim hatası: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 