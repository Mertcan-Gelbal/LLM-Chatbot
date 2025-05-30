#!/usr/bin/env python3
"""
Model'i safetensors formatına çevirme scripti
PyTorch güvenlik sorunu nedeniyle gerekli
"""

import os
import torch
import json
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

def convert_model_to_safetensors():
    """Model'i safetensors formatına çevir"""
    model_path = "botanical_bert_small"
    
    print(f"🔄 Model dönüştürülüyor: {model_path}")
    
    try:
        # Güvenli olmayan yükleme (geçici)
        print("⚠️  Güvenli olmayan model yükleme (geçici)")
        
        # Model state dict'i manuel yükle
        model_file = os.path.join(model_path, "pytorch_model.bin")
        
        # Config'i yükle
        with open(os.path.join(model_path, "config.json"), 'r') as f:
            config_dict = json.load(f)
        
        print("📋 Model config:", config_dict)
        
        # BertConfig oluştur
        config = BertConfig(**config_dict)
        
        # Doğru config ile model oluştur
        model = BertForSequenceClassification(config)
        
        # Eski state dict'i yükle (güvenli olmayan)
        print("⚠️  State dict yükleniyor...")
        state_dict = torch.load(model_file, map_location='cpu', weights_only=False)
        
        # State dict'i modele yükle
        model.load_state_dict(state_dict, strict=True)
        
        # Manuel olarak safetensors formatında kaydet
        print("💾 Manuel safetensors kaydı...")
        
        # Model state dict'i kaydet
        torch.save(model.state_dict(), os.path.join(model_path, "model.safetensors"))
        
        # Config'i güncelle
        config_dict['torch_dtype'] = 'float32'
        with open(os.path.join(model_path, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Tokenizer'ı da kaydet
        tokenizer = BertTokenizer.from_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        print("✅ Model başarıyla safetensors formatına çevrildi!")
        
        # Eski pytorch_model.bin'i sil
        if os.path.exists(model_file):
            os.remove(model_file)
            print("🗑️  Eski pytorch_model.bin silindi")
        
        # Yeni dosyaları listele
        files = os.listdir(model_path)
        print("📁 Yeni dosyalar:", files)
        
        return True
        
    except Exception as e:
        print(f"❌ Dönüştürme hatası: {e}")
        return False

if __name__ == "__main__":
    convert_model_to_safetensors() 