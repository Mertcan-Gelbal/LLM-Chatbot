#!/usr/bin/env python3
"""
Basit Tarımsal BERT Fine-tuning
Arkadaşının AG News koduna benzer yapı
"""

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
import matplotlib.pyplot as plt
import json
from pathlib import Path

# --- Tarımsal veri setini hazırlayan fonksiyon ---
def load_agricultural_data():
    """Tarımsal veri setini yükle ve hazırla"""
    
    # Eğer varolan veri seti varsa yükle
    data_path = Path("../Data/agricultural_bert_dataset.json")
    if data_path.exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        
        # Label mapping
        unique_labels = list(set(labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = [label_map[label] for label in labels]
        
        print(f"Loaded {len(texts)} samples with {len(unique_labels)} categories")
        print(f"Categories: {unique_labels}")
        
    else:
        # Manuel tarımsal veri seti oluştur
        texts = [
            # Bitki hastalıkları
            "Elmada erken yanıklığı bakteriyel bir hastalıktır ve hızla yayılır",
            "Domates yapraklarında sarı lekeler beslenme eksikliğinden kaynaklanır", 
            "Buğday paslanması fungal hastalık olup nem ile artış gösterir",
            "Patates mildiyösü soğuk nemli havalarda görülür",
            "Armut ateş yanıklığı bakteriyel kökenli ciddi hastalıktır",
            
            # Yetiştirme teknikleri
            "Buğday ekim zamanı toprak sıcaklığına bağlı olarak belirlenir",
            "Havuç yetiştirmede derin gevşek toprak tercih edilir",
            "Domates sulama düzenli yapılmalı aşırı su verilmemeli",
            "Mısır ekim derinliği 3-5 cm arasında olmalıdır",
            "Fasulye sıcak iklim seven bir bitkidir",
            
            # Gübreleme
            "Azotlu gübre yaprak gelişimini hızlandırır",
            "Fosforlu gübre kök sistemi gelişimini destekler",
            "Potasyum gübre meyve kalitesini artırır",
            "Organik gübre toprak yapısını iyileştirir",
            "Kireç asit toprakları nötralize eder",
            
            # Çevre faktörleri
            "Toprak pH değeri 6.0-7.0 arasında ideal",
            "Aşırı sıcaklık bitki stresine neden olur",
            "Don bitkiler için ciddi tehlike oluşturur",
            "Rüzgar tuz taşınımını etkiler",
            "Nem hastalık gelişimini etkiler"
        ]
        
        labels = [
            # Bitki hastalıkları
            "plant_disease", "plant_disease", "plant_disease", "plant_disease", "plant_disease",
            # Yetiştirme teknikleri  
            "crop_management", "crop_management", "crop_management", "crop_management", "crop_management",
            # Gübreleme
            "crop_management", "crop_management", "crop_management", "crop_management", "crop_management",
            # Çevre faktörleri
            "environmental_factors", "environmental_factors", "environmental_factors", "environmental_factors", "environmental_factors"
        ]
        
        # Label mapping
        unique_labels = list(set(labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = [label_map[label] for label in labels]
        
        print(f"Created {len(texts)} samples with {len(unique_labels)} categories")
        print(f"Categories: {unique_labels}")
    
    # Train/val/test split
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, numeric_labels, test_size=0.4, stratify=numeric_labels, random_state=42)
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)
    
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, len(unique_labels)

# --- Dataset sınıfı ---
class AgriculturalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
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

# --- Eğitim fonksiyonu ---
def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# --- Değerlendirme fonksiyonu ---
def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return acc, precision, f1

# --- Test fonksiyonu ---
def test_model(model, tokenizer, device, num_labels):
    """Eğitilmiş modeli test et"""
    model.eval()
    
    test_questions = [
        "Elmada erken yanıklığı nasıl tedavi edilir?",
        "Buğday ekim zamanı ne zaman?", 
        "Toprak pH değeri neden önemli?",
        "Azotlu gübre ne işe yarar?"
    ]
    
    categories = ["plant_disease", "crop_management", "environmental_factors"]
    
    print("\n🧪 Model Test Sonuçları:")
    print("-" * 50)
    
    for question in test_questions:
        # Tokenize
        encoding = tokenizer(
            question,
            truncation=True,
            padding='max_length', 
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item()
        
        pred_category = categories[pred_id] if pred_id < len(categories) else f"Category_{pred_id}"
        
        print(f"Soru: {question}")
        print(f"Tahmin: {pred_category} (Güven: {confidence:.3f})")
        print()

# --- Ana deney fonksiyonu ---
def run_agricultural_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🚀 Using device: {device}')

    # Veriyi yükle
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, num_labels = load_agricultural_data()

    # BERT modelleri
    models_info = {
        'bert-base-uncased': (BertTokenizer, BertForSequenceClassification),
    }

    results = []
    num_epochs = 3
    batch_size = 8  # Jetson için küçük batch

    for model_name, (TokenizerClass, ModelClass) in models_info.items():
        print(f"\n🧠 Training {model_name} on Agricultural Data")

        # Model ve tokenizer yükle
        tokenizer = TokenizerClass.from_pretrained(model_name)
        model = ModelClass.from_pretrained(model_name, num_labels=num_labels)
        model.to(device)

        # Dataset oluştur
        train_dataset = AgriculturalDataset(train_texts, train_labels, tokenizer)
        val_dataset = AgriculturalDataset(val_texts, val_labels, tokenizer)
        test_dataset = AgriculturalDataset(test_texts, test_labels, tokenizer)

        # DataLoader oluştur
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Optimizer ve scheduler
        optimizer = AdamW(model.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * num_epochs
        )

        # Eğitim döngüsü
        print("📚 Training started...")
        for epoch in range(num_epochs):
            loss = train(model, train_loader, optimizer, scheduler, device)
            val_acc, val_prec, val_f1 = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1}/{num_epochs}: loss={loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        # Test değerlendirmesi
        test_acc, test_prec, test_f1 = evaluate(model, test_loader, device)
        results.append({
            'Model': model_name,
            'Accuracy': test_acc,
            'Precision': test_prec,
            'F1-Score': test_f1
        })

        print(f"✅ {model_name} Test Results:")
        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   Precision: {test_prec:.4f}")
        print(f"   F1-Score: {test_f1:.4f}")

        # Model test et
        test_model(model, tokenizer, device, num_labels)

        # Modeli kaydet
        model_save_path = Path(f"agricultural_{model_name.replace('-', '_')}")
        model_save_path.mkdir(exist_ok=True)
        
        # Basit kaydetme (eski PyTorch için)
        try:
            # Config ve tokenizer kaydet
            model.config.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            # Model state dict kaydet
            torch.save(model.state_dict(), model_save_path / "pytorch_model.bin")
            
            print(f"💾 Model saved to {model_save_path}")
        except Exception as e:
            print(f"❌ Model kaydetme hatası: {e}")

        # Bellek temizle
        del model
        torch.cuda.empty_cache()

    # Sonuçları göster
    result_df = pd.DataFrame(results)
    print("\n📊 Final Results on Agricultural Test Set:")
    print(result_df)

    # Grafik çiz
    if len(results) > 0:
        result_df.set_index("Model")[["Accuracy", "Precision", "F1-Score"]].plot.bar(rot=45, figsize=(10,6))
        plt.title("BERT Model Performance on Agricultural Dataset")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig("agricultural_bert_results.png")
        plt.show()
        print("📈 Grafik agricultural_bert_results.png olarak kaydedildi")

if __name__ == '__main__':
    print("🌾 Tarımsal BERT Fine-tuning Başlatılıyor...")
    run_agricultural_experiment()
    print("🎉 Eğitim tamamlandı!") 