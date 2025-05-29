#!/usr/bin/env python3
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

print("📊 MODEL SKOR ANALİZİ")
print("="*50)

# Yükleme
index_dir = Path("./final_system/complete_index")
model = SentenceTransformer('all-MiniLM-L6-v2')

with open(index_dir / "chunks" / "all_chunks.json", 'r', encoding='utf-8') as f:
    chunks = json.load(f)

index = faiss.read_index(str(index_dir / "indices" / "faiss_index.bin"))

# Farklı kategorilerde test sorguları
test_categories = {
    "🍅 Domates Hastalıkları": [
        "tomato disease detection",
        "domates hastalık tespiti",
        "tomato leaf blight",
        "domates yaprak yanıklığı"
    ],
    "🦠 Fungal Enfeksiyonlar": [
        "fungal infection plants",
        "mantar hastalığı bitki",
        "antifungal treatment",
        "fungus detection"
    ],
    "🤖 Yapay Zeka Tarım": [
        "machine learning agriculture",
        "yapay zeka tarım",
        "AI crop monitoring",
        "deep learning plant"
    ],
    "🌾 Genel Bitki Hastalıkları": [
        "plant disease classification",
        "bitki hastalık sınıflandırma",
        "crop pathology",
        "plant health monitoring"
    ]
}

print("📈 SKOR ANALİZ RAPORU")
print("="*50)

all_scores = []
category_scores = {}

for category, queries in test_categories.items():
    print(f"\n{category}")
    print("-" * 40)
    
    category_score_list = []
    
    for query in queries:
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = index.search(query_embedding.astype('float32'), 10)
        
        top_scores = scores[0][:5]  # İlk 5 sonuç
        avg_score = np.mean(top_scores)
        max_score = np.max(top_scores)
        
        print(f"🔍 '{query}':")
        print(f"   📊 En yüksek: {max_score:.4f}")
        print(f"   📊 Ortalama: {avg_score:.4f}")
        
        category_score_list.extend(top_scores)
        all_scores.extend(top_scores)
    
    category_avg = np.mean(category_score_list)
    category_scores[category] = category_avg
    print(f"\n📋 {category} Kategorisi Ortalama: {category_avg:.4f}")

# Genel analiz
print(f"\n🎯 GENEL ANALİZ")
print("="*50)

overall_avg = np.mean(all_scores)
overall_std = np.std(all_scores)
score_percentiles = np.percentile(all_scores, [25, 50, 75, 90, 95])

print(f"📊 Genel İstatistikler:")
print(f"   🎯 Ortalama Skor: {overall_avg:.4f}")
print(f"   📈 Standart Sapma: {overall_std:.4f}")
print(f"   🏆 En Yüksek: {np.max(all_scores):.4f}")
print(f"   📉 En Düşük: {np.min(all_scores):.4f}")

print(f"\n📊 Percentile Analizi:")
print(f"   25%: {score_percentiles[0]:.4f}")
print(f"   50%: {score_percentiles[1]:.4f}")
print(f"   75%: {score_percentiles[2]:.4f}")
print(f"   90%: {score_percentiles[3]:.4f}")
print(f"   95%: {score_percentiles[4]:.4f}")

# Skor kalitesi değerlendirmesi
print(f"\n🎭 SKOR KALİTE DEĞERLENDİRMESİ")
print("="*50)

excellent_count = len([s for s in all_scores if s >= 0.7])
good_count = len([s for s in all_scores if 0.5 <= s < 0.7])
fair_count = len([s for s in all_scores if 0.3 <= s < 0.5])
poor_count = len([s for s in all_scores if s < 0.3])

total_results = len(all_scores)

print(f"🟢 Mükemmel (≥0.7): {excellent_count:3d} ({excellent_count/total_results*100:.1f}%)")
print(f"🟡 İyi (0.5-0.7):   {good_count:3d} ({good_count/total_results*100:.1f}%)")
print(f"🟠 Orta (0.3-0.5):  {fair_count:3d} ({fair_count/total_results*100:.1f}%)")
print(f"🔴 Zayıf (<0.3):    {poor_count:3d} ({poor_count/total_results*100:.1f}%)")

# Öneri
print(f"\n💡 ÖNERİLER")
print("="*50)

if overall_avg >= 0.65:
    print("✅ Model skoru ÇOK İYİ! RAG sistemi için uygun.")
elif overall_avg >= 0.50:
    print("✅ Model skoru İYİ! Kullanılabilir seviyede.")
    print("💡 İyileştirme için daha fazla domain-specific data eklenebilir.")
elif overall_avg >= 0.35:
    print("⚠️  Model skoru ORTA! Kullanılabilir ama iyileştirme gerekli.")
    print("💡 Öneriler:")
    print("   - Daha büyük embedding modeli kullan (all-MiniLM-L12-v2)")
    print("   - Domain-specific fine-tuning yap")
    print("   - Chunk boyutunu optimize et")
else:
    print("❌ Model skoru DÜŞÜK! İyileştirme şart.")
    print("💡 Acil öneriler:")
    print("   - Embedding modelini değiştir")
    print("   - Veri kalitesini gözden geçir")
    print("   - Preprocessing'i iyileştir")

print(f"\n🔍 KARŞILAŞTIRMA")
print("="*50)
print("📊 Embedding Model Skorları (Genel):")
print("   all-MiniLM-L6-v2:  0.40-0.70 (şu anki)")
print("   all-MiniLM-L12-v2: 0.45-0.75 (daha iyi)")
print("   all-mpnet-base-v2: 0.50-0.80 (en iyi)")
print("   sentence-t5-base:  0.45-0.75 (iyi)")

print(f"\n🎯 SONUÇ: Mevcut skorlar tarım RAG sistemi için {'BAŞARILI' if overall_avg >= 0.5 else 'GELİŞTİRİLMELİ'}!") 