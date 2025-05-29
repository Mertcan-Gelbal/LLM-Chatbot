#!/usr/bin/env python3
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

print("🔍 RAG İNDEKS TEST ARAMA")
print("="*40)

# Yükleme
print("📂 Veriler yükleniyor...")
index_dir = Path("./final_system/complete_index")

# Model yükle
model = SentenceTransformer('all-MiniLM-L6-v2')

# Chunk'ları yükle
with open(index_dir / "chunks" / "all_chunks.json", 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# FAISS indeksini yükle
index = faiss.read_index(str(index_dir / "indices" / "faiss_index.bin"))

print(f"✅ {len(chunks)} chunk yüklendi")
print(f"✅ FAISS indeksi yüklendi")

# Test sorguları
test_queries = [
    "tomato disease detection machine learning",
    "fungal infection plant pathology",
    "pest management agriculture",
    "crop yield prediction AI",
    "plant stress monitoring sensors"
]

for query in test_queries:
    print(f"\n🔎 Arama: '{query}'")
    print("-" * 50)
    
    # Query embedding
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Arama
    scores, indices = index.search(query_embedding.astype('float32'), 5)
    
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        chunk = chunks[idx]
        source_icon = "📄" if chunk['source'] == 'pdf' else "📚"
        
        print(f"{i+1}. {source_icon} Skor: {score:.4f}")
        if chunk['source'] == 'pdf':
            print(f"   📁 Dosya: {chunk['filename']}")
        else:
            print(f"   📋 ID: {chunk['paper_id']}")
        
        print(f"   📝 Metin: {chunk['text'][:150]}...")
        print()

print("�� Test tamamlandı!") 