#!/usr/bin/env python3
"""
Quick Indexer - Hızlı İndeksleme Sistemi
"""

import json
import logging
from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm
import fitz
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("🚀 HIZLI İNDEKSLEME SİSTEMİ")
    print("="*40)
    
    # Dizinler
    pdf_dir = Path("../data_processing/real_papers/pdfs")
    syn_dir = Path("../data_processing/synthetic_papers")
    out_dir = Path("../final_system/quick_index")
    
    # Çıktı dizini oluştur
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Model yükle
    print("🧠 Model yükleniyor...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model yüklendi")
    except Exception as e:
        print(f"❌ Model hatası: {e}")
        return
    
    chunks = []
    
    # PDF'leri işle
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"📄 {len(pdf_files)} PDF işleniyor...")
    
    for pdf_path in tqdm(pdf_files[:50], desc="PDF"):
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            
            if len(text.strip()) > 100:
                words = text.split()[:1000]
                chunk_text = " ".join(words)
                
                chunks.append({
                    'text': chunk_text,
                    'source': 'pdf',
                    'filename': pdf_path.name
                })
        except:
            continue
    
    print(f"✅ {len(chunks)} PDF chunk oluşturuldu")
    
    # Sentetik makaleleri işle
    syn_file = syn_dir / "all_synthetic_papers.json"
    if syn_file.exists():
        try:
            with open(syn_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            papers = data.get('papers', [])[:100]
            print(f"📚 {len(papers)} sentetik makale işleniyor...")
            
            for paper in tqdm(papers, desc="Sentetik"):
                try:
                    text = ""
                    if paper.get('abstract'):
                        text += paper['abstract'] + "\n"
                    
                    sections = paper.get('sections', [])[:3]
                    for section in sections:
                        content = section.get('content', '')
                        text += content[:500] + "\n"
                    
                    if len(text.strip()) > 100:
                        chunks.append({
                            'text': text.strip(),
                            'source': 'synthetic',
                            'id': paper.get('metadata', {}).get('id', 'unknown')
                        })
                except:
                    continue
        except Exception as e:
            print(f"❌ Sentetik makale hatası: {e}")
    
    print(f"✅ Toplam {len(chunks)} chunk oluşturuldu")
    
    if not chunks:
        print("❌ Hiç chunk oluşturulamadı!")
        return
    
    # Embedding oluştur
    print("🧠 Embedding oluşturuluyor...")
    texts = [chunk['text'] for chunk in chunks]
    
    try:
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)
        
        # FAISS indeksi
        dimension = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        
        # Kaydet
        chunks_file = out_dir / "chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        embeddings_file = out_dir / "embeddings.npy"
        np.save(embeddings_file, embeddings)
        
        index_file = out_dir / "index.bin"
        faiss.write_index(index, str(index_file))
        
        print(f"✅ İndeksleme tamamlandı!")
        print(f"📊 {len(chunks)} chunk, {dimension}D embedding")
        print(f"💾 Dosyalar: {out_dir}")
        
        # Test arama
        print("\n🔍 Test arama...")
        query = "plant disease detection"
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = index.search(query_embedding.astype('float32'), 3)
        
        print(f"Arama: '{query}'")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            chunk = chunks[idx]
            print(f"{i+1}. Skor: {score:.3f} - {chunk['source']} - {chunk['text'][:100]}...")
        
        print("\n🎉 HIZLI İNDEKSLEME BAŞARILI!")
        
    except Exception as e:
        print(f"❌ Embedding hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 