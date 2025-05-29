#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Indexer - Birleşik İndeksleme Sistemi
"""

import json
import logging
from pathlib import Path
import hashlib
import numpy as np
import faiss
from tqdm import tqdm
import fitz
import pdfplumber
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedIndexer:
    def __init__(self):
        # Dizinler
        self.real_pdf_dir = Path("../data_processing/real_papers/pdfs")
        self.synthetic_dir = Path("../data_processing/synthetic_papers")
        self.output_dir = Path("../final_system/unified_database")
        
        # Çıktı dizinini oluştur
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "chunks").mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        (self.output_dir / "indices").mkdir(exist_ok=True)
        
        # Model
        self.embedding_model = None
        self.load_model()
        
        # Veriler
        self.chunks = []
        
    def load_model(self):
        try:
            logger.info("🧠 Embedding modeli yükleniyor...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Model yüklendi")
        except Exception as e:
            logger.error(f"❌ Model yüklenemedi: {e}")
    
    def extract_pdf_text(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\\n"
            doc.close()
            return text.strip() if len(text.strip()) > 100 else None
        except:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\\n"
                    return text.strip() if len(text.strip()) > 100 else None
            except:
                return None
    
    def process_pdfs(self):
        pdf_files = list(self.real_pdf_dir.glob("*.pdf"))
        logger.info(f"📄 {len(pdf_files)} PDF işleniyor...")
        
        for pdf_path in tqdm(pdf_files, desc="PDF'ler"):
            text = self.extract_pdf_text(pdf_path)
            if text:
                # Chunk'lara böl
                words = text.split()
                for i in range(0, len(words), 500):
                    chunk_words = words[i:i+500]
                    chunk_text = " ".join(chunk_words)
                    
                    if len(chunk_text) > 50:
                        self.chunks.append({
                            'text': chunk_text,
                            'source': 'pdf',
                            'filename': pdf_path.name,
                            'chunk_id': len(self.chunks)
                        })
        
        logger.info(f"✅ {len(self.chunks)} PDF chunk oluşturuldu")
    
    def process_synthetic(self):
        syn_file = self.synthetic_dir / "all_synthetic_papers.json"
        if not syn_file.exists():
            logger.warning("❌ Sentetik makale dosyası bulunamadı")
            return
        
        try:
            with open(syn_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            papers = data.get('papers', [])
            logger.info(f"📚 {len(papers)} sentetik makale işleniyor...")
            
            initial_count = len(self.chunks)
            
            for paper in tqdm(papers, desc="Sentetik"):
                # Tam metni oluştur
                full_text = ""
                if paper.get('abstract'):
                    full_text += f"Abstract: {paper['abstract']}\\n\\n"
                
                for section in paper.get('sections', []):
                    title = section.get('title', '')
                    content = section.get('content', '')
                    full_text += f"{title}: {content}\\n\\n"
                
                if len(full_text) > 100:
                    # Chunk'lara böl
                    words = full_text.split()
                    for i in range(0, len(words), 500):
                        chunk_words = words[i:i+500]
                        chunk_text = " ".join(chunk_words)
                        
                        if len(chunk_text) > 50:
                            self.chunks.append({
                                'text': chunk_text,
                                'source': 'synthetic',
                                'paper_id': paper.get('metadata', {}).get('id', 'unknown'),
                                'chunk_id': len(self.chunks)
                            })
            
            syn_count = len(self.chunks) - initial_count
            logger.info(f"✅ {syn_count} sentetik chunk oluşturuldu")
            
        except Exception as e:
            logger.error(f"❌ Sentetik makale işleme hatası: {e}")
    
    def create_embeddings(self):
        if not self.chunks:
            logger.error("❌ Chunk bulunamadı!")
            return False
        
        logger.info(f"🧠 {len(self.chunks)} chunk için embedding oluşturuluyor...")
        
        texts = [chunk['text'] for chunk in self.chunks]
        
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # FAISS indeksi oluştur
            dimension = embeddings.shape[1]
            faiss.normalize_L2(embeddings)
            
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings.astype('float32'))
            
            # Kaydet
            chunks_file = self.output_dir / "chunks" / "all_chunks.json"
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            
            embeddings_file = self.output_dir / "embeddings" / "embeddings.npy"
            np.save(embeddings_file, embeddings)
            
            index_file = self.output_dir / "indices" / "faiss_index.bin"
            faiss.write_index(index, str(index_file))
            
            # Stats
            stats = {
                'total_chunks': len(self.chunks),
                'pdf_chunks': len([c for c in self.chunks if c['source'] == 'pdf']),
                'synthetic_chunks': len([c for c in self.chunks if c['source'] == 'synthetic']),
                'embedding_dimension': dimension,
                'index_size': index.ntotal
            }
            
            stats_file = self.output_dir / "stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"✅ {len(self.chunks)} chunk indekslendi")
            logger.info(f"📊 PDF: {stats['pdf_chunks']}, Sentetik: {stats['synthetic_chunks']}")
            logger.info(f"💾 Veriler kaydedildi: {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Embedding hatası: {e}")
            return False
    
    def run(self):
        logger.info("🚀 BİRLEŞİK İNDEKSLEME BAŞLIYOR")
        
        if not self.embedding_model:
            logger.error("❌ Embedding modeli yük")
            return False
        
        # PDF'leri işle
        self.process_pdfs()
        
        # Sentetik makaleleri işle
        self.process_synthetic()
        
        # Embedding ve indeksleme
        return self.create_embeddings()

def main():
    print("🌟 BİRLEŞİK İNDEKSLEME SİSTEMİ")
    print("="*50)
    
    indexer = UnifiedIndexer()
    success = indexer.run()
    
    if success:
        print("🎉 İNDEKSLEME TAMAMLANDI!")
    else:
        print("❌ İndeksleme başarısız!")

if __name__ == "__main__":
    main() 