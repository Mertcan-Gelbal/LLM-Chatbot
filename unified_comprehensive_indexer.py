#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Comprehensive Indexer
Gerçek PDF'ler + Sentetik makaleleri birlikte indeksleyen kapsamlı sistem
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import hashlib
import re
import concurrent.futures
from multiprocessing import Pool, cpu_count
import pickle
import numpy as np
import faiss
from tqdm import tqdm

# PDF işleme
import fitz  # PyMuPDF
import pdfplumber

# NLP ve Embedding
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedComprehensiveIndexer:
    """Birleşik kapsamlı indeksleme sistemi"""
    
    def __init__(self):
        # Giriş dizinleri
        self.real_pdf_dir = Path("../data_processing/real_papers/pdfs")
        self.synthetic_dir = Path("../data_processing/synthetic_papers")
        
        # Çıktı dizini
        self.output_dir = Path("../final_system/unified_indexed_database")
        self.setup_directories()
        
        # Embedding modelleri
        self.embedding_models = {}
        self.current_model = None
        self.load_embedding_models()
        
        # FAISS indeksleri
        self.faiss_indices = {}
        self.unified_chunks = []
        
        # İşleme parametreleri
        self.chunk_size = 500  # kelime
        self.chunk_overlap = 100  # kelime
        self.max_workers = min(cpu_count() - 1, 8)
        
        # Kalite kontrol
        self.min_text_length = 100
        self.min_pdf_pages = 2
        
        # İstatistikler
        self.processing_stats = {
            "total_real_pdfs": 0,
            "processed_real_pdfs": 0,
            "failed_real_pdfs": 0,
            "total_synthetic_papers": 0,
            "processed_synthetic_papers": 0,
            "total_chunks": 0,
            "real_pdf_chunks": 0,
            "synthetic_chunks": 0,
            "total_words": 0,
            "duplicate_removed": 0
        }
    
    def setup_directories(self):
        """Çıktı dizinlerini oluştur"""
        
        directories = [
            self.output_dir,
            self.output_dir / "chunks",
            self.output_dir / "embeddings", 
            self.output_dir / "indices",
            self.output_dir / "metadata",
            self.output_dir / "cache",
            self.output_dir / "reports",
            self.output_dir / "search_results"
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
    
    def load_embedding_models(self):
        """Embedding modellerini yükle"""
        
        models_to_load = [
            ("multilingual", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            ("english", "sentence-transformers/all-MiniLM-L6-v2"),
        ]
        
        for model_name, model_path in models_to_load:
            try:
                logger.info(f"📥 Embedding modeli yükleniyor: {model_name}")
                model = SentenceTransformer(model_path)
                self.embedding_models[model_name] = model
                
                if not self.current_model:
                    self.current_model = model_name
                    
                logger.info(f"✅ {model_name} modeli yüklendi")
                
            except Exception as e:
                logger.warning(f"⚠️ {model_name} modeli yüklenemedi: {e}")
        
        if not self.embedding_models:
            logger.error("❌ Hiçbir embedding modeli yüklenemedi!")
        else:
            logger.info(f"✅ {len(self.embedding_models)} model yüklendi, aktif: {self.current_model}")

def main():
    """Ana fonksiyon"""
    
    print("🌟 BİRLEŞİK KAPSAMLI İNDEKSLEME SİSTEMİ")
    print("="*80)
    print("🎯 Bu sistem şunları birlikte indeksleyecek:")
    print("   📄 Gerçek PDF makaleleri (182 adet)")
    print("   📚 Sentetik akademik makaleler (1000 adet)")
    print("   🧠 Çoklu embedding modeli")
    print("   🔍 Gelişmiş arama indeksleri")
    
    input("\n⏳ Devam etmek için Enter'a basın...")
    
    indexer = UnifiedComprehensiveIndexer()
    print("✅ İndeksleme sistemi başlatıldı!")

if __name__ == "__main__":
    main() 