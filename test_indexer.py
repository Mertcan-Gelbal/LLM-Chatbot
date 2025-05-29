#!/usr/bin/env python3
# Test İndeksleme Sistemi

import sys
from simple_unified_indexer import SimpleUnifiedIndexer

print("🧪 İNDEKSLEME SİSTEMİ TEST EDİLİYOR")
print("="*50)

try:
    # İndeksleyici oluştur
    print("📝 İndeksleyici oluşturuluyor...")
    indexer = SimpleUnifiedIndexer()
    
    print(f"📁 PDF dizini: {'✅ VAR' if indexer.real_pdf_dir.exists() else '❌ YOK'}")
    print(f"📁 Sentetik dizin: {'✅ VAR' if indexer.synthetic_dir.exists() else '❌ YOK'}")
    print(f"🧠 Embedding modeli: {'✅ YÜKLENDİ' if indexer.embedding_model else '❌ YÜKLENEMEDİ'}")
    
    if indexer.real_pdf_dir.exists():
        pdf_count = len(list(indexer.real_pdf_dir.glob("*.pdf")))
        print(f"📄 PDF sayısı: {pdf_count}")
    
    # Tam indeksleme çalıştır
    if indexer.embedding_model:
        print("\n🚀 TAM İNDEKSLEME BAŞLIYOR...")
        success = indexer.run_full_indexing()
        print(f"\n{'🎉 BAŞARILI!' if success else '❌ BAŞARISIZ!'}")
    else:
        print("\n❌ Embedding modeli yüklenemedi!")
        
except Exception as e:
    print(f"\n💥 HATA: {e}")
    import traceback
    traceback.print_exc() 