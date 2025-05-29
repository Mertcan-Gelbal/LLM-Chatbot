#!/usr/bin/env python3
import fitz, json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm

print('🚀 TAM İNDEKSLEME BAŞLIYOR')
print('='*50)

# Model yükle
print('🧠 Model yükleniyor...')
model = SentenceTransformer('all-MiniLM-L6-v2')
print('✅ Model yüklendi')

chunks = []

# PDF'leri işle
pdf_dir = Path('./data_processing/real_papers/pdfs')
pdf_files = list(pdf_dir.glob('*.pdf'))
print(f'📄 {len(pdf_files)} PDF işleniyor...')

for pdf_path in tqdm(pdf_files, desc='PDF'):
    try:
        doc = fitz.open(pdf_path)
        text = ''
        for page in doc:
            text += page.get_text() + ' '
        doc.close()
        
        if len(text.strip()) > 100:
            words = text.split()
            for j in range(0, len(words), 500):
                chunk_words = words[j:j+500]
                if len(chunk_words) > 50:
                    chunk_text = ' '.join(chunk_words)
                    chunks.append({
                        'text': chunk_text,
                        'source': 'pdf',
                        'filename': pdf_path.name,
                        'chunk_id': len(chunks)
                    })
    except:
        continue

print(f'✅ {len(chunks)} PDF chunk oluşturuldu')

# Sentetik makaleleri işle
syn_file = Path('./data_processing/synthetic_papers/all_synthetic_papers.json')
if syn_file.exists():
    print('📚 Sentetik makaleler yükleniyor...')
    try:
        with open(syn_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = data.get('papers', [])
        print(f'📚 {len(papers)} sentetik makale işleniyor...')
        
        for paper in tqdm(papers, desc='Sentetik'):
            try:
                full_text = ''
                if paper.get('abstract'):
                    full_text += paper['abstract'] + ' '
                
                for section in paper.get('sections', []):
                    content = section.get('content', '')
                    full_text += content + ' '
                
                if len(full_text.strip()) > 100:
                    words = full_text.split()
                    for j in range(0, len(words), 500):
                        chunk_words = words[j:j+500]
                        if len(chunk_words) > 50:
                            chunk_text = ' '.join(chunk_words)
                            chunks.append({
                                'text': chunk_text,
                                'source': 'synthetic',
                                'paper_id': paper.get('metadata', {}).get('id', 'unknown'),
                                'chunk_id': len(chunks)
                            })
            except:
                continue
                
    except Exception as e:
        print(f'❌ Sentetik makale hatası: {e}')

print(f'✅ Toplam {len(chunks)} chunk oluşturuldu')

# Embedding ve indeksleme
if chunks:
    print(f'🧠 {len(chunks)} chunk için embedding oluşturuluyor...')
    texts = [chunk['text'] for chunk in chunks]
    
    # Batch halinde embedding
    batch_size = 32
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc='Embedding'):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(all_embeddings)
    
    # FAISS indeksi
    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    
    # Kaydet
    out_dir = Path('./final_system/complete_index')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'chunks').mkdir(exist_ok=True)
    (out_dir / 'embeddings').mkdir(exist_ok=True)
    (out_dir / 'indices').mkdir(exist_ok=True)
    
    # Chunk'ları kaydet
    with open(out_dir / 'chunks' / 'all_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    # Embedding'leri kaydet
    np.save(out_dir / 'embeddings' / 'embeddings.npy', embeddings)
    
    # FAISS indeksini kaydet
    faiss.write_index(index, str(out_dir / 'indices' / 'faiss_index.bin'))
    
    # İstatistikleri kaydet
    stats = {
        'total_chunks': len(chunks),
        'pdf_chunks': len([c for c in chunks if c['source'] == 'pdf']),
        'synthetic_chunks': len([c for c in chunks if c['source'] == 'synthetic']),
        'embedding_dimension': dimension,
        'index_size': index.ntotal,
        'total_words': sum(len(c['text'].split()) for c in chunks)
    }
    
    with open(out_dir / 'stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f'✅ İndeksleme tamamlandı!')
    print(f'📊 İstatistikler:')
    print(f'   📝 Toplam chunk: {stats["total_chunks"]:,}')
    print(f'   📄 PDF chunk: {stats["pdf_chunks"]:,}')
    print(f'   📚 Sentetik chunk: {stats["synthetic_chunks"]:,}')
    print(f'   🧠 Embedding boyutu: {stats["embedding_dimension"]}D')
    print(f'   💬 Toplam kelime: {stats["total_words"]:,}')
    print(f'   💾 Çıktı dizini: {out_dir}')
    
    # Test arama
    print('')
    print('🔍 Test arama yapılıyor...')
    test_queries = [
        'plant disease detection',
        'fungal infection tomato'
    ]
    
    for query in test_queries:
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = index.search(query_embedding.astype('float32'), 3)
        
        print('')
        print(f'🔎 Arama: "{query}"')
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            chunk = chunks[idx]
            source_icon = '📄' if chunk['source'] == 'pdf' else '📚'
            print(f'  {i+1}. {source_icon} Skor: {score:.3f} - {chunk["text"][:80]}...')
    
    print('')
    print('🎉 KAPSAMLI İNDEKSLEME BAŞARILI!')
    print('✅ RAG sistemi için hazır!')
    
else:
    print('❌ Hiç chunk oluşturulamadı!') 