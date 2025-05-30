#!/usr/bin/env python3
"""
Dual Model Training Master Script
BERT-small ve DistilBERT modellerini sırayla eğiten ana script
"""

import os
import time
import sys
from pathlib import Path
import subprocess
from rich.console import Console
from rich.progress import track
from rich.panel import Panel
from rich.table import Table

console = Console()

def run_training_script(script_name: str, model_name: str) -> dict:
    """Eğitim scriptini çalıştır ve sonuçları topla"""
    console.print(f"\n🚀 {model_name} eğitimi başlıyor...", style="bold green")
    
    start_time = time.time()
    
    try:
        # Script çalıştır
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=3600  # 1 saat timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            console.print(f"✅ {model_name} eğitimi başarıyla tamamlandı!", style="bold green")
            console.print(f"⏱️  Süre: {duration/60:.1f} dakika")
            
            return {
                'success': True,
                'duration_minutes': duration / 60,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            console.print(f"❌ {model_name} eğitimi başarısız!", style="bold red")
            console.print(f"Hata: {result.stderr}")
            
            return {
                'success': False,
                'duration_minutes': duration / 60,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
    
    except subprocess.TimeoutExpired:
        console.print(f"⏰ {model_name} eğitimi timeout (1 saat)!", style="bold red")
        return {
            'success': False,
            'duration_minutes': 60,
            'stdout': '',
            'stderr': 'Timeout after 1 hour'
        }
    
    except Exception as e:
        console.print(f"❌ {model_name} eğitimi sırasında hata: {e}", style="bold red")
        return {
            'success': False,
            'duration_minutes': 0,
            'stdout': '',
            'stderr': str(e)
        }

def check_requirements():
    """Gerekli kütüphanelerin varlığını kontrol et"""
    console.print("🔍 Gereksinimler kontrol ediliyor...", style="bold blue")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'sklearn', 
        'pandas', 'numpy', 'matplotlib', 'rich'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            console.print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            console.print(f"❌ {package} - EKSİK!")
    
    if missing_packages:
        console.print(f"\n⚠️  Eksik paketler: {', '.join(missing_packages)}", style="bold yellow")
        console.print("Bu paketleri yüklemek için: pip install " + " ".join(missing_packages))
        return False
    
    console.print("\n✅ Tüm gereksinimler mevcut!", style="bold green")
    return True

def check_data():
    """Veri setlerinin varlığını kontrol et"""
    console.print("\n📊 Veri setleri kontrol ediliyor...", style="bold blue")
    
    data_dir = Path("../Data")
    required_files = ['train.csv', 'val.csv', 'test.csv']
    
    if not data_dir.exists():
        console.print(f"❌ Data klasörü bulunamadı: {data_dir}", style="bold red")
        return False
    
    missing_files = []
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            console.print(f"✅ {file}")
        else:
            missing_files.append(file)
            console.print(f"❌ {file} - EKSİK!")
    
    if missing_files:
        console.print(f"\n⚠️  Eksik dosyalar: {', '.join(missing_files)}", style="bold yellow")
        return False
    
    console.print("\n✅ Tüm veri setleri mevcut!", style="bold green")
    return True

def create_results_summary(bert_result: dict, distilbert_result: dict):
    """Sonuçların özetini oluştur"""
    console.print("\n" + "="*80)
    console.print(Panel.fit("🏆 EĞİTİM SONUÇLARI ÖZETİ", style="bold green"))
    
    # Results table
    table = Table(title="Model Eğitim Karşılaştırması")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Durum", style="magenta")
    table.add_column("Süre (dk)", style="green")
    table.add_column("Çıktı Klasörü", style="yellow")
    
    # BERT-small
    bert_status = "✅ Başarılı" if bert_result['success'] else "❌ Başarısız"
    table.add_row(
        "BERT-small",
        bert_status,
        f"{bert_result['duration_minutes']:.1f}",
        "bert_small_agricultural/"
    )
    
    # DistilBERT
    distil_status = "✅ Başarılı" if distilbert_result['success'] else "❌ Başarısız"
    table.add_row(
        "DistilBERT",
        distil_status,
        f"{distilbert_result['duration_minutes']:.1f}",
        "distilbert_agricultural/"
    )
    
    console.print(table)
    
    # Toplam süre
    total_time = bert_result['duration_minutes'] + distilbert_result['duration_minutes']
    console.print(f"\n⏱️  Toplam eğitim süresi: {total_time:.1f} dakika ({total_time/60:.1f} saat)")
    
    # Başarı durumu
    successful_models = sum([bert_result['success'], distilbert_result['success']])
    console.print(f"🎯 Başarılı modeller: {successful_models}/2")
    
    if successful_models == 2:
        console.print("\n🎉 Her iki model de başarıyla eğitildi!", style="bold green")
        console.print("\n📁 Model dosyaları:")
        console.print("   • bert_small_agricultural/ - BERT-small modeli")
        console.print("   • distilbert_agricultural/ - DistilBERT modeli")
        console.print("\n📊 Görselleştirmeler:")
        console.print("   • bert_small_agricultural/training_analysis.png")
        console.print("   • distilbert_agricultural/training_analysis.png")
    else:
        console.print("\n⚠️  Bazı modellerde hatalar oluştu. Logları kontrol edin.", style="bold yellow")

def main():
    """Ana eğitim koordinatörü"""
    console.print("🤖 Dual Model Training System", style="bold green")
    console.print("BERT-small vs DistilBERT Karşılaştırmalı Eğitim")
    console.print("=" * 80)
    
    # 1. Gereksinimler kontrolü
    if not check_requirements():
        console.print("\n❌ Gereksinimler eksik! Çıkılıyor...", style="bold red")
        return
    
    # 2. Veri setleri kontrolü
    if not check_data():
        console.print("\n❌ Veri setleri eksik! Çıkılıyor...", style="bold red")
        return
    
    console.print("\n🎯 Her iki model eğitime hazır!", style="bold green")
    
    # Eğitim başlangıç zamanı
    total_start_time = time.time()
    
    # 3. BERT-small eğitimi
    console.print("\n" + "="*80)
    bert_result = run_training_script("train_bert_small.py", "BERT-small")
    
    # 4. DistilBERT eğitimi
    console.print("\n" + "="*80)
    distilbert_result = run_training_script("train_distilbert.py", "DistilBERT")
    
    # 5. Sonuçlar özeti
    create_results_summary(bert_result, distilbert_result)
    
    # Toplam süre
    total_end_time = time.time()
    total_duration = (total_end_time - total_start_time) / 60
    
    console.print(f"\n🏁 Tüm işlemler tamamlandı! Toplam süre: {total_duration:.1f} dakika")
    
    # Next steps
    if bert_result['success'] and distilbert_result['success']:
        console.print("\n🔥 Bir sonraki adımlar:")
        console.print("1. 🧪 Modelleri test edin")
        console.print("2. 📊 Performans karşılaştırması yapın")
        console.print("3. 🚀 Production'a deploy edin")
        console.print("\nModelleri test etmek için:")
        console.print("   python test_models.py")

if __name__ == "__main__":
    main() 