#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Hızlı Başlangıç Script'i
Tarımsal AI Chatbot projesini kolayca çalıştırmak için

Kullanım:
    python3 quick_start.py
"""

import sys
import subprocess
import os
from pathlib import Path

def print_banner():
    """Başlık yazdır"""
    print("=" * 60)
    print("🌾 Tarımsal AI Chatbot - Hızlı Başlangıç")
    print("=" * 60)
    print()

def check_python_version():
    """Python versiyonunu kontrol et"""
    print("🐍 Python versiyonu kontrol ediliyor...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ gerekli. Mevcut versiyon:", f"{version.major}.{version.minor}")
        print("💡 Lütfen Python'u güncelleyin: https://python.org/downloads/")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Uygun")
    return True

def check_dependencies():
    """Bağımlılıkları kontrol et"""
    print("\n📦 Bağımlılıklar kontrol ediliyor...")
    
    required_packages = [
        "streamlit",
        "pandas", 
        "plotly",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Mevcut")
        except ImportError:
            print(f"❌ {package} - Eksik")
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(missing_packages):
    """Eksik bağımlılıkları kur"""
    if not missing_packages:
        return True
    
    print(f"\n📥 {len(missing_packages)} eksik paket kuruluyor...")
    
    try:
        # Requirements.txt'den kur
        if Path("requirements.txt").exists():
            print("📋 requirements.txt dosyasından kurulum...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True, capture_output=True)
        else:
            # Tek tek kur
            for package in missing_packages:
                print(f"📦 {package} kuruluyor...")
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
        
        print("✅ Tüm bağımlılıklar başarıyla kuruldu!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Kurulum hatası: {e}")
        print("💡 Manuel kurulum deneyin:")
        print("   python3 -m pip install streamlit pandas plotly")
        return False

def check_project_structure():
    """Proje yapısını kontrol et"""
    print("\n📁 Proje yapısı kontrol ediliyor...")
    
    required_files = [
        "demo_app.py",
        "requirements.txt",
        "ODEV_README.md"
    ]
    
    missing_files = []
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} - Mevcut")
        else:
            print(f"❌ {file} - Eksik")
            missing_files.append(file)
    
    return len(missing_files) == 0

def run_streamlit_app():
    """Streamlit uygulamasını çalıştır"""
    print("\n🚀 Streamlit uygulaması başlatılıyor...")
    print("📱 Uygulama tarayıcınızda açılacak: http://localhost:8501")
    print()
    print("💡 İpuçları:")
    print("   • Uygulamayı durdurmak için: Ctrl+C")
    print("   • Farklı port için: --server.port 8502")
    print("   • Tarayıcı açılmazsa: http://localhost:8501 adresine gidin")
    print()
    print("🌾 Tarımsal AI Chatbot Demo hazır!")
    print("=" * 60)
    
    try:
        # Streamlit uygulamasını çalıştır
        subprocess.run([sys.executable, "-m", "streamlit", "run", "demo_app.py"], 
                      check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Uygulama kapatılıyor...")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Streamlit başlatma hatası: {e}")
        print("\n💡 Alternatif çalıştırma yöntemleri:")
        print("   1. python3 -m streamlit run demo_app.py")
        print("   2. streamlit run demo_app.py")
        print("   3. python3 demo_app.py (eğer destekleniyorsa)")

def show_help():
    """Yardım menüsü"""
    print("\n❓ Yardım Menüsü")
    print("=" * 40)
    print("🎯 Bu script'in amacı:")
    print("   • Sistem gereksinimlerini kontrol etmek")
    print("   • Eksik bağımlılıkları otomatik kurmak") 
    print("   • Streamlit uygulamasını başlatmak")
    print()
    print("🔧 Kullanım:")
    print("   python3 quick_start.py        # Normal çalıştırma")
    print("   python3 quick_start.py --help # Bu yardım menüsü")
    print()
    print("📂 Proje Dosyaları:")
    print("   • demo_app.py      - Ana Streamlit uygulaması")
    print("   • requirements.txt - Python bağımlılıkları")
    print("   • ODEV_README.md   - Detaylı dokümantasyon")
    print()
    print("🐛 Sorun yaşıyorsanız:")
    print("   1. Python 3.9+ yüklü olduğundan emin olun")
    print("   2. İnternet bağlantınızı kontrol edin")
    print("   3. Manuel kurulum deneyin:")
    print("      python3 -m pip install streamlit pandas plotly")
    print("      python3 -m streamlit run demo_app.py")

def main():
    """Ana fonksiyon"""
    # Komut satırı argümanlarını kontrol et
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
        show_help()
        return
    
    print_banner()
    
    # 1. Python versiyonu kontrol
    if not check_python_version():
        sys.exit(1)
    
    # 2. Bağımlılıkları kontrol et
    missing_packages = check_dependencies()
    
    # 3. Eksik bağımlılıkları kur
    if missing_packages:
        install_success = install_dependencies(missing_packages)
        if not install_success:
            print("\n❌ Bağımlılık kurulumu başarısız. Lütfen manuel kurulum yapın.")
            sys.exit(1)
    
    # 4. Proje yapısını kontrol et
    if not check_project_structure():
        print("\n❌ Proje dosyaları eksik. Lütfen dosyaları kontrol edin.")
        sys.exit(1)
    
    # 5. Streamlit uygulamasını çalıştır
    run_streamlit_app()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Script sonlandırıldı.")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        print("💡 Yardım için: python3 quick_start.py --help") 