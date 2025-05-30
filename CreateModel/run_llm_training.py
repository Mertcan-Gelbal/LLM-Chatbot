#!/usr/bin/env python3
"""
Tarımsal LLM Eğitimi Başlatıcı
Hem eğitim hem de chatbot için tek script
"""

import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint

console = Console()

def check_requirements():
    """Gerekli kütüphaneleri kontrol et"""
    try:
        import torch
        import transformers
        import datasets
        from rich import print
        console.print("✅ Temel kütüphaneler mevcut", style="green")
        return True
    except ImportError as e:
        console.print(f"❌ Eksik kütüphane: {e}", style="red")
        console.print("📦 Lütfen requirements_llm.txt'i yükleyin: pip install -r requirements_llm.txt", style="yellow")
        return False

def install_requirements():
    """Requirements'ları yükle"""
    console.print("📦 Gerekli kütüphaneler yükleniyor...", style="cyan")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_llm.txt"])
        console.print("✅ Kütüphaneler başarıyla yüklendi!", style="green")
        return True
    except subprocess.CalledProcessError:
        console.print("❌ Kütüphane yükleme başarısız!", style="red")
        return False

def run_training():
    """LLM eğitimini başlat"""
    console.print("🚀 LLM eğitimi başlatılıyor...", style="bold cyan")
    try:
        subprocess.run([sys.executable, "train_agricultural_llm.py"])
    except Exception as e:
        console.print(f"❌ Eğitim hatası: {e}", style="red")

def run_chatbot():
    """Chatbot'u başlat"""
    console.print("🤖 LLM Chatbot başlatılıyor...", style="bold cyan")
    try:
        subprocess.run([sys.executable, "real_llm_agricultural_chatbot.py"])
    except Exception as e:
        console.print(f"❌ Chatbot hatası: {e}", style="red")

def main():
    """Ana fonksiyon"""
    welcome_panel = Panel.fit(
        "🧠 **Tarımsal LLM Sistemi**\n\n"
        "🎯 **Seçenekler:**\n"
        "1️⃣ Kütüphaneleri yükle\n"
        "2️⃣ LLM'i eğit (GPT-2 fine-tuning)\n"
        "3️⃣ LLM Chatbot'u başlat\n"
        "4️⃣ Hızlı başlangıç (tümü)\n"
        "❌ Çıkış\n\n"
        "💡 **Öneri:** İlk kullanımda seçenek 4'ü deneyin!",
        title="🚀 LLM Launcher",
        style="bold green"
    )
    console.print(welcome_panel)
    
    while True:
        choice = Prompt.ask(
            "🔢 Seçiminiz",
            choices=["1", "2", "3", "4", "exit", "çıkış"],
            default="4"
        )
        
        if choice in ["exit", "çıkış"]:
            console.print("👋 Hoşçakalın!", style="bold green")
            break
        elif choice == "1":
            install_requirements()
        elif choice == "2":
            if check_requirements():
                run_training()
            else:
                console.print("❗ Önce kütüphaneleri yükleyin (seçenek 1)", style="yellow")
        elif choice == "3":
            if check_requirements():
                if Path("agricultural_gpt2").exists():
                    run_chatbot()
                else:
                    console.print("❗ Önce modeli eğitin (seçenek 2)", style="yellow")
            else:
                console.print("❗ Önce kütüphaneleri yükleyin (seçenek 1)", style="yellow")
        elif choice == "4":
            # Hızlı başlangıç
            console.print("🚀 Hızlı başlangıç başlatılıyor...", style="bold cyan")
            
            # 1. Requirements check
            if not check_requirements():
                install_requirements()
            
            # 2. Model eğit (eğer yoksa)
            if not Path("agricultural_gpt2").exists():
                console.print("📚 Model eğitimi gerekli...", style="yellow")
                run_training()
            
            # 3. Chatbot başlat
            run_chatbot()
            break

if __name__ == "__main__":
    main() 