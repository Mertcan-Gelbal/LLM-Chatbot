#!/bin/bash

# 🚀 Jetson Orin Nano Setup Script
# Agricultural BERT Classification System

echo "🌾 Agricultural BERT Setup for Jetson Orin Nano"
echo "================================================"

# System update
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Python dependencies
echo "🐍 Installing Python dependencies..."
sudo apt install -y python3-pip python3-dev python3-venv

# CUDA and PyTorch dependencies
echo "⚡ Installing CUDA dependencies..."
sudo apt install -y cuda-toolkit-12-2

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv agricultural_bert_env
source agricultural_bert_env/bin/activate

# Install PyTorch for Jetson
echo "🔥 Installing PyTorch for Jetson..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers and other ML libraries
echo "🤖 Installing ML libraries..."
pip3 install transformers datasets accelerate
pip3 install scikit-learn pandas numpy matplotlib seaborn
pip3 install tensorboard wandb

# Install Jetson-specific optimizations
echo "⚡ Installing Jetson optimizations..."
pip3 install jetson-stats

# Create directories
echo "📁 Creating project directories..."
mkdir -p results logs models data

echo "✅ Setup completed successfully!"
echo "🔄 To activate environment: source agricultural_bert_env/bin/activate"
echo "🚀 Ready for agricultural BERT training!" 