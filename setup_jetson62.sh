#!/bin/bash

# 🚀 JetPack 6.2 Setup Script
# Agricultural BERT Classification System - Optimized for JetPack 6.2

echo "🌾 Agricultural BERT Setup for JetPack 6.2"
echo "============================================"

# Check JetPack version
echo "🔍 Checking JetPack version..."
if command -v jetson_release &> /dev/null; then
    jetson_release
else
    echo "⚠️  jetson_release not found, installing..."
    sudo apt update
    sudo apt install -y python3-pip
    pip3 install jetson-stats
fi

# System optimization for JetPack 6.2
echo "⚡ Optimizing system for JetPack 6.2..."
sudo jetson_clocks
sudo systemctl disable nvzramconfig

# Create swap file for memory optimization
echo "💾 Setting up swap file..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# Python environment setup
echo "🐍 Setting up Python environment..."
python3 -m venv agricultural_bert_jp62
source agricultural_bert_jp62/bin/activate

# Install PyTorch 2.3 for JetPack 6.2
echo "🔥 Installing PyTorch 2.3 for JetPack 6.2..."
pip3 install --upgrade pip
pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu122

# Install optimized transformers
echo "🤖 Installing optimized transformers..."
pip3 install transformers[torch]==4.36.0
pip3 install datasets==2.16.0
pip3 install accelerate==0.25.0

# Install additional ML libraries
echo "📊 Installing ML libraries..."
pip3 install scikit-learn==1.3.2
pip3 install pandas==2.1.4
pip3 install numpy==1.24.4
pip3 install matplotlib==3.8.2
pip3 install seaborn==0.13.0

# Install monitoring tools
echo "📈 Installing monitoring tools..."
pip3 install tensorboard==2.15.1
pip3 install wandb==0.16.1
pip3 install psutil==5.9.6

# Create optimized directories
echo "📁 Creating optimized directories..."
mkdir -p {results,logs,models,data,checkpoints}
mkdir -p agricultural_datasets/{train,val,test}
mkdir -p jetson_training/{configs,utils}

# Set environment variables for optimization
echo "🔧 Setting optimization environment variables..."
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128' >> ~/.bashrc
echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc

# Create requirements file for JetPack 6.2
echo "📝 Creating requirements file..."
cat > requirements_bert_jetpack62.txt << EOF
torch==2.3.0
torchvision==0.18.0
torchaudio==2.3.0
transformers[torch]==4.36.0
datasets==2.16.0
accelerate==0.25.0
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.4
matplotlib==3.8.2
seaborn==0.13.0
tensorboard==2.15.1
wandb==0.16.1
psutil==5.9.6
jetson-stats
EOF

echo "✅ JetPack 6.2 setup completed successfully!"
echo "🔄 To activate environment: source agricultural_bert_jp62/bin/activate"
echo "🚀 Ready for agricultural BERT training on JetPack 6.2!"
echo "📊 Monitor with: jtop" 