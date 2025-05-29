#!/bin/bash
# Jetson Orin Nano Setup Script
# ============================
# Agricultural RAG System için optimize edilmiş kurulum

set -e

echo "🚀 Jetson Orin Nano Setup Başlıyor..."
echo "======================================"

# System info
echo "📱 Sistem Bilgileri:"
uname -a
nvidia-smi || echo "⚠️ nvidia-smi bulunamadı"

# Update system
echo "🔄 Sistem güncelleniyor..."
sudo apt update
sudo apt upgrade -y

# Essential packages
echo "📦 Temel paketler yükleniyor..."
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3-pip \
    python3-dev \
    python3-venv \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config

# CUDA development
echo "🔧 CUDA development tools..."
sudo apt install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-dev \
    libcudnn8-dev

# Python virtual environment
echo "🐍 Python virtual environment oluşturuluyor..."
python3 -m venv jetson_rag_env
source jetson_rag_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

echo "📚 PyTorch ecosystem yükleniyor..."

# PyTorch for Jetson (pre-built wheel)
TORCH_URL="https://nvidia.box.com/shared/static/rehpfc4dwsxuhpv4jgqyu5ag33a66ew.whl"
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Torchvision (compatible version)
pip install torchvision==0.15.0

# TensorRT for PyTorch
pip install torch-tensorrt

echo "🤗 Transformers ecosystem..."

# Core ML libraries
pip install \
    transformers==4.35.0 \
    tokenizers==0.14.0 \
    datasets==2.14.0 \
    accelerate==0.24.0 \
    optimum[onnxruntime]==1.13.0

# RAG specific
pip install \
    sentence-transformers==2.2.2 \
    faiss-cpu==1.7.4

echo "🔧 Jetson optimizations..."

# ONNX Runtime GPU
pip install onnxruntime-gpu==1.16.0

# Memory management
pip install \
    psutil \
    gpustat \
    nvidia-ml-py

# Scientific computing
pip install \
    numpy==1.24.3 \
    scipy==1.11.0 \
    pandas==2.1.0 \
    scikit-learn==1.3.0

# Visualization
pip install \
    matplotlib==3.7.0 \
    seaborn==0.12.0 \
    tqdm==4.66.0

# Development tools
pip install \
    jupyter \
    ipykernel \
    black \
    flake8 \
    pytest

echo "⚙️ System optimizations..."

# Jetson performance mode
sudo nvpmodel -m 0  # Max performance
sudo jetson_clocks  # Max clocks

# Memory settings
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
echo "vm.vfs_cache_pressure=50" | sudo tee -a /etc/sysctl.conf

# GPU memory settings
echo "options nvidia-drm modeset=1" | sudo tee -a /etc/modprobe.d/nvidia.conf

echo "📁 Dizin yapısı oluşturuluyor..."

# Create directory structure
mkdir -p {logs,models,checkpoints,data,results}

echo "🔍 Environment test..."

# Test CUDA
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Test Transformers
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Test FAISS
python3 -c "import faiss; print(f'FAISS version: {faiss.__version__}')"

echo "📋 Environment info kaydet..."

# Save environment info
cat > jetson_env_info.txt << EOF
Jetson Agricultural RAG Environment
==================================
Date: $(date)
Hostname: $(hostname)
Architecture: $(uname -m)
Kernel: $(uname -r)
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader || echo "Not detected")
CUDA: $(nvcc --version | grep release || echo "Not detected")
Python: $(python3 --version)
PyTorch: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")
Transformers: $(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "Not installed")
Available Memory: $(free -h | grep Mem | awk '{print $2}')
Available GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits || echo "Unknown") MB
EOF

echo "📊 Performance benchmark..."

# Quick performance test
python3 -c "
import time
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(1000, 1000).to(device)
    start = time.time()
    for i in range(100):
        y = torch.mm(x, x)
    end = time.time()
    print(f'GPU MatMul benchmark: {(end-start)*1000:.2f}ms for 100 iterations')
    print(f'GPU Memory allocated: {torch.cuda.memory_allocated()/1e6:.1f}MB')
else:
    print('CUDA not available for benchmark')
"

echo "✅ Jetson setup tamamlandı!"
echo ""
echo "🚀 Başlamak için:"
echo "   source jetson_rag_env/bin/activate"
echo "   cd jetson_training"
echo "   python jetson_train.py --gpu --mixed_precision"
echo ""
echo "📊 Environment bilgileri: jetson_env_info.txt"
echo ""
echo "💡 Performance tips:"
echo "   - sudo nvpmodel -m 0  # Max performance mode"
echo "   - sudo jetson_clocks  # Max clocks"
echo "   - Monitor: watch -n 1 nvidia-smi"

# Create activation script
cat > activate_rag.sh << 'EOF'
#!/bin/bash
# Quick activation script
source jetson_rag_env/bin/activate
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
echo "🌾 Agricultural RAG environment activated!"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
EOF

chmod +x activate_rag.sh

echo "🎯 Hızlı aktivasyon için: ./activate_rag.sh" 